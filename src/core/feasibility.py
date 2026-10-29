"""Probability-of-feasibility helpers for constrained Bayesian optimization."""

import math

import torch
from botorch.acquisition.acquisition import AcquisitionFunction


def _normal_cdf(value):
    return 0.5 * (1.0 + torch.erf(value / math.sqrt(2.0)))


def constraint_probability(mean, stddev, operator_value, limit):
    """Return P(Y satisfies the raw-scale bound) for a normal posterior."""
    stddev = stddev.clamp_min(1e-9)
    if operator_value == ">=":
        z_score = (mean - float(limit)) / stddev
    elif operator_value == "<=":
        z_score = (float(limit) - mean) / stddev
    else:  # Validation normally prevents this path.
        raise ValueError(f"Unsupported objective constraint operator: {operator_value}")
    return _normal_cdf(z_score).clamp(1e-12, 1.0)


def joint_probability_of_feasibility(models_by_name, objective_constraints, x_value):
    """Calculate independent joint PoF for every point in ``x_value``."""
    joint = None
    for constraint in objective_constraints:
        model = models_by_name[constraint["objective"]]
        posterior = model.posterior(x_value)
        probability = constraint_probability(
            posterior.mean.squeeze(-1),
            posterior.stddev.squeeze(-1),
            constraint["operator"],
            constraint["value"],
        )
        joint = probability if joint is None else joint * probability
    if joint is None:
        shape = x_value.shape[:-1]
        return torch.ones(shape, dtype=x_value.dtype, device=x_value.device)
    return joint


class ProbabilityOfFeasibilityAcquisition(AcquisitionFunction):
    """Modulate an acquisition function with raw-objective feasibility models.

    For q-batches, every proposed experiment must be feasible, so the pointwise
    probabilities are multiplied across q. For logarithmic acquisition
    functions, adding log(PoF) is equivalent to multiplying the original value.
    """

    def __init__(
        self,
        base_acquisition,
        models_by_name,
        objective_constraints,
        logarithmic=None,
    ):
        super().__init__(base_acquisition.model)
        self.base_acquisition = base_acquisition
        self.objective_names = list(models_by_name)
        self.feasibility_models = torch.nn.ModuleList(
            [models_by_name[name] for name in self.objective_names]
        )
        self.objective_constraints = list(objective_constraints)
        self.logarithmic = bool(
            getattr(base_acquisition, "_log", False)
            if logarithmic is None
            else logarithmic
        )
        self.set_X_pending(getattr(base_acquisition, "X_pending", None))

    def set_X_pending(self, X_pending=None):
        """Keep pending candidates synchronized with the wrapped acquisition."""

        super().set_X_pending(X_pending)
        if hasattr(self, "base_acquisition"):
            self.base_acquisition.set_X_pending(X_pending)

    def _models_by_name(self):
        return {
            name: self.feasibility_models[index]
            for index, name in enumerate(self.objective_names)
        }

    def forward(self, x_value):
        acquisition_value = self.base_acquisition(x_value)
        probability = joint_probability_of_feasibility(
            self._models_by_name(), self.objective_constraints, x_value
        )
        # Collapse the q dimension while retaining t-batch dimensions.
        while probability.ndim > acquisition_value.ndim:
            probability = probability.prod(dim=-1)
        if self.logarithmic:
            return acquisition_value + torch.log(probability.clamp_min(1e-12))
        return acquisition_value * probability


def candidate_feasibility_report(
    models_by_name, objective_constraints, candidates, threshold=0.5
):
    """Return per-candidate probabilities and raw posterior predictions."""
    if not objective_constraints:
        return {
            "constrained": False,
            "all_likely_feasible": True,
            "candidates": [],
        }

    reports = []
    with torch.no_grad():
        for candidate in candidates:
            point = candidate.reshape(1, 1, -1)
            constraint_reports = []
            joint_probability = 1.0
            mean_satisfied = True
            for constraint in objective_constraints:
                model = models_by_name[constraint["objective"]]
                posterior = model.posterior(point)
                mean = float(posterior.mean.squeeze().item())
                stddev = float(posterior.stddev.squeeze().item())
                probability = float(
                    constraint_probability(
                        posterior.mean.squeeze(),
                        posterior.stddev.squeeze(),
                        constraint["operator"],
                        constraint["value"],
                    ).item()
                )
                joint_probability *= probability
                expected_satisfied = (
                    mean >= constraint["value"] - 1e-7
                    if constraint["operator"] == ">="
                    else mean <= constraint["value"] + 1e-7
                )
                mean_satisfied = mean_satisfied and expected_satisfied
                constraint_reports.append(
                    {
                        "objective": constraint["objective"],
                        "operator": constraint["operator"],
                        "limit": float(constraint["value"]),
                        "predicted": mean,
                        "std_error": stddev,
                        "probability": probability,
                        "expected_satisfied": expected_satisfied,
                    }
                )
            reports.append(
                {
                    "probability": joint_probability,
                    "likely_feasible": mean_satisfied
                    and joint_probability >= float(threshold),
                    "constraints": constraint_reports,
                }
            )

    return {
        "constrained": True,
        "all_likely_feasible": all(
            report["likely_feasible"] for report in reports
        ),
        "candidates": reports,
    }
