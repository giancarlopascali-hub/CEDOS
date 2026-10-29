"""Resource and acquisition policy for bounded Bayesian optimization jobs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os
from typing import Any, Sequence


class BOCalculationCancelled(RuntimeError):
    """Raised when a BO calculation is cancelled by the user."""


class BOTimeoutExceeded(TimeoutError):
    """Raised when a BO phase or the complete job exceeds its deadline."""


class BOModelFitError(RuntimeError):
    """Raised when a required GP model cannot be fitted safely."""


class FeasibleInitializationError(ValueError):
    """Raised when constrained optimization cannot find feasible starts."""


VALID_MULTI_OBJECTIVE_METHODS = {
    "auto",
    "qLogNEHVI",
    "qLogNParEGO",
    "scalarized-qLogEI",
}
VALID_BO_DATA_MODES = {"auto", "full", "representative"}


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")
    return value


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    raw = os.environ.get(name, str(default))
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number.") from exc
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")
    return value


@dataclass(frozen=True)
class BOBudget:
    """Effective numerical and wall-clock limits for one BO calculation."""

    mc_samples: int
    num_restarts: int
    raw_samples: int
    fit_maxiter: int
    fit_maxls: int
    fit_retries: int
    fit_phase_timeout_sec: float
    acq_maxiter: int
    candidate_timeout_sec: float
    total_job_timeout_sec: float
    nehvi_max_objectives: int
    nehvi_alpha: float
    feasible_ic_max_draws: int
    feasible_ic_chunk_size: int
    discrete_eval_batch_size: int
    max_batch_size: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BOLargeDataSettings:
    """Server-owned thresholds for exact-GP workload and row reduction."""

    review_risk_score: float
    unsafe_risk_score: float
    representative_target_risk_score: float
    safe_memory_mb: float
    representative_min_rows: int
    representative_max_rows: int
    target_fidelity_score: float
    representative_selection_attempts: int
    representative_selection_search_timeout_sec: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_bo_budget(is_cuda: bool = False) -> BOBudget:
    """Load bounded deployment settings without accepting request-level overrides."""

    suffix = "GPU" if is_cuda else "CPU"
    return BOBudget(
        mc_samples=_env_int(f"BO_MC_SAMPLES_{suffix}", 128 if is_cuda else 64, 8, 4096),
        num_restarts=_env_int(
            f"BO_NUM_RESTARTS_{suffix}", 16 if is_cuda else 8, 1, 128
        ),
        raw_samples=_env_int(
            f"BO_RAW_SAMPLES_{suffix}", 256 if is_cuda else 128, 16, 8192
        ),
        fit_maxiter=_env_int("BO_MODEL_FIT_MAXITER", 75, 5, 1000),
        fit_maxls=_env_int("BO_MODEL_FIT_MAXLS", 50, 10, 500),
        fit_retries=_env_int("BO_MODEL_FIT_RETRIES", 1, 0, 3),
        fit_phase_timeout_sec=_env_float(
            "BO_MODEL_FIT_PHASE_TIMEOUT_SEC", 60.0, 1.0, 1800.0
        ),
        acq_maxiter=_env_int("BO_ACQ_OPT_MAXITER", 75, 5, 1000),
        candidate_timeout_sec=_env_float(
            "BO_CANDIDATE_TIMEOUT_SEC", 20.0, 1.0, 600.0
        ),
        total_job_timeout_sec=_env_float(
            "BO_TOTAL_JOB_TIMEOUT_SEC", 300.0, 10.0, 7200.0
        ),
        nehvi_max_objectives=_env_int("BO_NEHVI_MAX_OBJECTIVES", 3, 2, 10),
        nehvi_alpha=_env_float("BO_NEHVI_ALPHA", 0.01, 0.0, 1.0),
        feasible_ic_max_draws=_env_int(
            "BO_FEASIBLE_IC_MAX_DRAWS", 8192, 128, 1_000_000
        ),
        feasible_ic_chunk_size=_env_int(
            "BO_FEASIBLE_IC_CHUNK_SIZE", 512, 16, 65_536
        ),
        discrete_eval_batch_size=_env_int(
            "BO_DISCRETE_EVAL_BATCH_SIZE", 1024, 16, 65_536
        ),
        max_batch_size=_env_int("BO_MAX_BATCH_SIZE", 20, 1, 100),
    )


def load_bo_large_data_settings() -> BOLargeDataSettings:
    """Load deployment safety thresholds that requests cannot increase."""

    review = _env_float("BO_WORKLOAD_REVIEW_SCORE", 0.55, 0.05, 10.0)
    unsafe = _env_float("BO_WORKLOAD_UNSAFE_SCORE", 1.0, review, 20.0)
    target = _env_float(
        "BO_REPRESENTATIVE_TARGET_SCORE", 0.95, 0.05, unsafe
    )
    minimum_rows = _env_int("BO_REPRESENTATIVE_MIN_ROWS", 20, 5, 10_000)
    maximum_rows = _env_int("BO_REPRESENTATIVE_MAX_ROWS", 250, 5, 100_000)
    if maximum_rows < minimum_rows:
        raise ValueError(
            "BO_REPRESENTATIVE_MAX_ROWS must be at least "
            "BO_REPRESENTATIVE_MIN_ROWS."
        )
    fidelity_environment_name = "BO_SUBSET_TARGET_FIDELITY"
    if (
        fidelity_environment_name not in os.environ
        and "BO_SUBSET_MIN_FIDELITY" in os.environ
    ):
        # Backward compatibility for deployments created before fidelity became
        # a target with a best-safe-subset fallback rather than a hard gate.
        fidelity_environment_name = "BO_SUBSET_MIN_FIDELITY"
    return BOLargeDataSettings(
        review_risk_score=review,
        unsafe_risk_score=unsafe,
        representative_target_risk_score=target,
        safe_memory_mb=_env_float("BO_SAFE_MEMORY_MB", 6144.0, 128.0, 1_000_000.0),
        representative_min_rows=minimum_rows,
        representative_max_rows=maximum_rows,
        target_fidelity_score=_env_float(
            fidelity_environment_name, 90.0, 0.0, 100.0
        ),
        representative_selection_attempts=_env_int(
            "BO_SUBSET_SELECTION_ATTEMPTS", 3, 1, 8
        ),
        representative_selection_search_timeout_sec=_env_float(
            "BO_SUBSET_SELECTION_TIMEOUT_SEC", 12.0, 0.1, 120.0
        ),
    )


def normalized_objective_weights(values: Sequence[float]) -> list[float]:
    """Validate positive finite objective weights and normalize them to sum to one."""

    if not values:
        raise ValueError("At least one objective weight is required.")
    parsed = [float(value) for value in values]
    if any(not math.isfinite(value) or value <= 0 for value in parsed):
        raise ValueError("Objective importance values must be finite and greater than zero.")
    total = sum(parsed)
    return [value / total for value in parsed]


def weights_are_equal(values: Sequence[float]) -> bool:
    """Treat only numerical round-off as equal; deliberate differences are priorities."""

    normalized = normalized_objective_weights(values)
    first = normalized[0]
    return all(math.isclose(value, first, rel_tol=1e-6, abs_tol=1e-9) for value in normalized)


@dataclass(frozen=True)
class BOAcquisitionPolicy:
    key: str
    label: str
    reason: str
    recommended_key: str
    requested_key: str
    is_override: bool = False
    warning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_multi_objective_method(value: Any) -> str:
    """Return a canonical multi-objective method key."""

    raw = str(value or "auto").strip()
    aliases = {
        "auto": "auto",
        "qlognehvi": "qLogNEHVI",
        "nehvi": "qLogNEHVI",
        "qlognparego": "qLogNParEGO",
        "nparego": "qLogNParEGO",
        "scalarized-qlogei": "scalarized-qLogEI",
        "scalarized_ei": "scalarized-qLogEI",
        "scalarized-ei": "scalarized-qLogEI",
    }
    canonical = aliases.get(raw.lower())
    if canonical not in VALID_MULTI_OBJECTIVE_METHODS:
        choices = ", ".join(sorted(VALID_MULTI_OBJECTIVE_METHODS))
        raise ValueError(f"Unknown multi-objective method. Choose one of: {choices}.")
    return canonical


def validate_bo_data_mode(value: Any) -> str:
    """Validate the requested exact-GP data handling mode."""

    mode = str(value or "auto").strip().lower()
    if mode not in VALID_BO_DATA_MODES:
        choices = ", ".join(sorted(VALID_BO_DATA_MODES))
        raise ValueError(f"Unknown BO data mode. Choose one of: {choices}.")
    return mode


def _recommended_multi_objective_key(
    objective_count: int,
    weights: Sequence[float],
    nehvi_max_objectives: int,
) -> str:
    if not weights_are_equal(weights):
        return "scalarized-qLogEI"
    if objective_count <= nehvi_max_objectives:
        return "qLogNEHVI"
    return "qLogNParEGO"


def _policy_details(key: str, objective_count: int, weights: Sequence[float]):
    if key == "qLogNEHVI":
        return (
            "Approximate log hypervolume improvement",
            f"Pareto hypervolume discovery with {objective_count} objectives.",
        )
    if key == "qLogNParEGO":
        return (
            "Randomized Pareto scalarization",
            (
                f"Equal-weight discovery with {objective_count} objectives avoids "
                "high-dimensional hypervolume partitioning."
            ),
        )
    return (
        "Priority-weighted log expected improvement",
        (
            "Objective importance values define a fixed weighted compromise."
            if not weights_are_equal(weights)
            else "A fixed equal-weight compromise was selected by the user."
        ),
    )


def select_bo_policy(
    objective_count: int,
    weights: Sequence[float],
    acq_type: str,
    nehvi_max_objectives: int,
    requested_method: str = "auto",
) -> BOAcquisitionPolicy:
    """Select the acquisition semantics independently from its implementation."""

    if objective_count < 1:
        raise ValueError("At least one objective is required.")
    normalized_objective_weights(weights)
    if objective_count == 1:
        requested = validate_multi_objective_method(requested_method)
        if requested != "auto":
            raise ValueError(
                "Multi-objective acquisition methods require at least two objectives."
            )
        if str(acq_type).upper() == "LCB":
            return BOAcquisitionPolicy(
                key="qUCB",
                label="Confidence-bound optimization",
                reason="Single-objective confidence-bound acquisition selected by the user.",
                recommended_key="qUCB",
                requested_key="auto",
            )
        return BOAcquisitionPolicy(
            key="qLogEI",
            label="Log expected improvement",
            reason="Single-objective expected-improvement optimization.",
            recommended_key="qLogEI",
            requested_key="auto",
        )

    requested = validate_multi_objective_method(requested_method)
    recommended = _recommended_multi_objective_key(
        objective_count, weights, nehvi_max_objectives
    )
    if requested == "qLogNParEGO" and not weights_are_equal(weights):
        raise ValueError(
            "qLogNParEGO is unavailable when objective importance values are "
            "unequal. Use Auto, qLogNEHVI, or scalarized qLogEI."
        )
    effective = recommended if requested == "auto" else requested
    label, reason = _policy_details(effective, objective_count, weights)
    is_override = requested != "auto" and effective != recommended
    warning = None
    if is_override:
        if effective == "qLogNEHVI":
            warning = (
                "Manual qLogNEHVI can be substantially slower as the number of "
                "objectives grows."
            )
        elif effective == "scalarized-qLogEI":
            warning = (
                "Scalarized qLogEI searches for one weighted compromise instead "
                "of broadly mapping the Pareto trade-off surface."
            )
        else:
            warning = "A non-default multi-objective method was selected."
        reason = f"Manual override selected. {reason}"
    return BOAcquisitionPolicy(
        key=effective,
        label=label,
        reason=reason,
        recommended_key=recommended,
        requested_key=requested,
        is_override=is_override,
        warning=warning,
    )


@dataclass(frozen=True)
class BOWorkloadAssessment:
    """Transparent planning estimate for an exact-GP BO request."""

    row_count: int
    feature_count: int
    objective_count: int
    model_count: int
    batch_size: int
    method: str
    time_risk_score: float
    memory_risk_score: float
    risk_score: float
    tier: str
    controls_required: bool
    estimated_seconds: float
    estimated_memory_mb: float
    recommended_training_rows: int
    representative_feasible: bool
    formula_version: str = "exact-gp-v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _workload_method_factor(method: str, objective_count: int) -> float:
    if method in {"qLogEI", "qUCB"}:
        return 0.35
    if method == "scalarized-qLogEI":
        return 0.65
    if method == "qLogNParEGO":
        return 1.0
    if method == "qLogNEHVI":
        # Hypervolume partitioning grows sharply beyond the recommended
        # low-objective range. The cap avoids numerical overflow in the planner.
        return min(32.0, 1.35 * (1.8 ** max(0, objective_count - 3)))
    return 1.0


def _workload_components(
    *,
    row_count: int,
    feature_count: int,
    objective_count: int,
    model_count: int,
    batch_size: int,
    method: str,
    per_model_rows: Sequence[int],
    budget: BOBudget,
    settings: BOLargeDataSettings,
) -> tuple[float, float, float, float]:
    if row_count <= 0:
        return 0.0, 0.0, 0.0, 0.0
    feature_factor = max(0.5, (max(1, feature_count) / 16.0) ** 0.6)
    fit_score = feature_factor * sum(
        (max(0, int(count)) / 700.0) ** 3 for count in per_model_rows
    )
    observed_rows = max([row_count, *[int(value) for value in per_model_rows]])
    model_factor = max(0.15, math.sqrt(max(1, model_count) / 7.0))
    acquisition_score = (
        0.65
        * (max(1, batch_size) / 10.0)
        * ((max(1, observed_rows) / 69.0) ** 0.2)
        * math.sqrt(feature_factor)
        * model_factor
        * _workload_method_factor(method, objective_count)
    )
    time_score = fit_score + acquisition_score

    # Approximate native double-precision covariance storage plus factorization
    # workspaces for independently fitted exact GP models.
    memory_mb = 96.0 * sum(
        max(0, int(count)) ** 2 for count in per_model_rows
    ) / (1024.0 ** 2)
    memory_score = memory_mb / settings.safe_memory_mb
    estimated_seconds = time_score * budget.total_job_timeout_sec
    return time_score, memory_score, estimated_seconds, memory_mb


def assess_bo_workload(
    *,
    row_count: int,
    feature_count: int,
    objective_count: int,
    batch_size: int,
    method: str,
    budget: BOBudget,
    per_model_rows: Sequence[int] | None = None,
    model_count: int | None = None,
    settings: BOLargeDataSettings | None = None,
) -> BOWorkloadAssessment:
    """Estimate exact-GP risk and derive a bounded representative row cap.

    The score combines the cubic exact-GP fit term, acquisition work, and a
    native-memory estimate. It is a conservative deployment guard, not a
    promise of wall-clock runtime.
    """

    parsed_rows = max(0, int(row_count))
    parsed_features = max(1, int(feature_count))
    parsed_objectives = max(1, int(objective_count))
    parsed_batch = max(1, int(batch_size))
    effective_settings = settings or load_bo_large_data_settings()
    if per_model_rows is None:
        effective_model_count = max(parsed_objectives, int(model_count or 0))
        counts = [parsed_rows] * effective_model_count
    else:
        counts = [max(0, min(parsed_rows, int(value))) for value in per_model_rows]
        effective_model_count = max(
            parsed_objectives, int(model_count or len(counts)), len(counts)
        )
        if len(counts) < effective_model_count:
            counts.extend([parsed_rows] * (effective_model_count - len(counts)))

    time_score, memory_score, seconds, memory_mb = _workload_components(
        row_count=parsed_rows,
        feature_count=parsed_features,
        objective_count=parsed_objectives,
        model_count=effective_model_count,
        batch_size=parsed_batch,
        method=method,
        per_model_rows=counts,
        budget=budget,
        settings=effective_settings,
    )
    risk_score = max(time_score, memory_score)
    if risk_score >= effective_settings.unsafe_risk_score:
        tier = "unsafe"
    elif risk_score >= effective_settings.review_risk_score:
        tier = "review"
    else:
        tier = "safe"

    dynamic_minimum = max(
        effective_settings.representative_min_rows,
        parsed_features + 2,
        parsed_objectives * 3,
    )
    maximum_candidate = min(parsed_rows, effective_settings.representative_max_rows)
    recommended_rows = parsed_rows
    representative_feasible = True
    if risk_score > effective_settings.representative_target_risk_score:
        if maximum_candidate < dynamic_minimum:
            recommended_rows = maximum_candidate
            representative_feasible = False
        else:
            def candidate_risk(candidate_rows: int) -> float:
                candidate_counts = [
                    min(candidate_rows, count) if count else 0 for count in counts
                ]
                candidate_time, candidate_memory, _, _ = _workload_components(
                    row_count=candidate_rows,
                    feature_count=parsed_features,
                    objective_count=parsed_objectives,
                    model_count=effective_model_count,
                    batch_size=parsed_batch,
                    method=method,
                    per_model_rows=candidate_counts,
                    budget=budget,
                    settings=effective_settings,
                )
                return max(candidate_time, candidate_memory)

            if candidate_risk(dynamic_minimum) > effective_settings.representative_target_risk_score:
                recommended_rows = dynamic_minimum
                representative_feasible = False
            else:
                low, high = dynamic_minimum, maximum_candidate
                while low < high:
                    midpoint = (low + high + 1) // 2
                    if candidate_risk(midpoint) <= effective_settings.representative_target_risk_score:
                        low = midpoint
                    else:
                        high = midpoint - 1
                recommended_rows = low

    return BOWorkloadAssessment(
        row_count=parsed_rows,
        feature_count=parsed_features,
        objective_count=parsed_objectives,
        model_count=effective_model_count,
        batch_size=parsed_batch,
        method=method,
        time_risk_score=round(time_score, 4),
        memory_risk_score=round(memory_score, 4),
        risk_score=round(risk_score, 4),
        tier=tier,
        controls_required=(
            risk_score >= effective_settings.representative_target_risk_score
        ),
        estimated_seconds=round(seconds, 1),
        estimated_memory_mb=round(memory_mb, 1),
        recommended_training_rows=int(recommended_rows),
        representative_feasible=representative_feasible,
    )


@dataclass
class BOOptimizationResult:
    suggestions: list[dict[str, Any]] | None
    performance: dict[str, Any]
    constraint_status: dict[str, Any]
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
