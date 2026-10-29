import ast
import re
import itertools
import time
import numpy as np
import pandas as pd
import torch
from gpytorch.mlls import ExactMarginalLogLikelihood, SumMarginalLogLikelihood
from gpytorch.kernels import RBFKernel, MaternKernel, ScaleKernel

from botorch.models import SingleTaskGP, MixedSingleTaskGP, ModelListGP
from botorch.models.transforms import Standardize, Normalize
from botorch.fit import fit_gpytorch_mll, fit_gpytorch_mll_scipy

# Acquisition functions
from botorch.acquisition import qLogExpectedImprovement, qUpperConfidenceBound
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.parego import qLogNParEGO

from botorch.acquisition.multi_objective.objective import WeightedMCMultiOutputObjective
from botorch.acquisition.objective import ScalarizedPosteriorTransform

from botorch.optim import optimize_acqf_discrete
from botorch.optim.core import OptimizationStatus
from botorch.optim.optimize import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from botorch.sampling.normal import SobolQMCNormalSampler

from src.core.constraints import (
    CONSTRAINT_TOLERANCE,
    check_constraints,
    check_objective_constraints,
    compile_numerical_expression,
    encode_categorical_training_series,
    get_categorical_choices,
)
from src.core.feasibility import (
    ProbabilityOfFeasibilityAcquisition,
    candidate_feasibility_report,
)
from src.core.bo_policy import (
    BOCalculationCancelled,
    BOModelFitError,
    BOOptimizationResult,
    BOTimeoutExceeded,
    FeasibleInitializationError,
    assess_bo_workload,
    load_bo_budget,
    load_bo_large_data_settings,
    normalized_objective_weights,
    select_bo_policy,
    validate_bo_data_mode,
)
from src.core.bo_sampling import select_representative_bo_rows


class SurfaceCalculationCancelled(RuntimeError):
    """Raised when a response-surface job is cancelled by the user."""


STANDARD_BO_MAX_BETA = 10.0
BO_TENSOR_DTYPE = torch.float64


def _check_bo_runtime(cancel_event=None, deadline=None):
    if cancel_event is not None and cancel_event.is_set():
        raise BOCalculationCancelled("Bayesian optimization was cancelled by the user.")
    if deadline is not None and time.monotonic() >= deadline:
        raise BOTimeoutExceeded("Bayesian optimization exceeded its wall-clock budget.")


def _remaining_bo_seconds(deadline):
    if deadline is None:
        return float("inf")
    return max(0.0, deadline - time.monotonic())


def _fit_required_model(
    mll,
    *,
    label,
    maxiter,
    timeout_sec,
    maxls=50,
    retries=1,
    cancel_event=None,
    deadline=None,
):
    """Fit and validate one model through BoTorch's timed SciPy path.

    L-BFGS-B can report ``ABNORMAL`` after retaining a perfectly finite best
    point, particularly with single-precision arithmetic. BO uses double
    precision, allows a longer line search, and retries that narrow failure
    mode once. A persistent ``ABNORMAL`` result is accepted only when its loss,
    parameters, posterior mean, and posterior variance are all finite.
    """

    _check_bo_runtime(cancel_event, deadline)
    operation_deadline = time.monotonic() + float(timeout_sec)
    if deadline is not None:
        operation_deadline = min(operation_deadline, float(deadline))
    effective_timeout = operation_deadline - time.monotonic()
    if effective_timeout < 1.0:
        raise BOTimeoutExceeded(f"No time remains to fit {label}.")
    try:
        attempt_history = []
        fit_result = None
        max_attempts = 1 + max(0, int(retries))
        for attempt_index in range(max_attempts):
            _check_bo_runtime(cancel_event, deadline)
            attempt_timeout = operation_deadline - time.monotonic()
            if attempt_timeout < 1.0:
                if fit_result is None:
                    raise BOTimeoutExceeded(f"No time remains to fit {label}.")
                break
            attempt_maxls = int(maxls) * (attempt_index + 1)
            fit_result = fit_gpytorch_mll_scipy(
                mll,
                options={
                    "maxiter": int(maxiter),
                    "maxls": attempt_maxls,
                },
                timeout_sec=attempt_timeout,
            )
            attempt_history.append({
                "attempt": attempt_index + 1,
                "status": fit_result.status.name.lower(),
                "message": str(fit_result.message),
                "step": int(fit_result.step),
                "maxls": attempt_maxls,
            })
            if fit_result.status in {
                OptimizationStatus.SUCCESS,
                OptimizationStatus.STOPPED,
            }:
                break
            if "ABNORMAL" not in str(fit_result.message).upper():
                break

        if fit_result is None:  # Defensive; the loop always assigns or raises.
            raise RuntimeError("optimizer did not return a result")
        is_abnormal = (
            fit_result.status is OptimizationStatus.FAILURE
            and "ABNORMAL" in str(fit_result.message).upper()
        )
        if fit_result.status not in {
            OptimizationStatus.SUCCESS,
            OptimizationStatus.STOPPED,
        } and not is_abnormal:
            raise RuntimeError(
                f"optimizer stopped with {fit_result.status.name}: "
                f"{fit_result.message}"
            )

        try:
            objective_value = float(fit_result.fval)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("optimizer returned an invalid objective value") from exc
        if not np.isfinite(objective_value):
            raise RuntimeError("optimizer returned a non-finite objective value")
        for parameter_name, parameter in mll.named_parameters():
            if not torch.isfinite(parameter.detach()).all():
                raise RuntimeError(
                    f"model parameter '{parameter_name}' contains non-finite values"
                )

        mll.eval()
        model = mll.model.eval()
        train_inputs = getattr(model, "train_inputs", None)
        if train_inputs:
            with torch.no_grad():
                posterior = model.posterior(train_inputs[0])
            if not torch.isfinite(posterior.mean).all():
                raise RuntimeError("posterior mean contains non-finite values")
            if not torch.isfinite(posterior.variance).all():
                raise RuntimeError("posterior variance contains non-finite values")
    except (BOCalculationCancelled, BOTimeoutExceeded):
        raise
    except Exception as exc:
        raise BOModelFitError(f"Failed to fit {label}: {exc}") from exc
    _check_bo_runtime(cancel_event, deadline)
    return {
        "label": label,
        "status": fit_result.status.name.lower(),
        "degraded": (
            fit_result.status is OptimizationStatus.STOPPED or is_abnormal
        ),
        "accepted_after_validation": bool(is_abnormal),
        "message": str(fit_result.message),
        "step": int(fit_result.step),
        "attempts": len(attempt_history),
        "attempt_history": attempt_history,
        "dtype": str(train_inputs[0].dtype) if train_inputs else None,
        "runtime_seconds": (
            round(float(fit_result.runtime), 3)
            if fit_result.runtime is not None
            else None
        ),
    }


def validate_exploration(value):
    """Return a finite exploration setting constrained to the public 0-1 scale."""
    try:
        exploration = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Exploration must be a number between 0 and 1.") from exc
    if not np.isfinite(exploration) or not 0.0 <= exploration <= 1.0:
        raise ValueError("Exploration must be a number between 0 and 1.")
    return exploration


def exploration_to_beta(exploration, max_beta=STANDARD_BO_MAX_BETA):
    """Map 0 to pure exploitation and 1 to the configured maximum exploration."""
    return validate_exploration(exploration) * float(max_beta)


def generate_feasible_initial_conditions(
    bounds_t,
    q,
    num_restarts,
    p_constraints,
    *,
    max_draws=8192,
    chunk_size=512,
    cancel_event=None,
    deadline=None,
):
    """Return feasible restart points without ever falling back to invalid points."""

    d = bounds_t.shape[1]
    needed = int(num_restarts) * int(q)
    feasible_points = []
    draws = 0
    while draws < int(max_draws):
        _check_bo_runtime(cancel_event, deadline)
        draw_count = min(int(chunk_size), int(max_draws) - draws)
        samples = draw_sobol_samples(bounds=bounds_t, n=draw_count, q=1).squeeze(-2)
        draws += draw_count
        mask = torch.ones(samples.shape[0], dtype=torch.bool, device=samples.device)
        for constraint_func, _ in p_constraints:
            values = constraint_func(samples)
            mask &= values >= 0
        if mask.any():
            feasible_points.append(samples[mask])
            all_feasible = torch.unique(torch.cat(feasible_points, dim=0), dim=0)
            if all_feasible.shape[0] >= needed:
                return all_feasible[:needed].view(num_restarts, q, d)

    feasible_count = (
        torch.unique(torch.cat(feasible_points, dim=0), dim=0).shape[0]
        if feasible_points
        else 0
    )
    raise FeasibleInitializationError(
        "Unable to find enough feasible optimizer starting points "
        f"({feasible_count} found, {needed} required after {draws} draws). "
        "Review or relax the numerical feature constraints."
    )


def clean_dataframe(df, features_config, objectives_config):
    # 1. Feature filtering: Drop rows missing continuous/discrete numerical features.
    # For categorical features, treat missing/empty as 'None' level rather than dropping rows.
    df_clean = df.copy()
    valid_feat_mask = pd.Series(True, index=df_clean.index)
    for f in features_config:
        name = f['name']
        if name not in df_clean.columns:
            continue
        col_series = df_clean[name]
        if f.get('type') in ['continuous', 'discrete', 'regular']:
            num_col = pd.to_numeric(col_series, errors='coerce')
            valid_feat_mask = valid_feat_mask & num_col.notna()
        else:
            df_clean[name] = col_series.fillna('None').astype(str).replace(r'^\s*$', 'None', regex=True)
            
    df_clean = df_clean[valid_feat_mask].copy()
    
    if not objectives_config:
        return df_clean
        
    # 2. Objective filtering: Differentiate single vs multi-objective
    obj_df_numeric = pd.DataFrame()
    for obj in objectives_config:
        if obj['name'] in df_clean.columns:
            obj_df_numeric[obj['name']] = pd.to_numeric(df_clean[obj['name']], errors='coerce')
        else:
            obj_df_numeric[obj['name']] = pd.Series(np.nan, index=df_clean.index)
            
    if len(objectives_config) == 1:
        # Single-objective case: drop any row missing the objective value
        valid_obj_mask = obj_df_numeric[objectives_config[0]['name']].notna()
    else:
        # Multi-objective case: drop rows missing ALL objectives
        valid_obj_mask = obj_df_numeric.notna().any(axis=1)
        
    df_clean = df_clean[valid_obj_mask].copy()
    return df_clean


def analyze_objective_data(df, objectives_config, feature_count):
    """Return serializable objective statistics and non-blocking quality notes."""

    statistics = {}
    warnings = []
    numeric_columns = {}
    recommended_minimum = max(3, int(feature_count) + 1)

    for objective in objectives_config:
        name = objective["name"]
        values = pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)
        finite_values = values[np.isfinite(values)]
        finite_count = int(finite_values.size)
        unique_count = int(np.unique(finite_values).size)
        missing_count = int(values.size - finite_count)
        numeric_columns[name] = values
        statistics[name] = {
            "finite_count": finite_count,
            "missing_count": missing_count,
            "unique_count": unique_count,
            "minimum": float(finite_values.min()) if finite_count else None,
            "maximum": float(finite_values.max()) if finite_count else None,
            "standard_deviation": (
                float(np.std(finite_values)) if finite_count else None
            ),
        }
        if missing_count:
            warnings.append(
                f"Objective '{name}' has {missing_count} missing value(s); "
                f"its model will use the remaining {finite_count} row(s)."
            )
        if finite_count and finite_count < recommended_minimum:
            warnings.append(
                f"Objective '{name}' has only {finite_count} finite observation(s) "
                f"for {feature_count} feature(s); its surrogate may be unreliable."
            )
        if finite_count and unique_count <= 1:
            warnings.append(
                f"Objective '{name}' is constant and cannot influence optimization; "
                "consider excluding it or representing it as a constraint."
            )

    redundant_pairs = []
    objective_names = [objective["name"] for objective in objectives_config]
    for left_index, left_name in enumerate(objective_names):
        left_values = numeric_columns[left_name]
        for right_name in objective_names[left_index + 1:]:
            right_values = numeric_columns[right_name]
            finite = np.isfinite(left_values) & np.isfinite(right_values)
            if int(finite.sum()) < 3:
                continue
            left_overlap = left_values[finite]
            right_overlap = right_values[finite]
            if np.unique(left_overlap).size <= 1 or np.unique(right_overlap).size <= 1:
                continue
            correlation = float(np.corrcoef(left_overlap, right_overlap)[0, 1])
            if np.isfinite(correlation) and abs(correlation) >= 0.999:
                pair = {
                    "left": left_name,
                    "right": right_name,
                    "correlation": correlation,
                }
                redundant_pairs.append(pair)
                warnings.append(
                    f"Objectives '{left_name}' and '{right_name}' are almost perfectly "
                    f"correlated (r={correlation:.4f}); consider whether both are needed."
                )

    return {
        "objectives": statistics,
        "recommended_minimum_observations": recommended_minimum,
        "redundant_pairs": redundant_pairs,
    }, warnings


def infer_decimal_precision(col_series: pd.Series = None, range_str: str = '') -> int:
    """
    Return the maximum number of decimal places seen in col_series or range_str.
    Fallback chain:
      1. Maximum decimals across non-null dataset values
      2. Maximum decimals seen in range_str tokens (if provided)
      3. 1  (safe default – avoids integer-only snapping when no decimals present)
    """
    max_dec = 0
    if col_series is not None and hasattr(col_series, 'empty') and not col_series.empty:
        for raw in col_series.dropna().astype(str):
            m = re.search(r'\.(\d+)$', raw.strip())
            if m:
                max_dec = max(max_dec, len(m.group(1)))
    if max_dec == 0 and range_str:
        for tok in re.findall(r'[-+]?\d*\.?\d+', range_str):
            m = re.search(r'\.(\d+)$', tok)
            if m:
                max_dec = max(max_dec, len(m.group(1)))
    return max_dec if max_dec > 0 else 1


def _eval_constraint_arithmetic(node, x_value, placeholder_indices):
    """Evaluate the arithmetic side of a validated feature constraint on tensors."""

    if isinstance(node, ast.Expression):
        return _eval_constraint_arithmetic(node.body, x_value, placeholder_indices)
    if isinstance(node, ast.Constant):
        return float(node.value)
    if isinstance(node, ast.Num):  # pragma: no cover - Python < 3.8
        return float(node.n)
    if isinstance(node, ast.Name):
        return x_value[..., placeholder_indices[node.id]]
    if isinstance(node, ast.UnaryOp):
        operand = _eval_constraint_arithmetic(
            node.operand, x_value, placeholder_indices
        )
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return operand
    if isinstance(node, ast.BinOp):
        left = _eval_constraint_arithmetic(node.left, x_value, placeholder_indices)
        right = _eval_constraint_arithmetic(node.right, x_value, placeholder_indices)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return left ** right
    raise ValueError(
        f"Unsupported numerical constraint element: {type(node).__name__}."
    )


def _constraint_margin(left, right, operator_node):
    """Convert a comparison to BoTorch's required non-negative margin."""

    tolerance = float(CONSTRAINT_TOLERANCE)
    if isinstance(operator_node, ast.LtE):
        return right - left
    if isinstance(operator_node, ast.Lt):
        return right - left - tolerance
    if isinstance(operator_node, ast.GtE):
        return left - right
    if isinstance(operator_node, ast.Gt):
        return left - right - tolerance
    if isinstance(operator_node, ast.Eq):
        return tolerance - torch.abs(left - right)
    if isinstance(operator_node, ast.NotEq):
        return torch.abs(left - right) - tolerance
    raise ValueError(
        f"Unsupported numerical constraint comparison: {type(operator_node).__name__}."
    )


def compile_botorch_feature_constraints(numerical_rules, features_config):
    """Compile validated UI equations into intra-point BoTorch constraints.

    The Boolean in BoTorch's ``(callable, bool)`` tuple describes whether the
    callable is intra-point, not whether it represents equality. Every feature
    equation here applies independently to one proposed experiment, so it is
    always ``True``. Equality is represented by a tolerance margin.
    """

    feature_names = [feature["name"] for feature in features_config]
    feature_indices = {name: index for index, name in enumerate(feature_names)}
    compiled = []
    for expression in numerical_rules or []:
        tree, placeholders = compile_numerical_expression(expression, feature_names)
        comparison = tree.body
        if not isinstance(comparison, ast.Compare):
            raise ValueError(
                f"Numerical constraint '{expression}' must be a comparison."
            )
        placeholder_indices = {
            placeholder: feature_indices[feature_name]
            for placeholder, feature_name in placeholders.items()
        }
        left_node = comparison.left
        for operator_node, right_node in zip(
            comparison.ops, comparison.comparators
        ):
            def constraint_function(
                x_value,
                lhs=left_node,
                rhs=right_node,
                operator_value=operator_node,
                mapping=placeholder_indices,
            ):
                left = _eval_constraint_arithmetic(lhs, x_value, mapping)
                right = _eval_constraint_arithmetic(rhs, x_value, mapping)
                return _constraint_margin(left, right, operator_value)

            compiled.append((constraint_function, True))
            left_node = right_node
    return compiled

def run_bo_optimization(
    df,
    features_config,
    objectives_config,
    tweaks,
    device,
    compute_device,
    *,
    progress_callback=None,
    cancel_event=None,
    deadline=None,
    compute_performance=False,
    random_seed=1729,
):
    """Run bounded BO and return a serializable, named result.

    Candidate generation is deliberately sequential. Each candidate gets a
    fresh acquisition with all earlier candidates supplied as ``X_pending`` and
    an independent optimizer timeout. The outer process supervisor provides the
    hard-stop boundary; these checks make normal cancellation and deadlines
    cooperative and fast between native calls.
    """

    started = time.monotonic()
    is_cuda = str(getattr(device, "type", device)).lower().startswith("cuda")
    budget = load_bo_budget(is_cuda=is_cuda)
    if deadline is None:
        deadline = started + budget.total_job_timeout_sec
    else:
        deadline = min(float(deadline), started + budget.total_job_timeout_sec)

    try:
        seed = int(random_seed)
    except (TypeError, ValueError) as exc:
        raise ValueError("The BO random seed must be an integer.") from exc
    if not 0 <= seed < 2**63:
        raise ValueError("The BO random seed must be between 0 and 2^63 - 1.")
    np.random.seed(seed)
    torch.manual_seed(seed)

    diagnostics = {
        "algorithm": None,
        "algorithm_label": None,
        "algorithm_reason": None,
        "acquisition_policy": None,
        "batch_size": None,
        "rows_received": int(len(df)),
        "rows_valid": None,
        "rows_used": None,
        "feature_count": len(features_config or []),
        "objective_count": len(objectives_config or []),
        "random_seed": seed,
        "compute_performance_requested": bool(compute_performance),
        "compute_performance_completed": False,
        "tensor_dtype": str(BO_TENSOR_DTYPE),
        "effective_budget": budget.to_dict(),
        "phase_seconds": {},
        "model_fits": [],
        "objective_data": {},
        "data_warnings": [],
        "data_handling": None,
        "workload": None,
    }

    def report(phase, percent, **details):
        _check_bo_runtime(cancel_event, deadline)
        payload = {
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "algorithm": diagnostics.get("algorithm"),
            **details,
        }
        if progress_callback is not None:
            progress_callback(phase, max(0, min(99, int(percent))), payload)

    def record_phase(name, phase_started):
        diagnostics["phase_seconds"][name] = round(
            time.monotonic() - phase_started, 3
        )

    report("Validating and preparing data", 1)
    preparation_started = time.monotonic()
    if not features_config:
        raise ValueError("At least one feature is required for optimization.")
    if not objectives_config:
        raise ValueError("At least one objective is required for optimization.")
    try:
        raw_batch_size = float(tweaks.get("batch_size", 1))
    except (TypeError, ValueError) as exc:
        raise ValueError("Batch size must be a whole number.") from exc
    if not np.isfinite(raw_batch_size) or not raw_batch_size.is_integer():
        raise ValueError("Batch size must be a whole number.")
    batch_size = int(raw_batch_size)
    if not 1 <= batch_size <= budget.max_batch_size:
        raise ValueError(
            f"Batch size must be between 1 and {budget.max_batch_size}."
        )
    diagnostics["batch_size"] = batch_size

    required_columns = {
        config["name"] for config in [*features_config, *objectives_config]
    }
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(
            "The dataset is missing configured column(s): "
            + ", ".join(missing_columns)
            + "."
        )
    df_clean = clean_dataframe(df, features_config, objectives_config)
    if not df.empty and df_clean.empty:
        raise ValueError(
            "No rows contain valid selected features and at least one valid "
            "objective value. Review missing or non-numeric cells."
        )
    df_all_clean = df_clean.copy()
    diagnostics["rows_valid"] = int(len(df_all_clean))
    objective_data, data_warnings = analyze_objective_data(
        df_all_clean, objectives_config, len(features_config)
    )
    diagnostics["objective_data"] = objective_data
    diagnostics["data_warnings"] = data_warnings

    raw_weights = [
        float(objective.get("importance", 1)) for objective in objectives_config
    ]
    normalized_weights = normalized_objective_weights(raw_weights)
    acquisition_type = str(tweaks.get("acq_type", "EI"))
    requested_method = str(tweaks.get("multi_objective_method", "auto"))
    policy = select_bo_policy(
        len(objectives_config),
        raw_weights,
        acquisition_type,
        budget.nehvi_max_objectives,
        requested_method=requested_method,
    )
    diagnostics.update({
        "algorithm": policy.key,
        "algorithm_label": policy.label,
        "algorithm_reason": policy.reason,
        "acquisition_policy": policy.to_dict(),
    })

    constraints_config = dict(tweaks.get("constraints") or {})
    numerical_rules = constraints_config.get("numerical") or []
    disallowed_rules = constraints_config.get("disallowed") or []
    objective_constraints = tweaks.get("objective_constraints") or []
    constrained_names = list(dict.fromkeys(
        constraint["objective"] for constraint in objective_constraints
    ))

    large_data_settings = load_bo_large_data_settings()
    requested_data_mode = validate_bo_data_mode(tweaks.get("data_mode", "auto"))
    per_model_rows = [
        int(pd.to_numeric(df_all_clean[objective["name"]], errors="coerce").notna().sum())
        for objective in objectives_config
    ]
    per_model_rows.extend(
        int(pd.to_numeric(df_all_clean[name], errors="coerce").notna().sum())
        for name in constrained_names
    )
    full_workload = assess_bo_workload(
        row_count=len(df_all_clean),
        feature_count=len(features_config),
        objective_count=len(objectives_config),
        batch_size=batch_size,
        method=policy.key,
        budget=budget,
        per_model_rows=per_model_rows,
        model_count=len(per_model_rows),
        settings=large_data_settings,
    )
    allow_unsafe_full = bool(tweaks.get("_allow_unsafe_full", False))
    should_reduce = (
        requested_data_mode == "representative"
        or (requested_data_mode == "auto" and full_workload.tier == "unsafe")
    )
    if (
        requested_data_mode == "full"
        and full_workload.tier == "unsafe"
        and not allow_unsafe_full
    ):
        raise ValueError(
            "Full exact-GP training is above this deployment's safety limit "
            f"(workload score {full_workload.risk_score:.2f}). Choose Auto or "
            "Representative exact GP. A standalone installation can allow this "
            "override with a larger runtime and memory budget."
        )

    selection_diagnostic = None
    effective_data_mode = "full"
    if should_reduce and len(df_all_clean) > full_workload.recommended_training_rows:
        if not full_workload.representative_feasible:
            raise ValueError(
                "This acquisition and problem size remain above the safety limit "
                "even at the minimum representative training size. Use the Auto "
                "acquisition method, reduce the objectives or batch size, or run "
                "the standalone app with larger server budgets."
            )
        report(
            "Selecting representative training rows",
            5,
            source_rows=len(df_all_clean),
            target_rows=full_workload.recommended_training_rows,
        )
        df_clean, selection_diagnostic = select_representative_bo_rows(
            df_all_clean,
            features_config,
            objectives_config,
            max_rows=full_workload.recommended_training_rows,
            target_fidelity_score=large_data_settings.target_fidelity_score,
            random_seed=seed,
            maximum_attempts=(
                large_data_settings.representative_selection_attempts
            ),
            search_time_limit_seconds=(
                large_data_settings.representative_selection_search_timeout_sec
            ),
            check_runtime=lambda: _check_bo_runtime(cancel_event, deadline),
        )
        effective_data_mode = "representative"
        if selection_diagnostic.fidelity_target_met:
            diagnostics["data_warnings"].append(
                "Large-data protection trained the exact GP on "
                f"{len(df_clean):,} representative rows from "
                f"{len(df_all_clean):,} valid rows "
                f"(fidelity {selection_diagnostic.fidelity_score:.1f}/100; "
                f"target {selection_diagnostic.target_fidelity_score:.1f}/100). "
                "All uploaded rows were still retained for duplicate avoidance."
            )
        else:
            attempt_word = (
                "attempt"
                if selection_diagnostic.selection_attempts == 1
                else "attempts"
            )
            time_limit_note = (
                " The bounded subset search reached its selection-time limit."
                if selection_diagnostic.search_time_limit_reached
                else ""
            )
            diagnostics["data_warnings"].append(
                "Representative-subset fallback used: large-data protection "
                f"trained the exact GP on the best safe subset found after "
                f"{selection_diagnostic.selection_attempts} bounded "
                f"selection {attempt_word} ({len(df_clean):,} of "
                f"{len(df_all_clean):,} valid rows). Its fidelity was "
                f"{selection_diagnostic.fidelity_score:.1f}/100, below the "
                f"target of {selection_diagnostic.target_fidelity_score:.1f}/100. "
                "Optimization continued, but the recommendations may represent "
                "the uploaded dataset less accurately. All uploaded rows were "
                f"retained for duplicate avoidance.{time_limit_note}"
            )
    elif requested_data_mode == "full" and full_workload.tier == "unsafe":
        diagnostics["data_warnings"].append(
            "Full exact-GP training is running above the standard safety score "
            "because this standalone deployment permits the override."
        )

    selected_model_rows = [
        int(pd.to_numeric(df_clean[objective["name"]], errors="coerce").notna().sum())
        for objective in objectives_config
    ]
    selected_model_rows.extend(
        int(pd.to_numeric(df_clean[name], errors="coerce").notna().sum())
        for name in constrained_names
    )
    selected_workload = assess_bo_workload(
        row_count=len(df_clean),
        feature_count=len(features_config),
        objective_count=len(objectives_config),
        batch_size=batch_size,
        method=policy.key,
        budget=budget,
        per_model_rows=selected_model_rows,
        model_count=len(selected_model_rows),
        settings=large_data_settings,
    )
    diagnostics["rows_used"] = int(len(df_clean))
    diagnostics["data_handling"] = {
        "requested_mode": requested_data_mode,
        "effective_mode": effective_data_mode,
        "source_rows": int(len(df_all_clean)),
        "training_rows": int(len(df_clean)),
        "selection": (
            selection_diagnostic.to_dict() if selection_diagnostic else None
        ),
    }
    diagnostics["workload"] = {
        "full_data": full_workload.to_dict(),
        "training_data": selected_workload.to_dict(),
        "thresholds": large_data_settings.to_dict(),
        "note": (
            "Planning estimate only; actual runtime depends on data geometry and "
            "hardware. The server always enforces the hard job deadline."
        ),
    }
    opt_device = compute_device
    bo_dtype = BO_TENSOR_DTYPE

    train_x_columns = []
    bounds = []
    categorical_features = []
    feature_levels = []
    for feature_index, feature in enumerate(features_config):
        name = feature["name"]
        range_value = str(feature.get("range", ""))
        feature_type = feature.get("type", "continuous")
        if feature_type == "continuous":
            matches = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", range_value)
            low, high = sorted(map(float, matches[:2]))
            values = pd.to_numeric(df_clean[name], errors="coerce").fillna(low).values
            levels = None
        elif feature_type == "discrete":
            levels = sorted(
                float(value.strip())
                for value in range_value.replace("[", "").replace("]", "").split(",")
                if value.strip()
            )
            if not levels:
                raise ValueError(f"Feature '{name}' requires at least one value.")
            low, high = levels[0], levels[-1]
            values = pd.to_numeric(df_clean[name], errors="coerce").fillna(low).values
        elif feature_type == "regular":
            matches = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", range_value)
            if len(matches) >= 3:
                low, high = sorted(map(float, matches[:2]))
                levels = np.linspace(low, high, int(float(matches[2]))).tolist()
            elif len(matches) == 2:
                low, high = sorted(map(float, matches))
                levels = np.linspace(low, high, 10).tolist()
            else:
                raise ValueError(
                    f"Feature '{name}' requires a regular range like [min, max]steps."
                )
            values = pd.to_numeric(df_clean[name], errors="coerce").fillna(low).values
        elif feature_type == "categorical":
            values, _ = encode_categorical_training_series(df_clean[name], feature)
            choices = get_categorical_choices(feature)
            low, high = 0.0, float(len(choices) - 1)
            levels = list(range(len(choices)))
            categorical_features.append(feature_index)
        else:
            raise ValueError(f"Feature '{name}' has unknown type '{feature_type}'.")
        if low == high:
            high = low + max(abs(low) * 0.01, 0.1)
        train_x_columns.append(values)
        bounds.append([float(low), float(high)])
        feature_levels.append(levels)

    if df_clean.empty:
        train_x = torch.empty(
            (0, len(features_config)), dtype=bo_dtype, device=opt_device
        )
    else:
        train_x = torch.tensor(
            np.stack(train_x_columns, axis=1),
            dtype=bo_dtype,
            device=opt_device,
        )
    bounds_t = torch.tensor(bounds, dtype=bo_dtype, device=opt_device).T

    feature_precision = {}
    for feature in features_config:
        column = (
            df_all_clean[feature["name"]].astype(str)
            if not df_all_clean.empty and feature["name"] in df_all_clean
            else pd.Series(dtype=str)
        )
        feature_precision[feature["name"]] = infer_decimal_precision(
            column, feature.get("range", "")
        )

    def decode_candidate(candidate):
        decoded = {}
        for index, feature in enumerate(features_config):
            value = float(candidate[index].item())
            feature_type = feature.get("type", "continuous")
            levels = feature_levels[index]
            if feature_type in {"discrete", "regular"} and levels:
                value = min(levels, key=lambda level: abs(level - value))
            elif feature_type == "categorical":
                choices = get_categorical_choices(feature)
                choice_index = max(0, min(len(choices) - 1, int(round(value))))
                value = choices[choice_index]
            if isinstance(value, (int, float)):
                value = round(float(value), feature_precision[feature["name"]])
            decoded[feature["name"]] = value
        return decoded

    feature_names = [feature["name"] for feature in features_config]

    def fingerprint(row):
        parts = []
        for name in feature_names:
            value = row[name]
            if isinstance(value, (int, float)):
                parts.append(f"{float(value):.{feature_precision[name]}f}")
            else:
                parts.append(str(value))
        return "|".join(parts)

    existing_fingerprints = set()
    if not df_all_clean.empty:
        for _, row in df_all_clean[feature_names].iterrows():
            existing_fingerprints.add(fingerprint(row.to_dict()))

    weights_t = torch.tensor(
        normalized_weights, dtype=bo_dtype, device=opt_device
    )

    train_y_columns = []
    if not df_clean.empty:
        for objective in objectives_config:
            values = pd.to_numeric(df_clean[objective["name"]], errors="coerce").values
            objective_type = objective.get("type", "maximize")
            if objective_type == "maximize":
                transformed = values
            elif objective_type == "minimize":
                transformed = -values
            elif objective_type == "target":
                target = float(objective.get("target", 0))
                valid = values[~np.isnan(values)]
                variance = float(np.var(valid)) if valid.size else 1.0
                if not np.isfinite(variance) or variance <= 0:
                    variance = 1.0
                transformed = -((values - target) ** 2) / variance
            else:
                raise ValueError(
                    f"Objective '{objective['name']}' has unknown type '{objective_type}'."
                )
            train_y_columns.append(transformed)
        train_y = torch.tensor(
            np.stack(train_y_columns, axis=1),
            dtype=bo_dtype,
            device=opt_device,
        )
    else:
        train_y = torch.empty(
            (0, len(objectives_config)), dtype=bo_dtype, device=opt_device
        )

    avoid_revalidation = bool(tweaks.get("avoid_reval", True))
    exploration = validate_exploration(tweaks.get("exploration", 0.5))
    record_phase("preparation", preparation_started)

    raw_candidates = []
    feasibility_models = {}
    build_single_model = None
    strategy = "cold_start"

    if df_clean.empty:
        diagnostics.update({
            "algorithm": "sobol-cold-start",
            "algorithm_label": "Constrained Sobol cold start",
            "algorithm_reason": "No completed observations are available for GP fitting.",
        })
        report("Generating cold-start candidates", 35)
        phase_started = time.monotonic()
        seen = set(existing_fingerprints if avoid_revalidation else ())
        draws = 0
        while len(raw_candidates) < batch_size and draws < budget.feasible_ic_max_draws:
            _check_bo_runtime(cancel_event, deadline)
            chunk_size = min(
                budget.feasible_ic_chunk_size,
                budget.feasible_ic_max_draws - draws,
            )
            sample = draw_sobol_samples(
                bounds=bounds_t, n=chunk_size, q=1
            ).squeeze(-2)
            draws += chunk_size
            for point in sample:
                decoded = decode_candidate(point)
                candidate_fingerprint = fingerprint(decoded)
                if candidate_fingerprint in seen:
                    continue
                if not check_constraints(
                    decoded, numerical_rules, disallowed_rules, features_config
                ):
                    continue
                seen.add(candidate_fingerprint)
                raw_candidates.append(point.detach().cpu())
                if len(raw_candidates) >= batch_size:
                    break
        if len(raw_candidates) < batch_size:
            raise FeasibleInitializationError(
                f"Only {len(raw_candidates)} distinct feasible cold-start conditions "
                f"were found after {draws} bounded draws; {batch_size} were requested. "
                "Review the feature ranges, precision, or constraints."
            )
        record_phase("cold_start", phase_started)
    else:
        kernel_name = str(tweaks.get("kernel", "matern52"))
        is_noiseless = bool(tweaks.get("noiseless", False))
        continuous_dimensions = [
            index for index in range(train_x.shape[1])
            if index not in categorical_features
        ]

        def get_kernel():
            if kernel_name == "rbf":
                return ScaleKernel(RBFKernel()).to(
                    device=opt_device, dtype=bo_dtype
                )
            if kernel_name == "matern32":
                return ScaleKernel(MaternKernel(nu=1.5)).to(
                    device=opt_device, dtype=bo_dtype
                )
            if kernel_name == "matern12":
                return ScaleKernel(MaternKernel(nu=0.5)).to(
                    device=opt_device, dtype=bo_dtype
                )
            return ScaleKernel(MaternKernel(nu=2.5)).to(
                device=opt_device, dtype=bo_dtype
            )

        def build_single_model(training_x, training_y):
            input_transform = (
                Normalize(
                    d=training_x.shape[1],
                    bounds=bounds_t[:, continuous_dimensions],
                    indices=continuous_dimensions,
                ).to(opt_device)
                if continuous_dimensions
                else None
            )
            if categorical_features and continuous_dimensions:
                fitted_model = MixedSingleTaskGP(
                    training_x.to(opt_device),
                    training_y.to(opt_device),
                    cat_dims=categorical_features,
                    outcome_transform=Standardize(m=1),
                    input_transform=input_transform,
                ).to(device=opt_device, dtype=bo_dtype)
            else:
                fitted_model = SingleTaskGP(
                    training_x.to(opt_device),
                    training_y.to(opt_device),
                    covar_module=get_kernel(),
                    outcome_transform=Standardize(m=1),
                    input_transform=input_transform,
                ).to(device=opt_device, dtype=bo_dtype)
            if is_noiseless:
                with torch.no_grad():
                    fitted_model.likelihood.noise_covar.noise = train_x.new_tensor(1e-4)
                fitted_model.likelihood.noise_covar.raw_noise.requires_grad_(False)
            return fitted_model

        total_models = len(objectives_config) + len(constrained_names)
        fitted_count = 0
        fit_phase_started = time.monotonic()
        fit_deadline = min(
            deadline, fit_phase_started + budget.fit_phase_timeout_sec
        )
        objective_models = []
        for objective_index, objective in enumerate(objectives_config):
            _check_bo_runtime(cancel_event, fit_deadline)
            y_slice = train_y[:, objective_index:objective_index + 1]
            valid_mask = torch.isfinite(y_slice.squeeze(-1))
            if not valid_mask.any():
                raise ValueError(
                    f"No valid observations are available for objective '{objective['name']}'."
                )
            fitted_model = build_single_model(train_x[valid_mask], y_slice[valid_mask])
            fitted_count += 1
            tasks_left = max(1, total_models - fitted_count + 1)
            per_model_timeout = max(
                1.0, _remaining_bo_seconds(fit_deadline) / tasks_left
            )
            report(
                f"Fitting objective model {objective_index + 1} of {len(objectives_config)}",
                10 + round(18 * objective_index / max(1, len(objectives_config))),
                model_index=objective_index + 1,
                model_count=len(objectives_config),
            )
            objective_mll = ExactMarginalLogLikelihood(
                fitted_model.likelihood, fitted_model
            ).to(opt_device)
            fit_diagnostic = _fit_required_model(
                objective_mll,
                label=f"objective '{objective['name']}'",
                maxiter=budget.fit_maxiter,
                maxls=budget.fit_maxls,
                retries=budget.fit_retries,
                timeout_sec=per_model_timeout,
                cancel_event=cancel_event,
                deadline=fit_deadline,
            )
            diagnostics["model_fits"].append(fit_diagnostic)
            objective_models.append(fitted_model.eval())

        model = (
            objective_models[0]
            if len(objective_models) == 1
            else ModelListGP(*objective_models).to(
                device=opt_device, dtype=bo_dtype
            )
        )

        for objective_name in constrained_names:
            _check_bo_runtime(cancel_event, fit_deadline)
            raw_values = pd.to_numeric(
                df_clean[objective_name], errors="coerce"
            ).to_numpy(dtype=float)
            valid_mask_np = np.isfinite(raw_values)
            if not valid_mask_np.any():
                raise ValueError(
                    f"No valid observations are available for constrained objective '{objective_name}'."
                )
            valid_mask = torch.tensor(
                valid_mask_np, dtype=torch.bool, device=opt_device
            )
            raw_y = torch.tensor(
                raw_values[valid_mask_np], dtype=bo_dtype, device=opt_device
            ).unsqueeze(-1)
            feasibility_model = build_single_model(train_x[valid_mask], raw_y)
            fitted_count += 1
            tasks_left = max(1, total_models - fitted_count + 1)
            per_model_timeout = max(
                1.0, _remaining_bo_seconds(fit_deadline) / tasks_left
            )
            report(
                f"Fitting feasibility model for {objective_name}",
                28,
                model_index=fitted_count,
                model_count=total_models,
            )
            feasibility_mll = ExactMarginalLogLikelihood(
                feasibility_model.likelihood, feasibility_model
            ).to(opt_device)
            fit_diagnostic = _fit_required_model(
                feasibility_mll,
                label=f"feasibility objective '{objective_name}'",
                maxiter=budget.fit_maxiter,
                maxls=budget.fit_maxls,
                retries=budget.fit_retries,
                timeout_sec=per_model_timeout,
                cancel_event=cancel_event,
                deadline=fit_deadline,
            )
            diagnostics["model_fits"].append(fit_diagnostic)
            feasibility_models[objective_name] = feasibility_model.eval()
        record_phase("model_fitting", fit_phase_started)

        report(
            f"Selected {policy.label}", 35,
            algorithm=policy.key,
            algorithm_reason=policy.reason,
        )
        sampler = SobolQMCNormalSampler(
            sample_shape=torch.Size([budget.mc_samples]), seed=seed
        )
        train_y_filled = train_y.clone()
        for objective_index in range(train_y.shape[1]):
            column = train_y_filled[:, objective_index]
            valid = column[torch.isfinite(column)]
            replacement = valid.min() if valid.numel() else torch.tensor(
                0.0, dtype=bo_dtype, device=opt_device
            )
            column[~torch.isfinite(column)] = replacement

        def build_acquisition(x_pending):
            if policy.key == "qLogNEHVI":
                minima = train_y_filled.min(dim=0).values
                maxima = train_y_filled.max(dim=0).values
                reference = (
                    minima - 0.2 * (maxima - minima + 1e-6)
                ) * weights_t
                acquisition = qLogNoisyExpectedHypervolumeImprovement(
                    model=model,
                    ref_point=reference,
                    X_baseline=train_x,
                    sampler=sampler,
                    objective=WeightedMCMultiOutputObjective(weights=weights_t),
                    X_pending=x_pending,
                    prune_baseline=True,
                    alpha=budget.nehvi_alpha,
                )
                logarithmic = True
            elif policy.key == "qLogNParEGO":
                acquisition = qLogNParEGO(
                    model=model,
                    X_baseline=train_x,
                    sampler=sampler,
                    X_pending=x_pending,
                    prune_baseline=True,
                )
                logarithmic = True
            elif policy.key == "scalarized-qLogEI":
                posterior_transform = ScalarizedPosteriorTransform(
                    weights=weights_t
                )
                best_f = (train_y_filled * weights_t).sum(dim=-1).max()
                acquisition = qLogExpectedImprovement(
                    model=model,
                    best_f=best_f,
                    sampler=sampler,
                    posterior_transform=posterior_transform,
                    X_pending=x_pending,
                )
                logarithmic = True
            elif policy.key == "qUCB":
                acquisition = qUpperConfidenceBound(
                    model=model,
                    beta=exploration_to_beta(exploration),
                    sampler=sampler,
                    X_pending=x_pending,
                )
                logarithmic = False
            else:
                acquisition = qLogExpectedImprovement(
                    model=model,
                    best_f=train_y_filled.max(),
                    sampler=sampler,
                    X_pending=x_pending,
                )
                logarithmic = True
            if objective_constraints:
                acquisition = ProbabilityOfFeasibilityAcquisition(
                    acquisition,
                    feasibility_models,
                    objective_constraints,
                    logarithmic=logarithmic,
                )
            return acquisition.to(opt_device)

        strategy = str(tweaks.get("optimization_strategy", "gradient_based"))
        candidate_phase_started = time.monotonic()
        if strategy == "exhaustive_grid":
            final_levels = []
            for index, levels in enumerate(feature_levels):
                if levels is None:
                    final_levels.append(
                        np.linspace(bounds[index][0], bounds[index][1], 10).tolist()
                    )
                else:
                    final_levels.append(levels)
            combination_count = int(np.prod([len(levels) for levels in final_levels]))
            if combination_count > 100_000:
                raise ValueError(
                    f"Exhaustive grid size ({combination_count:,}) exceeds the "
                    "100,000-condition safety limit. Reduce feature levels or use "
                    "gradient-based optimization."
                )
            available = []
            seen_training = existing_fingerprints if avoid_revalidation else set()
            for combination in itertools.product(*final_levels):
                point = torch.tensor(combination, dtype=bo_dtype)
                decoded = decode_candidate(point)
                if fingerprint(decoded) in seen_training:
                    continue
                if not check_constraints(
                    decoded, numerical_rules, disallowed_rules, features_config
                ):
                    continue
                available.append(combination)
            if len(available) < batch_size:
                if not available and avoid_revalidation:
                    diagnostics["exhausted"] = True
                else:
                    raise ValueError(
                        f"Only {len(available)} unevaluated feasible grid conditions "
                        f"remain, but {batch_size} were requested."
                    )
            if available:
                choices = torch.tensor(
                    available, dtype=bo_dtype, device=opt_device
                )
                pending = None
                for candidate_index in range(min(batch_size, len(available))):
                    report(
                        f"Optimizing candidate {candidate_index + 1} of {batch_size}",
                        40 + round(38 * candidate_index / batch_size),
                        candidate_index=candidate_index + 1,
                        candidate_count=batch_size,
                    )
                    acquisition = build_acquisition(pending)
                    candidate, _ = optimize_acqf_discrete(
                        acquisition,
                        q=1,
                        choices=choices,
                        max_batch_size=budget.discrete_eval_batch_size,
                        unique=True,
                    )
                    _check_bo_runtime(cancel_event, deadline)
                    raw_candidates.append(candidate.squeeze(0).detach().cpu())
                    pending = (
                        candidate
                        if pending is None
                        else torch.cat([pending, candidate], dim=0)
                    )
                    selected_mask = torch.isclose(
                        choices, candidate.squeeze(0), rtol=0.0, atol=1e-7
                    ).all(dim=-1)
                    choices = choices[~selected_mask]
        else:
            optimizer_constraints = compile_botorch_feature_constraints(
                numerical_rules, features_config
            )
            pending = None
            for candidate_index in range(batch_size):
                report(
                    f"Optimizing candidate {candidate_index + 1} of {batch_size}",
                    40 + round(38 * candidate_index / batch_size),
                    candidate_index=candidate_index + 1,
                    candidate_count=batch_size,
                )
                acquisition = build_acquisition(pending)
                initial_conditions = None
                if optimizer_constraints:
                    initial_conditions = generate_feasible_initial_conditions(
                        bounds_t,
                        q=1,
                        num_restarts=budget.num_restarts,
                        p_constraints=optimizer_constraints,
                        max_draws=budget.feasible_ic_max_draws,
                        chunk_size=budget.feasible_ic_chunk_size,
                        cancel_event=cancel_event,
                        deadline=deadline,
                    )
                candidate_timeout = min(
                    budget.candidate_timeout_sec,
                    _remaining_bo_seconds(deadline),
                )
                if candidate_timeout < 1.0:
                    raise BOTimeoutExceeded(
                        f"No time remains to optimize candidate {candidate_index + 1}."
                    )
                candidate, _ = optimize_acqf(
                    acq_function=acquisition,
                    bounds=bounds_t,
                    q=1,
                    num_restarts=budget.num_restarts,
                    raw_samples=budget.raw_samples,
                    options={
                        "maxiter": budget.acq_maxiter,
                        "batch_limit": (
                            1
                            if optimizer_constraints
                            else min(4, budget.num_restarts)
                        ),
                    },
                    nonlinear_inequality_constraints=(
                        optimizer_constraints or None
                    ),
                    batch_initial_conditions=initial_conditions,
                    timeout_sec=candidate_timeout,
                    retry_on_optimization_warning=False,
                )
                _check_bo_runtime(cancel_event, deadline)
                if not torch.isfinite(candidate).all():
                    raise RuntimeError(
                        f"Candidate {candidate_index + 1} contains non-finite values."
                    )
                raw_candidates.append(candidate.squeeze(0).detach().cpu())
                pending = (
                    candidate
                    if pending is None
                    else torch.cat([pending, candidate], dim=0)
                )
        record_phase("candidate_optimization", candidate_phase_started)

    report("Decoding and validating candidates", 82)
    validation_started = time.monotonic()
    suggestions = []
    seen_fingerprints = set(
        existing_fingerprints if avoid_revalidation else ()
    )
    duplicate_count = 0
    for candidate in raw_candidates:
        decoded = decode_candidate(candidate)
        candidate_fingerprint = fingerprint(decoded)
        if candidate_fingerprint in seen_fingerprints:
            duplicate_count += 1
            continue
        if not check_constraints(
            decoded, numerical_rules, disallowed_rules, features_config
        ):
            continue
        suggestions.append(decoded)
        seen_fingerprints.add(candidate_fingerprint)

    # Bounded Sobol completion handles rounding collisions and non-differentiable
    # disallowed combinations. It never returns an invalid or duplicate point.
    completion_draws = 0
    while (
        len(suggestions) < batch_size
        and strategy != "exhaustive_grid"
        and completion_draws < budget.feasible_ic_max_draws
    ):
        _check_bo_runtime(cancel_event, deadline)
        chunk_size = min(
            budget.feasible_ic_chunk_size,
            budget.feasible_ic_max_draws - completion_draws,
        )
        sample = draw_sobol_samples(
            bounds=bounds_t, n=chunk_size, q=1
        ).squeeze(-2).cpu()
        completion_draws += chunk_size
        for point in sample:
            decoded = decode_candidate(point)
            candidate_fingerprint = fingerprint(decoded)
            if candidate_fingerprint in seen_fingerprints:
                continue
            if not check_constraints(
                decoded, numerical_rules, disallowed_rules, features_config
            ):
                continue
            suggestions.append(decoded)
            seen_fingerprints.add(candidate_fingerprint)
            if len(suggestions) >= batch_size:
                break

    exhausted = (
        avoid_revalidation
        and not suggestions
        and (duplicate_count > 0 or diagnostics.get("exhausted", False))
    )
    if not exhausted and len(suggestions) != batch_size:
        raise FeasibleInitializationError(
            f"Only {len(suggestions)} distinct feasible suggestions could be "
            f"generated after {completion_draws} bounded completion draws; "
            f"{batch_size} were requested. Review the ranges, rounding precision, "
            "or constraints."
        )
    record_phase("candidate_validation", validation_started)

    performance_results = {}
    if compute_performance and not df_clean.empty and train_x.shape[0] >= 5:
        report("Computing optional validation diagnostics", 91)
        performance_started = time.monotonic()
        from sklearn.model_selection import KFold

        validation_deadline = min(deadline - 3.0, time.monotonic() + 60.0)
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)

        def validation_score(values, label):
            finite = np.isfinite(values)
            x_values = train_x.detach().cpu().numpy()[finite]
            y_values = values[finite]
            if y_values.size < 5:
                return {"r2": None, "status": "Unavailable", "message": "Fewer than five valid observations."}
            predictions = []
            observed = []
            try:
                for fold_index, (training_indices, test_indices) in enumerate(
                    kfold.split(x_values), start=1
                ):
                    if time.monotonic() >= validation_deadline:
                        raise BOTimeoutExceeded("Optional validation budget was reached.")
                    x_training = torch.tensor(
                        x_values[training_indices], dtype=bo_dtype, device=opt_device
                    )
                    y_training = torch.tensor(
                        y_values[training_indices], dtype=bo_dtype, device=opt_device
                    ).unsqueeze(-1)
                    x_test = torch.tensor(
                        x_values[test_indices], dtype=bo_dtype, device=opt_device
                    )
                    temporary_model = build_single_model(x_training, y_training)
                    temporary_mll = ExactMarginalLogLikelihood(
                        temporary_model.likelihood, temporary_model
                    ).to(opt_device)
                    _fit_required_model(
                        temporary_mll,
                        label=f"validation model for {label}, fold {fold_index}",
                        maxiter=min(25, budget.fit_maxiter),
                        maxls=budget.fit_maxls,
                        retries=budget.fit_retries,
                        timeout_sec=min(5.0, _remaining_bo_seconds(validation_deadline)),
                        cancel_event=cancel_event,
                        deadline=validation_deadline,
                    )
                    with torch.no_grad():
                        fold_predictions = temporary_model.posterior(x_test).mean
                    predictions.extend(fold_predictions.detach().cpu().numpy().ravel())
                    observed.extend(y_values[test_indices])
            except (BOModelFitError, BOTimeoutExceeded) as exc:
                return {"r2": None, "status": "Unavailable", "message": str(exc)}
            predictions = np.asarray(predictions)
            observed = np.asarray(observed)
            residual = np.sum((observed - predictions) ** 2)
            total = np.sum((observed - observed.mean()) ** 2)
            score = max(0.0, float(1.0 - residual / (total + 1e-9)))
            return {
                "r2": round(score, 3),
                "status": "Good" if score > 0.7 else ("Limited" if score >= 0.4 else "Poor"),
            }

        for objective_index, objective in enumerate(objectives_config):
            if time.monotonic() >= validation_deadline:
                performance_results[objective["name"]] = {
                    "r2": None,
                    "status": "Unavailable",
                    "message": "Optional validation budget was reached.",
                }
                continue
            values = train_y[:, objective_index].detach().cpu().numpy()
            performance_results[objective["name"]] = validation_score(
                values, objective["name"]
            )
        if len(objectives_config) > 1:
            filled = train_y.detach().cpu().numpy().copy()
            for objective_index in range(filled.shape[1]):
                column = filled[:, objective_index]
                finite = column[np.isfinite(column)]
                replacement = finite.min() if finite.size else 0.0
                column[~np.isfinite(column)] = replacement
            minima = filled.min(axis=0)
            maxima = filled.max(axis=0)
            normalized = (filled - minima) / (maxima - minima + 1e-9)
            global_values = np.average(
                normalized, axis=1, weights=np.asarray(normalized_weights)
            )
            performance_results["global_success"] = validation_score(
                global_values, "global success"
            )
        diagnostics["compute_performance_completed"] = all(
            result.get("r2") is not None
            for result in performance_results.values()
        )
        record_phase("optional_validation", performance_started)
    elif compute_performance:
        diagnostics["performance_note"] = (
            "Validation requires at least five complete observations."
        )
    else:
        diagnostics["performance_note"] = "Optional cross-validation was not requested."

    constraint_status = {
        "constrained": bool(objective_constraints),
        "all_likely_feasible": True,
        "candidates": [],
    }
    if objective_constraints and df_clean.empty:
        constraint_status.update({
            "cold_start": True,
            "all_likely_feasible": None,
            "message": (
                "Cold Start: objective constraints will be evaluated after the "
                "first experimental observations are recorded."
            ),
        })
    elif objective_constraints and suggestions:
        encoded_suggestions = []
        for suggestion in suggestions:
            encoded = []
            for feature in features_config:
                value = suggestion[feature["name"]]
                if feature.get("type") == "categorical":
                    value = get_categorical_choices(feature).index(str(value))
                encoded.append(float(value))
            encoded_suggestions.append(encoded)
        report_tensor = torch.tensor(
            encoded_suggestions, dtype=bo_dtype, device=opt_device
        )
        constraint_status = candidate_feasibility_report(
            feasibility_models, objective_constraints, report_tensor
        )
        if not constraint_status["all_likely_feasible"]:
            best_probability = max(
                (
                    item["probability"]
                    for item in constraint_status["candidates"]
                ),
                default=0.0,
            )
            constraint_status["message"] = (
                "Feasibility warning: at least one recommendation does not "
                "confidently satisfy every objective constraint "
                f"(best predicted feasibility: {best_probability:.1%})."
            )

    diagnostics["elapsed_seconds"] = round(time.monotonic() - started, 3)
    diagnostics["suggestion_count"] = 0 if exhausted else len(suggestions)
    diagnostics["duplicate_candidates_rejected"] = duplicate_count
    diagnostics["completion_draws"] = completion_draws
    return BOOptimizationResult(
        suggestions=None if exhausted else suggestions,
        performance=performance_results,
        constraint_status=constraint_status,
        diagnostics=diagnostics,
    )


def run_bo_estimation(df, input_values, features_config, objectives_config, tweaks, device, compute_device):
    # Filter the dataframe to drop incomplete rows
    df_clean = clean_dataframe(df, features_config, objectives_config)
    if df_clean.empty:
        raise ValueError('No valid data points left after filtering missing features/objectives.')

    # Build Feature Matrix (X)
    train_x_list      = []
    bounds            = []
    categorical_features = []

    for idx, f in enumerate(features_config):
        name    = f['name']
        f_range = f['range']

        if f['type'] == 'continuous':
            matches = re.findall(r"[-+]?\d*\.?\d+", f_range)
            low, high = sorted([float(matches[0]), float(matches[1])]) if len(matches) >= 2 else (0.0, 1.0)
            if low == high: high = low + abs(low) * 0.01 + 0.1
            feat_vals = pd.to_numeric(df_clean[name], errors='coerce').fillna(0).values
            train_x_list.append(feat_vals)
            bounds.append([low, high])

        elif f['type'] in ['discrete', 'regular']:
            matches = re.findall(r"[-+]?\d*\.?\d+", f_range)
            if f['type'] == 'discrete':
                values = sorted([float(x.strip()) for x in f_range.replace('[', '').replace(']', '').split(',') if x.strip()])
            else:
                if len(matches) >= 3:
                    low, high, steps = sorted([float(matches[0]), float(matches[1])]) + [int(float(matches[2]))]
                    values = np.linspace(low, high, steps).tolist()
                else:
                    values = [0.0]
            low, high = values[0], values[-1]
            if low == high: high = low + abs(low) * 0.01 + 0.1
            feat_vals = pd.to_numeric(df_clean[name], errors='coerce').fillna(values[0]).values
            train_x_list.append(feat_vals)
            bounds.append([low, high])

        elif f['type'] == 'categorical':
            encoded, _ = encode_categorical_training_series(df_clean[name], f)
            choices = get_categorical_choices(f)
            train_x_list.append(encoded)
            low, high = 0, len(choices) - 1
            if low == high: high = low + 0.1
            bounds.append([low, high])
            categorical_features.append(idx)

    train_x  = torch.tensor(np.stack(train_x_list, axis=1), dtype=torch.float32).to(device)
    bounds_t = torch.tensor(bounds, dtype=torch.float32).T.to(device)

    # Build Objective Matrix (Y) in optimization-transformed space
    train_y_list = []
    weights_list = []
    for obj in objectives_config:
        weights_list.append(float(obj.get('importance', 1)))
        vals = pd.to_numeric(df_clean[obj['name']], errors='coerce').values
        if   obj['type'] == 'minimize': train_y_list.append(-vals)
        elif obj['type'] == 'maximize': train_y_list.append(vals)
        elif obj['type'] == 'target':
            target   = float(obj.get('target', 0))
            valid_vals = vals[~np.isnan(vals)]
            variance = np.var(valid_vals) if len(valid_vals) > 0 and np.var(valid_vals) > 0 else 1.0
            train_y_list.append(-((vals - target) ** 2) / variance)

    train_y  = torch.tensor(np.stack(train_y_list, axis=1), dtype=torch.float32).to(device)

    kernel_name = tweaks.get('kernel', 'matern52')

    def get_kernel(k_name):
        if k_name == 'rbf':      return ScaleKernel(RBFKernel()).to(device)
        if k_name == 'matern32': return ScaleKernel(MaternKernel(nu=1.5)).to(device)
        if k_name == 'matern12': return ScaleKernel(MaternKernel(nu=0.5)).to(device)
        return ScaleKernel(MaternKernel(nu=2.5)).to(device)

    def build_single_model(tx, ty_slice):
        all_dims  = list(range(tx.shape[1]))
        cont_dims = [idx for idx in all_dims if idx not in categorical_features]
        input_tf  = (
            Normalize(d=tx.shape[1], bounds=bounds_t.to(compute_device)[:, cont_dims], indices=cont_dims).to(compute_device)
            if cont_dims else None
        )
        if categorical_features and len(cont_dims) > 0:
            return MixedSingleTaskGP(tx.to(compute_device), ty_slice.to(compute_device),
                                      cat_dims=categorical_features, outcome_transform=Standardize(m=1),
                                      input_transform=input_tf).to(compute_device)
        return SingleTaskGP(tx.to(compute_device), ty_slice.to(compute_device),
                             covar_module=get_kernel(kernel_name).to(compute_device),
                             outcome_transform=Standardize(m=1), input_transform=input_tf).to(compute_device)

    if train_y.shape[1] > 1:
        models = []
        for idx in range(train_y.shape[1]):
            y_slice = train_y[:, idx:idx+1]
            non_nan_mask = ~torch.isnan(y_slice.squeeze(-1))
            tx_i = train_x[non_nan_mask]
            ty_i = y_slice[non_nan_mask]
            if tx_i.shape[0] == 0:
                raise ValueError(f"No valid observations for objective '{objectives_config[idx]['name']}' after filtering missing values.")
            models.append(build_single_model(tx_i, ty_i))
        model   = ModelListGP(*models).to(compute_device)
        mll     = SumMarginalLogLikelihood(model.likelihood, model).to(compute_device)
    else:
        model   = build_single_model(train_x, train_y)
        mll     = ExactMarginalLogLikelihood(model.likelihood, model).to(compute_device)

    try:
        fit_gpytorch_mll(mll, optimizer_kwargs={"options": {"maxiter": 100}})
    except Exception as e:
        print(f"Warning: Main model fit failed in estimate_bo: {e}")
    model.eval()

    # Encode Query Point
    input_point_list = []
    for idx, f in enumerate(features_config):
        val = input_values[f['name']]
        if f['type'] == 'categorical':
            choices = get_categorical_choices(f)
            if str(val) not in choices:
                raise ValueError(f"Feature '{f['name']}' has unknown categorical value {val!r}.")
            val = choices.index(str(val))
        input_point_list.append(float(val))

    test_x = torch.tensor([input_point_list], dtype=torch.float32).to(compute_device)

    # Fit RAW (un-transformed) models to predict values in original units
    raw_predictions = {}
    raw_std_errors  = {}
    for idx, obj in enumerate(objectives_config):
        name  = obj['name']
        prec  = infer_decimal_precision(df_clean[name])
        y_raw = pd.to_numeric(df_clean[name], errors='coerce').values
        
        # Filter NaNs for raw model fitting
        non_nan_mask = ~np.isnan(y_raw)
        train_x_i = train_x[non_nan_mask]
        y_raw_t = torch.tensor(y_raw[non_nan_mask], dtype=torch.float32).unsqueeze(-1).to(device)

        if train_x_i.shape[0] == 0:
            raise ValueError(f"No valid observations for objective '{name}' after filtering missing values.")

        m_raw   = build_single_model(train_x_i, y_raw_t)
        mll_raw = ExactMarginalLogLikelihood(m_raw.likelihood, m_raw).to(compute_device)
        try:
            fit_gpytorch_mll(mll_raw, optimizer_kwargs={"options": {"maxiter": 100}})
        except Exception as e:
            print(f"Warning: Raw model fit failed in estimate_bo: {e}")
        m_raw.eval()

        with torch.no_grad():
            post_raw   = m_raw.posterior(test_x)
            pred_raw   = post_raw.mean.item()
            std_raw    = post_raw.stddev.item()
            raw_predictions[name] = round(float(pred_raw), prec)
            raw_std_errors[name]  = round(float(std_raw), prec + 2)

    # Compute Per-Objective and Global Success Scores (0-100%)
    success_scores    = {}
    processed_success = []

    for idx, obj in enumerate(objectives_config):
        name        = obj['name']
        val         = raw_predictions[name]
        train_vals  = pd.to_numeric(df_clean[name], errors='coerce').values
        
        # Clean training values for scoring bounds
        valid_train = train_vals[~np.isnan(train_vals)]
        if len(valid_train) == 0:
            success_scores[name] = 0.0
            continue

        if   obj['type'] == 'maximize': s = val;       t_s = valid_train
        elif obj['type'] == 'minimize': s = -val;      t_s = -valid_train
        else:
            target = float(obj.get('target', 0))
            s = -abs(val - target)
            t_s = -np.abs(valid_train - target)

        s_min, s_max = t_s.min(), t_s.max()
        norm_s = (s - s_min) / (s_max - s_min + 1e-9) if s_max > s_min else 1.0
        success_scores[name] = round(float(np.clip(norm_s, 0, 1) * 100), 1)
        processed_success.append(np.clip(norm_s, 0, 1))

    _, violations = check_objective_constraints(
        raw_predictions, tweaks.get('objective_constraints', [])
    )
    violated_objectives = {item['objective'] for item in violations}
    for objective_name in violated_objectives:
        if objective_name in success_scores:
            success_scores[objective_name] = 0.0

    weighted_scores = [
        float(success_scores.get(obj['name'], 0.0)) / 100.0
        for obj in objectives_config
    ]
    global_score = (
        np.average(weighted_scores, weights=weights_list)
        if weighted_scores else 0.0
    )

    return (
        raw_predictions,
        raw_std_errors,
        success_scores,
        round(float(global_score * 100), 1),
        violations,
    )


def run_bo_surface(
    df,
    features_config,
    objectives_config,
    tweaks,
    x_feat,
    y_feat,
    z_axis=None,
    color_obj=None,
    fixed_values=None,
    resolution=25,
    device=torch.device('cpu'),
    compute_device=torch.device('cpu'),
    cancel_event=None,
    progress_callback=None,
):
    """
    Computes a 2D prediction grid (response surface) using the fitted GP surrogate
    for contour and 3D surface visualizations.
    """
    def check_cancelled():
        if cancel_event is not None and cancel_event.is_set():
            raise SurfaceCalculationCancelled('Response-surface calculation cancelled by user.')

    def report(phase, percent):
        if progress_callback is not None:
            progress_callback(phase, percent)

    check_cancelled()
    report('Preparing response-surface data', 10)
    df_clean = clean_dataframe(df, features_config, objectives_config)
    if df_clean.empty:
        raise ValueError('No valid data points left after filtering missing features/objectives.')

    feat_names = [f['name'] for f in features_config if f['name'] in df_clean.columns]
    if len(feat_names) < 2:
        raise ValueError('At least two included numerical features are required for response surface.')

    if not x_feat or x_feat not in feat_names:
        x_feat = feat_names[0]
    if not y_feat or y_feat not in feat_names or y_feat == x_feat:
        remaining = [f for f in feat_names if f != x_feat]
        y_feat = remaining[0] if remaining else feat_names[1]

    obj_names = [o['name'] for o in objectives_config if o['name'] in df_clean.columns]
    if not obj_names:
        raise ValueError('At least one included objective is required for response surface.')

    if not color_obj or color_obj not in obj_names:
        color_obj = obj_names[0]
    # z_axis can be an objective OR a numerical feature (not x_feat or y_feat)
    num_feat_names = [
        f['name'] for f in features_config
        if f['name'] in df_clean.columns and f.get('type', 'continuous') != 'categorical'
    ]
    valid_z_options = obj_names + [f for f in num_feat_names if f != x_feat and f != y_feat]
    if not z_axis or z_axis not in valid_z_options:
        z_axis = color_obj

    if fixed_values is None:
        fixed_values = {}

    # Build Feature Matrix (X) and bounds
    train_x_list = []
    bounds = []
    categorical_features = []
    feat_ranges = {}

    for idx, f in enumerate(features_config):
        name = f['name']
        f_range = f.get('range', '')

        if f['type'] == 'continuous':
            matches = re.findall(r"[-+]?\d*\.?\d+", f_range)
            low, high = sorted([float(matches[0]), float(matches[1])]) if len(matches) >= 2 else (0.0, 1.0)
            if low == high:
                high = low + abs(low) * 0.01 + 0.1
            feat_vals = pd.to_numeric(df_clean[name], errors='coerce').fillna(0).values
            train_x_list.append(feat_vals)
            bounds.append([low, high])
            feat_ranges[name] = (low, high, 'continuous', None)

        elif f['type'] in ['discrete', 'regular']:
            matches = re.findall(r"[-+]?\d*\.?\d+", f_range)
            if f['type'] == 'discrete':
                clean_range = f_range.replace('[', '').replace(']', '')
                values = sorted([float(x.strip()) for x in clean_range.split(',') if x.strip()])
                if not values:
                    values = [0.0]
            else:
                if len(matches) >= 3:
                    low, high, steps = sorted([float(matches[0]), float(matches[1])]) + [int(float(matches[2]))]
                    values = np.linspace(low, high, steps).tolist()
                elif len(matches) == 2:
                    low, high = sorted([float(matches[0]), float(matches[1])])
                    values = np.linspace(low, high, 10).tolist()
                else:
                    values = [0.0]
            low, high = values[0], values[-1]
            if low == high:
                high = low + abs(low) * 0.01 + 0.1
            feat_vals = pd.to_numeric(df_clean[name], errors='coerce').fillna(values[0]).values
            train_x_list.append(feat_vals)
            bounds.append([low, high])
            feat_ranges[name] = (low, high, f['type'], values)

        elif f['type'] == 'categorical':
            encoded, _ = encode_categorical_training_series(df_clean[name], f)
            choices = get_categorical_choices(f)
            train_x_list.append(encoded)
            low, high = 0, len(choices) - 1
            if low == high:
                high = low + 0.1
            bounds.append([low, high])
            categorical_features.append(idx)
            feat_ranges[name] = (0, len(choices) - 1, 'categorical', choices)

    train_x = torch.tensor(np.stack(train_x_list, axis=1), dtype=torch.float32).to(device)
    bounds_t = torch.tensor(bounds, dtype=torch.float32).T.to(device)

    res = max(5, min(int(resolution), 50))

    x_info = feat_ranges.get(x_feat, (0.0, 1.0, 'continuous', None))
    x_vals = np.linspace(x_info[0], x_info[1], res)

    y_info = feat_ranges.get(y_feat, (0.0, 1.0, 'continuous', None))
    y_vals = np.linspace(y_info[0], y_info[1], res)

    X_mesh, Y_mesh = np.meshgrid(x_vals, y_vals)
    x_flat = X_mesh.ravel()
    y_flat = Y_mesh.ravel()
    n_points = len(x_flat)

    eval_matrix = np.zeros((n_points, len(features_config)), dtype=np.float32)
    for idx, f in enumerate(features_config):
        fname = f['name']
        if fname == x_feat:
            eval_matrix[:, idx] = x_flat
        elif fname == y_feat:
            eval_matrix[:, idx] = y_flat
        else:
            raw_fixed = fixed_values.get(fname, None)
            f_inf = feat_ranges.get(fname, (0.0, 1.0, 'continuous', None))
            if raw_fixed is None:
                if f_inf[2] == 'categorical' and f_inf[3]:
                    val = 0.0
                else:
                    val = (f_inf[0] + f_inf[1]) / 2.0
            else:
                if f_inf[2] == 'categorical' and f_inf[3]:
                    val = float(f_inf[3].index(str(raw_fixed))) if str(raw_fixed) in f_inf[3] else 0.0
                else:
                    try:
                        val = float(raw_fixed)
                    except Exception:
                        val = (f_inf[0] + f_inf[1]) / 2.0
            eval_matrix[:, idx] = val

    eval_x = torch.tensor(eval_matrix, dtype=torch.float32).to(compute_device)
    check_cancelled()
    report('Building response-surface grid', 25)

    kernel_name = tweaks.get('kernel', 'matern52')
    def get_kernel(k_name):
        if k_name == 'rbf':      return ScaleKernel(RBFKernel()).to(device)
        if k_name == 'matern32': return ScaleKernel(MaternKernel(nu=1.5)).to(device)
        if k_name == 'matern12': return ScaleKernel(MaternKernel(nu=0.5)).to(device)
        return ScaleKernel(MaternKernel(nu=2.5)).to(device)

    def build_single_model(tx, ty_slice):
        all_dims = list(range(tx.shape[1]))
        cont_dims = [i for i in all_dims if i not in categorical_features]
        input_tf = (
            Normalize(d=tx.shape[1], bounds=bounds_t.to(compute_device)[:, cont_dims], indices=cont_dims).to(compute_device)
            if cont_dims else None
        )
        if categorical_features and len(cont_dims) > 0:
            return MixedSingleTaskGP(tx.to(compute_device), ty_slice.to(compute_device),
                                     cat_dims=categorical_features, outcome_transform=Standardize(m=1),
                                     input_transform=input_tf).to(compute_device)
        return SingleTaskGP(tx.to(compute_device), ty_slice.to(compute_device),
                            covar_module=get_kernel(kernel_name).to(compute_device),
                            outcome_transform=Standardize(m=1), input_transform=input_tf).to(compute_device)

    obj_dict = {o['name']: o for o in objectives_config}

    def fit_and_predict_obj(obj_name, position, total):
        check_cancelled()
        report(f'Fitting surrogate for {obj_name}', 35 + int(45 * (position / max(total, 1))))
        prec = infer_decimal_precision(df_clean[obj_name])
        y_raw = pd.to_numeric(df_clean[obj_name], errors='coerce').values
        non_nan_mask = ~np.isnan(y_raw)
        tx_i = train_x[non_nan_mask]
        y_t = torch.tensor(y_raw[non_nan_mask], dtype=torch.float32).unsqueeze(-1).to(device)
        if tx_i.shape[0] == 0:
            return np.zeros((res, res), dtype=float)
        m = build_single_model(tx_i, y_t)
        mll = ExactMarginalLogLikelihood(m.likelihood, m).to(compute_device)
        try:
            fit_gpytorch_mll(
                mll,
                optimizer_kwargs={
                    "options": {"maxiter": 35},
                    "callback": lambda *_: check_cancelled(),
                },
            )
        except SurfaceCalculationCancelled:
            raise
        except Exception:
            pass
        check_cancelled()
        m.eval()
        with torch.no_grad():
            post = m.posterior(eval_x)
            pred = post.mean.squeeze(-1).cpu().numpy()
        pred_grid = np.round(pred.reshape(res, res), prec)
        return pred_grid

    predicted_objectives = [color_obj]
    if z_axis in obj_dict and z_axis != color_obj:
        predicted_objectives.append(z_axis)

    color_grid = fit_and_predict_obj(
        color_obj, 0, len(predicted_objectives)
    ).tolist()

    if z_axis in obj_dict:
        # z_axis is an objective — predict with GP
        if z_axis == color_obj:
            z_grid = color_grid
        else:
            z_grid = fit_and_predict_obj(
                z_axis, 1, len(predicted_objectives)
            ).tolist()
    else:
        # z_axis is a numerical feature — use the actual feature values from the eval grid
        feat_idx_map = {f['name']: idx for idx, f in enumerate(features_config)}
        if z_axis in feat_idx_map:
            col_idx = feat_idx_map[z_axis]
            z_flat = eval_matrix[:, col_idx]
            prec_z = infer_decimal_precision(df_clean[z_axis])
            z_grid = np.round(z_flat.reshape(res, res), prec_z).tolist()
        else:
            z_grid = color_grid

    x_col_raw = df_clean[x_feat].tolist()
    y_col_raw = df_clean[y_feat].tolist()
    color_col_raw = df_clean[color_obj].tolist() if color_obj in df_clean.columns else []
    z_col_raw = df_clean[z_axis].tolist() if z_axis and z_axis in df_clean.columns else color_col_raw

    # Determine remaining features and their target slice values (nearest for numeric, exact for categorical)
    remaining_feats = [f for f in features_config if f['name'] != x_feat and f['name'] != y_feat]
    slice_targets = {}

    for f in remaining_feats:
        fname = f['name']
        ftype = f.get('type', 'continuous')
        raw_fixed = fixed_values.get(fname, None)
        f_inf = feat_ranges.get(fname, (0.0, 1.0, 'continuous', None))

        if ftype == 'categorical':
            choices = f_inf[3] or []
            selected_cat = str(raw_fixed) if raw_fixed is not None and str(raw_fixed) in choices else (choices[0] if choices else '')
            slice_targets[fname] = {
                "type": "categorical",
                "target_val": selected_cat,
                "label": selected_cat
            }
        else:
            try:
                selected_num = float(raw_fixed) if raw_fixed is not None else (f_inf[0] + f_inf[1]) / 2.0
            except Exception:
                selected_num = (f_inf[0] + f_inf[1]) / 2.0
            
            # Find nearest unique value in the actual dataset
            col_vals = pd.to_numeric(df_clean[fname], errors='coerce').dropna().values
            if len(col_vals) > 0:
                unique_vals = np.unique(col_vals)
                nearest_idx = np.argmin(np.abs(unique_vals - selected_num))
                nearest_val = float(unique_vals[nearest_idx])
            else:
                nearest_val = selected_num
            
            slice_targets[fname] = {
                "type": "numerical",
                "target_val": nearest_val,
                "selected_val": selected_num,
                "label": f"{nearest_val:g}"
            }

    report('Classifying experiments for the selected slice', 85)
    experiments = []
    for i in range(len(df_clean)):
        if i % 256 == 0:
            check_cancelled()
        try:
            exp_x = float(x_col_raw[i]) if pd.notna(x_col_raw[i]) else None
            exp_y = float(y_col_raw[i]) if pd.notna(y_col_raw[i]) else None
            exp_c = float(color_col_raw[i]) if i < len(color_col_raw) and pd.notna(color_col_raw[i]) else None
            exp_z = float(z_col_raw[i]) if i < len(z_col_raw) and pd.notna(z_col_raw[i]) else exp_c

            if exp_x is not None and exp_y is not None:
                # Check if this row matches slice conditions
                is_slice_match = True
                row_remaining_info = []

                for f in remaining_feats:
                    fname = f['name']
                    target_info = slice_targets[fname]
                    row_val_raw = df_clean.iloc[i][fname]

                    if target_info['type'] == 'categorical':
                        row_val_str = str(row_val_raw)
                        row_remaining_info.append(f"{fname}={row_val_str}")
                        if row_val_str != target_info['target_val']:
                            is_slice_match = False
                    else:
                        try:
                            row_val_num = float(row_val_raw)
                            row_remaining_info.append(f"{fname}={row_val_num:g}")
                            if abs(row_val_num - target_info['target_val']) > 1e-6:
                                is_slice_match = False
                        except Exception:
                            is_slice_match = False
                            row_remaining_info.append(f"{fname}={row_val_raw}")

                experiments.append({
                    "exp_id": i + 1,
                    "x": exp_x,
                    "y": exp_y,
                    "z": exp_z,
                    "color": exp_c,
                    "in_slice": is_slice_match,
                    "conditions": ", ".join(row_remaining_info) if row_remaining_info else ""
                })
        except Exception:
            pass

    check_cancelled()
    report('Finalizing response surface', 95)
    prec_x = infer_decimal_precision(df_clean[x_feat])
    prec_y = infer_decimal_precision(df_clean[y_feat])

    return {
        "x": [round(float(v), prec_x) for v in x_vals],
        "y": [round(float(v), prec_y) for v in y_vals],
        "z": z_grid,
        "color": color_grid,
        "x_feat": x_feat,
        "y_feat": y_feat,
        "z_axis": z_axis or color_obj,
        "color_obj": color_obj,
        "experiments": experiments,
        "slice_targets": {k: v['label'] for k, v in slice_targets.items()}
    }
