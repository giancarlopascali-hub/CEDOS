"""
TransEDOS - Transfer Learning Bayesian Optimization Engine
============================================================
Extends EDOS Bayesian Optimization with transfer learning across experimental domains.
Weights historical/prior data based on proximity to target experimental conditions
(SMILES molecular similarity, numerical proximity, or categorical matching)
via heteroscedastic noise GP modeling (FixedNoiseGP formulation).
"""

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
from botorch.fit import fit_gpytorch_mll
from botorch.optim.initializers import gen_batch_initial_conditions

# Acquisition functions
from botorch.acquisition import (
    ExpectedImprovement,
    UpperConfidenceBound,
    qExpectedImprovement,
    qUpperConfidenceBound,
)
try:
    from botorch.acquisition import qLogExpectedImprovement
except ImportError:
    qLogExpectedImprovement = qExpectedImprovement

from botorch.acquisition.multi_objective.monte_carlo import (
    qNoisyExpectedHypervolumeImprovement,
    qExpectedHypervolumeImprovement,
)
try:
    from botorch.acquisition.multi_objective.monte_carlo import qLogNoisyExpectedHypervolumeImprovement
except ImportError:
    qLogNoisyExpectedHypervolumeImprovement = qNoisyExpectedHypervolumeImprovement

from botorch.acquisition.multi_objective.objective import WeightedMCMultiOutputObjective
from botorch.acquisition.objective import ScalarizedPosteriorTransform

from botorch.optim import optimize_acqf, optimize_acqf_discrete
from botorch.utils.sampling import draw_sobol_samples
from botorch.sampling.normal import SobolQMCNormalSampler

from src.core.constraints import (
    check_constraints,
    check_objective_constraints,
    encode_categorical_training_series,
    get_categorical_choices,
)
from src.core.feasibility import (
    ProbabilityOfFeasibilityAcquisition,
    candidate_feasibility_report,
)
from src.core.prior_utils import (
    assert_rdkit,
    compute_combined_prior_relevance,
)
from src.core.bo import (
    clean_dataframe,
    exploration_to_beta,
    infer_decimal_precision,
    generate_feasible_initial_conditions,
    run_bo_optimization,
    SurfaceCalculationCancelled,
    validate_exploration,
)
MAX_GP_ROWS = 650
MIN_GP_TRAINING_ROWS = 3
TRANSFER_BO_MAX_BETA = 4.0


class TransferGPFitError(RuntimeError):
    """A GP fitting failure that may be recoverable with fewer training rows."""


class TransferGPFitTimeout(TransferGPFitError):
    """An exact-GP attempt exceeded its server-side time budget."""


class TransferCalculationCancelled(RuntimeError):
    """A transfer calculation was cancelled at a cooperative checkpoint."""


def _transfer_control_check(cancel_event=None, deadline_monotonic=None):
    if cancel_event is not None and cancel_event.is_set():
        raise TransferCalculationCancelled('Transfer calculation cancelled by user.')
    if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
        raise TransferGPFitTimeout(
            'Transfer GP attempt exceeded its server-side time budget.'
        )


def _remaining_transfer_seconds(deadline_monotonic):
    if deadline_monotonic is None:
        return None
    remaining = deadline_monotonic - time.monotonic()
    if remaining <= 0:
        raise TransferGPFitTimeout(
            'Transfer GP attempt exceeded its server-side time budget.'
        )
    return max(0.1, remaining)


def transfer_exploration_to_beta(exploration):
    """Map the public 0-1 slider to Transfer BO's UCB beta range."""
    return exploration_to_beta(exploration, max_beta=TRANSFER_BO_MAX_BETA)


def resolve_gp_row_capacity(
    available_rows,
    objective_count=1,
    constraint_count=0,
    has_categorical=False,
):
    """Return the full dataset size selectable by the user for exact-GP fitting.

    Workload complexity informs the warning and time estimate, rather than
    silently reducing the requested number of rows.
    """
    return max(0, int(available_rows or 0))


def resolve_gp_row_limit(
    requested_rows,
    available_rows,
    objective_count=1,
    constraint_count=0,
    has_categorical=False,
):
    """Clamp a requested row count only to the available dataset rows."""
    available = max(0, int(available_rows or 0))
    if available == 0:
        return 0
    requested = max(1, min(int(requested_rows or MAX_GP_ROWS), available))
    capacity = resolve_gp_row_capacity(
        available,
        objective_count=objective_count,
        constraint_count=constraint_count,
        has_categorical=has_categorical,
    )
    return min(requested, capacity)


def _subsample_training_data(
    train_x,
    train_y,
    train_yvar,
    combined_weights,
    max_rows=MAX_GP_ROWS,
    eligible_mask=None,
    min_relevance=None,
):
    """
    Filters by the explicit Prior eligibility mask, then deterministically keeps
    the highest-ranked rows up to ``max_rows``. Boundary ties are spread across
    source-row order. ``min_relevance`` remains as a compatibility fallback for
    callers outside the current TransBO workflow.

    Returns:
    - sub_x: torch.Tensor
    - sub_y: torch.Tensor
    - sub_yvar: torch.Tensor
    - sub_weights: np.ndarray
    - chosen_indices: np.ndarray
    """
    n = train_x.shape[0]
    weights_np = np.asarray(combined_weights, dtype=float)
    if weights_np.ndim > 1:
        weights_np = weights_np.squeeze()

    if eligible_mask is not None:
        mask = np.asarray(eligible_mask, dtype=bool).reshape(-1)
        if len(mask) != n:
            raise ValueError('Prior eligibility mask does not match the training rows.')
        eligible = np.flatnonzero(mask)
    elif min_relevance is not None:
        threshold = float(np.clip(min_relevance, 0.0, 1.0))
        eligible = np.flatnonzero(weights_np >= threshold)
    else:
        eligible = np.arange(n, dtype=int)
    if len(eligible) <= max_rows:
        chosen_indices = eligible
    else:
        eligible_weights = weights_np[eligible]
        ranked_order = np.argsort(-eligible_weights, kind='stable')
        boundary_weight = eligible_weights[ranked_order[max_rows - 1]]
        higher = eligible[eligible_weights > boundary_weight]
        tied = np.sort(eligible[eligible_weights == boundary_weight])
        tied_needed = max_rows - len(higher)
        if tied_needed >= len(tied):
            selected_ties = tied
        else:
            positions = np.linspace(0, len(tied) - 1, tied_needed, dtype=int)
            selected_ties = tied[positions]
        chosen_indices = np.sort(np.concatenate([higher, selected_ties]))

    sub_x = train_x[chosen_indices]
    sub_y = train_y[chosen_indices]
    sub_yvar = train_yvar[chosen_indices]
    sub_weights = weights_np[chosen_indices]

    return sub_x, sub_y, sub_yvar, sub_weights, chosen_indices


def _reuse_training_data(
    train_x, train_y, train_yvar, source_indices, reuse_source_indices
):
    """Return an exact prior training set when all source rows still exist.

    ``source_indices`` belong to the original uploaded dataframe. Reuse is
    deliberately all-or-nothing: a changed dataset falls back to normal
    relevance-ranked selection instead of mixing old and new row sets.
    """
    if not isinstance(reuse_source_indices, (list, tuple, np.ndarray)):
        return None
    try:
        requested = [int(index) for index in reuse_source_indices]
    except (TypeError, ValueError):
        return None
    if len(requested) < MIN_GP_TRAINING_ROWS or len(set(requested)) != len(requested):
        return None

    source_to_position = {
        int(source_index): position
        for position, source_index in enumerate(source_indices)
    }
    if any(index not in source_to_position for index in requested):
        return None

    positions = np.asarray([source_to_position[index] for index in requested], dtype=int)
    return (
        train_x[positions],
        train_y[positions],
        train_yvar[positions],
        positions,
    )


def preview_transfer_selection(
    df,
    features_config,
    objectives_config,
    priors_config,
    prior_combine='multiply',
    objective_constraints=None,
    requested_rows=None,
):
    """Preview the exact rows eligible for and permitted in TransferBO fitting."""
    df_clean = clean_dataframe(df, features_config, objectives_config)
    valid_rows = len(df_clean)
    relevance = compute_combined_prior_relevance(
        df_clean,
        priors_config,
        method=prior_combine,
    )
    eligible_rows = relevance['eligible_rows']
    selectable_rows = int(len(df))
    requested = selectable_rows if requested_rows in (None, '') else int(requested_rows)
    requested = max(1, min(requested, selectable_rows)) if selectable_rows else 0
    return {
        'available_rows': int(len(df)),
        'valid_rows': int(valid_rows),
        'eligible_rows': eligible_rows,
        'permitted_max_rows': selectable_rows,
        'requested_rows': int(requested),
        'rows_to_fit': int(min(requested, eligible_rows)),
        'prior_combine_rule': prior_combine,
        'prior_stats': relevance['prior_stats'],
    }


def run_transfer_bo_optimization(
    df,
    features_config,
    objectives_config,
    priors_config,
    tweaks,
    device,
    compute_device,
    prior_combine="multiply",
    max_gp_rows=None,
    cancel_event=None,
    deadline_monotonic=None,
):
    """
    Runs Transfer Learning Bayesian Optimization.

    Parameters:
    - df: pandas DataFrame containing training data (including prior columns).
    - features_config: list of feature configs (from Tab 1).
    - objectives_config: list of objective configs (from Tab 3).
    - priors_config: list of prior configs (from Tab 2), each with:
        {'name': str, 'col_type': str, 'target_value': any,
         'tolerance' or 'min_similarity': float, 'prior_weight': float}
    - tweaks: dict of BO hyperparameters (from Tab 4).
    - device: torch device for optimization.
    - compute_device: torch device for GP modeling (CPU recommended for stability).
    - prior_combine: 'multiply' | 'average'

    Returns:
    - suggestions: list of candidate dicts.
    - performance: dict of model fit diagnostics (R2, MSE, scatter data).
    - transfer_diagnostics: dict with weight stats, algorithm used, cold-start warnings, etc.
    """
    def check_control():
        _transfer_control_check(cancel_event, deadline_monotonic)

    check_control()
    # Defensive check: if no priors provided, fall back to standard BO
    if not priors_config:
        requested_limit = max_gp_rows or tweaks.get('max_gp_rows') or MAX_GP_ROWS
        result = run_bo_optimization(
            df=df,
            features_config=features_config,
            objectives_config=objectives_config,
            tweaks=tweaks,
            device=device,
            compute_device=compute_device,
            cancel_event=cancel_event,
            deadline=deadline_monotonic,
            compute_performance=True,
        )
        diagnostics = result.diagnostics or {}
        data_handling = diagnostics.get('data_handling') or {}
        source_rows = int(data_handling.get('source_rows') or len(df))
        training_rows = int(data_handling.get('training_rows') or source_rows)
        return result.suggestions, result.performance, {
            "algorithm_used": "standard_bo",
            "cold_start_warning": False,
            "rdkit_available": True,
            "weight_stats": {},
            "combined_weights": [],
            "requested_gp_rows": int(requested_limit),
            "model_rows_used": training_rows,
            "rows_downsampled": max(0, source_rows - training_rows),
            "eligible_rows": int(diagnostics.get('rows_valid') or source_rows),
            "permitted_max_rows": int(len(df)),
            "data_handling": data_handling,
            "data_warnings": diagnostics.get('data_warnings') or [],
            "standard_bo_diagnostics": diagnostics,
        }, result.constraint_status

    # Filter the dataframe to drop rows with missing features or objectives
    df_clean = clean_dataframe(df, features_config, objectives_config)

    # -------------------------------------------------------------
    # 1. Process Prior Columns & Calculate Transfer Weights
    # -------------------------------------------------------------
    check_control()
    n_rows = len(df_clean)
    relevance = compute_combined_prior_relevance(
        df_clean,
        priors_config,
        method=prior_combine,
    )
    combined_weights = relevance['combined_weights']
    eligible_mask = relevance['eligible_mask']
    weight_stats = relevance['prior_stats']
    objective_constraints = tweaks.get('objective_constraints', [])
    effective_count = relevance['eligible_rows']
    is_cold_start = (n_rows == 0) or effective_count < MIN_GP_TRAINING_ROWS

    # Build transfer diagnostics payload
    transfer_diagnostics = {
        "algorithm_used": "fixed_noise_gp" if not is_cold_start else "cold_start_sobol",
        "cold_start_warning": is_cold_start,
        "warning_message": (
            "No source data points are sufficiently close to the specified prior target(s). "
            "Generating cold-start exploratory candidates."
            if is_cold_start else None
        ),
        "rdkit_available": True,
        "prior_combine_rule": prior_combine,
        "effective_rows": effective_count,
        "eligible_rows": effective_count,
        "total_rows": n_rows,
        "weight_stats": weight_stats,
        "combined_weights": [round(float(w), 4) for w in combined_weights],
        "source_labels": [f"Row {i+1}" for i in range(n_rows)]
    }

    # -------------------------------------------------------------
    # 2. Build Training Features (X) and Bounds
    # -------------------------------------------------------------
    train_x_list = []
    bounds = []
    categorical_features = []
    feature_levels = []

    for i, f in enumerate(features_config):
        name = f['name']
        f_range = f['range']

        if f['type'] == 'continuous':
            matches = re.findall(r"[-+]?\d*\.?\d+", f_range)
            low, high = sorted([float(matches[0]), float(matches[1])]) if len(matches) >= 2 else (0.0, 1.0)
            if low == high:
                high = low + abs(low) * 0.01 + 0.1
            feat_vals = pd.to_numeric(df_clean[name], errors='coerce').fillna(0).values if not df_clean.empty else np.array([])
            train_x_list.append(feat_vals)
            bounds.append([low, high])
            feature_levels.append(None)

        elif f['type'] == 'discrete':
            clean_range = f_range.replace('[', '').replace(']', '')
            values = sorted([float(x.strip()) for x in clean_range.split(',') if x.strip()])
            if not values:
                values = [0.0]
            low, high = values[0], values[-1]
            if low == high:
                high = low + abs(low) * 0.01 + 0.1
            feat_vals = pd.to_numeric(df_clean[name], errors='coerce').fillna(values[0]).values if not df_clean.empty else np.array([])
            train_x_list.append(feat_vals)
            bounds.append([low, high])
            feature_levels.append(values)

        elif f['type'] == 'regular':
            matches = re.findall(r"[-+]?\d*\.?\d+", f_range)
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
            feat_vals = pd.to_numeric(df_clean[name], errors='coerce').fillna(values[0]).values if not df_clean.empty else np.array([])
            train_x_list.append(feat_vals)
            bounds.append([low, high])
            feature_levels.append(values)

        elif f['type'] == 'categorical':
            choices = get_categorical_choices(f)
            if df_clean.empty:
                encoded = np.array([])
            else:
                encoded, _ = encode_categorical_training_series(df_clean[name], f)
            train_x_list.append(encoded)
            low, high = 0, len(choices) - 1
            if low == high:
                high = low + 0.1
            bounds.append([low, high])
            categorical_features.append(i)
            feature_levels.append(list(range(len(choices))))

    if not df_clean.empty:
        train_x = torch.tensor(np.stack(train_x_list, axis=1), dtype=torch.float32).to(device)
    else:
        train_x = torch.zeros((0, len(features_config)), dtype=torch.float32).to(device)
    bounds_t = torch.tensor(bounds, dtype=torch.float32).T.to(device)

    # Feature precision dictionary for candidate rounding
    feature_precision = {}
    for f in features_config:
        col_s = (
            df_clean[f['name']].astype(str)
            if (not df_clean.empty and f['name'] in df_clean.columns)
            else pd.Series(dtype=str)
        )
        feature_precision[f['name']] = infer_decimal_precision(col_s, f.get('range', ''))

    # -------------------------------------------------------------
    # 3. Build Training Objectives (Y)
    # -------------------------------------------------------------
    train_y_list = []
    weights_list = []

    for obj in objectives_config:
        weights_list.append(float(obj.get('importance', 1)))
        if not df_clean.empty:
            vals = pd.to_numeric(df_clean[obj['name']], errors='coerce').values
            if obj['type'] == 'minimize':
                train_y_list.append(-vals)
            elif obj['type'] == 'maximize':
                train_y_list.append(vals)
            elif obj['type'] == 'target':
                target = float(obj.get('target', 0))
                valid_vals = vals[~np.isnan(vals)]
                variance = np.var(valid_vals) if len(valid_vals) > 0 and np.var(valid_vals) > 0 else 1.0
                train_y_list.append(-((vals - target) ** 2) / variance)

    if not df_clean.empty:
        train_y = torch.tensor(np.stack(train_y_list, axis=1), dtype=torch.float32).to(device)
    else:
        train_y = torch.zeros((0, len(objectives_config)), dtype=torch.float32).to(device)

    weights_t = torch.tensor(weights_list, dtype=torch.float32).to(device)
    batch_size = int(float(tweaks.get('batch_size', 1)))
    avoid_reval = tweaks.get('avoid_reval', True)
    acq_type = tweaks.get('acq_type', 'EI')
    exploration = validate_exploration(tweaks.get('exploration', 0.5))

    constraints_cfg = tweaks.get('constraints', {})
    num_rules = constraints_cfg.get('numerical', [])
    disallowed_rules = constraints_cfg.get('disallowed', [])

    # -------------------------------------------------------------
    # 4. Optimization Logic: Cold Start vs GP Fitting
    # -------------------------------------------------------------
    if is_cold_start:
        n_pool = max(batch_size * 500, 2000)
        pool = draw_sobol_samples(bounds=bounds_t, n=1, q=n_pool).squeeze(0)
        if num_rules or disallowed_rules:
            feasible = []
            for pt in pool:
                decoded = {}
                for idx_f, f in enumerate(features_config):
                    v = pt[idx_f].item()
                    if f['type'] == 'categorical':
                        choices = get_categorical_choices(f)
                        v = choices[max(0, min(len(choices) - 1, int(round(v))))]
                    elif f['type'] in ('discrete', 'regular'):
                        matches = re.findall(r"[-+]?\d*\.?\d+", f['range'])
                        if f['type'] == 'discrete':
                            d_vals = sorted([float(x.strip()) for x in f['range'].replace('[','').replace(']','').split(',') if x.strip()])
                        elif len(matches) >= 3:
                            lo, hi, st = sorted([float(matches[0]), float(matches[1])]) + [int(float(matches[2]))]
                            d_vals = np.linspace(lo, hi, st).tolist()
                        elif len(matches) == 2:
                            d_vals = np.linspace(float(matches[0]), float(matches[1]), 10).tolist()
                        else:
                            d_vals = []
                        if d_vals:
                            v = min(d_vals, key=lambda x: abs(x - v))
                    if isinstance(v, (int, float)):
                        v = round(float(v), 6)
                    decoded[f['name']] = v
                if check_constraints(decoded, num_rules, disallowed_rules, features_config):
                    feasible.append(pt)
                if len(feasible) >= batch_size:
                    break
            if feasible:
                candidates = torch.stack(feasible[:batch_size])
            else:
                candidates = draw_sobol_samples(bounds=bounds_t, n=1, q=batch_size).squeeze(0)
        else:
            candidates = draw_sobol_samples(bounds=bounds_t, n=1, q=batch_size).squeeze(0)
        
        # Format cold start suggestions
        def decode_cand_cold(pt):
            cand_dict = {}
            for idx_f, f in enumerate(features_config):
                raw_val = pt[idx_f].item()
                if f['type'] == 'categorical':
                    choices = get_categorical_choices(f)
                    v = choices[max(0, min(len(choices) - 1, int(round(raw_val))))]
                elif f['type'] in ('discrete', 'regular'):
                    matches = re.findall(r"[-+]?\d*\.?\d+", f['range'])
                    if f['type'] == 'discrete':
                        d_vals = sorted([float(x.strip()) for x in f['range'].replace('[','').replace(']','').split(',') if x.strip()])
                    elif len(matches) >= 3:
                        lo, hi, st = sorted([float(matches[0]), float(matches[1])]) + [int(float(matches[2]))]
                        d_vals = np.linspace(lo, hi, st).tolist()
                    elif len(matches) == 2:
                        d_vals = np.linspace(float(matches[0]), float(matches[1]), 10).tolist()
                    else:
                        d_vals = []
                    v = min(d_vals, key=lambda x: abs(x - raw_val)) if d_vals else raw_val
                    decimals = feature_precision.get(f['name'], 2)
                    v = round(float(v), decimals)
                else:
                    decimals = feature_precision.get(f['name'], 2)
                    v = round(float(raw_val), decimals)
                cand_dict[f['name']] = v
            return cand_dict

        suggestions = [decode_cand_cold(pt) for pt in candidates]
        constraint_status = {
            'constrained': bool(objective_constraints),
            'cold_start': bool(objective_constraints),
            'all_likely_feasible': None if objective_constraints else True,
            'candidates': [],
        }
        if objective_constraints:
            constraint_status['message'] = (
                'Cold Start: objective constraints will be evaluated once valid '
                'prior or experimental objective observations are available.'
            )
        return suggestions, {}, transfer_diagnostics, constraint_status

    # -------------------------------------------------------------
    # 5. Fit Transfer-Weighted Gaussian Process Model
    # -------------------------------------------------------------
    kernel_name = tweaks.get('kernel', 'matern52')
    is_noiseless = tweaks.get('noiseless', False)

    base_noise_var = 1e-4 if is_noiseless else 1e-2
    # Heteroscedastic noise: Yvar = base_noise_var / (weight + 1e-4)
    yvar_vals = np.clip(base_noise_var / (combined_weights + 1e-4), base_noise_var, 50.0)
    train_yvar = torch.tensor(yvar_vals, dtype=torch.float32, device=compute_device).unsqueeze(-1)

    # Select exactly the requested number of highest-relevance eligible rows.
    requested_max_rows = int(max_gp_rows or tweaks.get('max_gp_rows') or MAX_GP_ROWS)
    model_x, model_y, model_yvar, _, model_indices = _subsample_training_data(
        train_x,
        train_y,
        train_yvar,
        combined_weights,
        max_rows=requested_max_rows,
        eligible_mask=eligible_mask,
    )
    if model_x.shape[0] < MIN_GP_TRAINING_ROWS:
        raise ValueError(
            'Too few rows satisfy the configured Prior rules. '
            'Broaden a numerical tolerance or SMILES similarity range, or use Average mode.'
        )

    transfer_diagnostics.update({
        'requested_gp_rows': requested_max_rows,
        'model_rows_used': int(model_x.shape[0]),
        'rows_downsampled': int(train_x.shape[0] - model_x.shape[0]),
        'permitted_max_rows': int(len(df)),
        'deterministic_subsample': True,
        'training_row_indices': [int(df_clean.index[index]) for index in model_indices],
    })

    def get_kernel(k_name):
        if k_name == 'rbf':      return ScaleKernel(RBFKernel()).to(device)
        if k_name == 'matern32': return ScaleKernel(MaternKernel(nu=1.5)).to(device)
        if k_name == 'matern12': return ScaleKernel(MaternKernel(nu=0.5)).to(device)
        return ScaleKernel(MaternKernel(nu=2.5)).to(device)

    def build_single_model(tx, ty_slice, tyvar_slice):
        all_dims = list(range(tx.shape[1]))
        cont_dims = [idx for idx in all_dims if idx not in categorical_features]

        # Normalize bounds must be [2, d] (full feature count) when indices= is used
        input_tf = (
            Normalize(
                d=tx.shape[1],
                bounds=bounds_t.to(compute_device),
                indices=cont_dims
            ).to(compute_device)
            if cont_dims else None
        )

        if categorical_features and len(cont_dims) > 0:
            m = MixedSingleTaskGP(
                tx.to(compute_device),
                ty_slice.to(compute_device),
                cat_dims=categorical_features,
                train_Yvar=tyvar_slice.to(compute_device),
                outcome_transform=Standardize(m=1),
                input_transform=input_tf
            ).to(compute_device)
        else:
            m = SingleTaskGP(
                tx.to(compute_device),
                ty_slice.to(compute_device),
                train_Yvar=tyvar_slice.to(compute_device),
                covar_module=get_kernel(kernel_name).to(compute_device),
                outcome_transform=Standardize(m=1),
                input_transform=input_tf
            ).to(compute_device)

        return m.to(compute_device)

    if model_y.shape[1] > 1:
        models = []
        for idx in range(model_y.shape[1]):
            check_control()
            y_slice = model_y[:, idx:idx+1]
            non_nan_mask = ~torch.isnan(y_slice.squeeze(-1))
            tx_i = model_x[non_nan_mask]
            ty_i = y_slice[non_nan_mask]
            tyvar_i = model_yvar[non_nan_mask]
            if tx_i.shape[0] == 0:
                raise ValueError(f"No valid observations for objective '{objectives_config[idx]['name']}'.")
            models.append(build_single_model(tx_i, ty_i, tyvar_i))

        model = ModelListGP(*models).to(compute_device)
        mll = SumMarginalLogLikelihood(model.likelihood, model).to(compute_device)
    else:
        model = build_single_model(model_x, model_y, model_yvar).to(compute_device)
        mll = ExactMarginalLogLikelihood(model.likelihood, model).to(compute_device)

    try:
        check_control()
        fit_iterations = 60 if model_x.shape[0] <= 400 else 35
        fit_gpytorch_mll(
            mll,
            optimizer_kwargs={
                "options": {"maxiter": fit_iterations},
                "callback": lambda *_: check_control(),
                "timeout_sec": _remaining_transfer_seconds(deadline_monotonic),
            },
        )
        check_control()
    except (TransferCalculationCancelled, TransferGPFitTimeout):
        raise
    except Exception as e:
        raise TransferGPFitError(f"Transfer GP model fitting failed: {e}") from e

    OPT_DEVICE = device if device.type == 'cuda' else compute_device
    model = model.to(OPT_DEVICE)

    # -------------------------------------------------------------
    # 6. Acquisition Function Construction & Optimization
    # -------------------------------------------------------------
    is_mo = model_y.shape[1] > 1
    is_uniform = torch.allclose(weights_t, weights_t[0].expand_as(weights_t), atol=1e-2) if is_mo else True

    acq_is_logarithmic = False
    if is_mo and is_uniform:
        mc_samples = 64 if model_x.shape[0] <= 400 else 32
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]))
        ref_point = model_y.min(dim=0).values - 0.1 * (model_y.max(dim=0).values - model_y.min(dim=0).values).abs()
        ref_point = torch.where(torch.isnan(ref_point), torch.zeros_like(ref_point), ref_point).to(OPT_DEVICE)
        acq_func = qLogNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point,
            X_baseline=model_x.to(OPT_DEVICE),
            sampler=sampler,
            prune_baseline=True
        )
        acq_is_logarithmic = True
    elif is_mo and not is_uniform:
        mc_samples = 64 if model_x.shape[0] <= 400 else 32
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]))
        norm_weights = weights_t / weights_t.sum()
        scalarized_obj = ScalarizedPosteriorTransform(weights=norm_weights.to(OPT_DEVICE))
        scalarized_y = (model_y.to(OPT_DEVICE) * norm_weights.to(OPT_DEVICE)).sum(dim=-1)
        best_f = scalarized_y.max()
        acq_func = qLogExpectedImprovement(
            model=model,
            best_f=best_f,
            sampler=sampler,
            posterior_transform=scalarized_obj
        )
        acq_is_logarithmic = True
    else:
        best_f = model_y.max().to(OPT_DEVICE)
        if batch_size > 1:
            mc_samples = 64 if model_x.shape[0] <= 400 else 32
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]))
            if acq_type == 'UCB':
                acq_func = qUpperConfidenceBound(
                    model=model,
                    beta=transfer_exploration_to_beta(exploration),
                    sampler=sampler,
                )
            else:
                acq_func = qLogExpectedImprovement(model=model, best_f=best_f, sampler=sampler)
                acq_is_logarithmic = True
        else:
            if acq_type == 'UCB':
                acq_func = UpperConfidenceBound(
                    model=model,
                    beta=transfer_exploration_to_beta(exploration),
                )
            else:
                acq_func = ExpectedImprovement(model=model, best_f=best_f)

    feasibility_models = {}
    if objective_constraints:
        constrained_names = {c['objective'] for c in objective_constraints}
        model_indices_np = np.asarray(model_indices, dtype=int)
        for objective_name in constrained_names:
            check_control()
            raw_values_all = pd.to_numeric(
                df_clean[objective_name], errors='coerce'
            ).values
            raw_values = raw_values_all[model_indices_np]
            valid_mask_np = ~np.isnan(raw_values)
            valid_mask = torch.tensor(
                valid_mask_np, dtype=torch.bool, device=model_x.device
            )
            raw_x = model_x[valid_mask]
            raw_y = torch.tensor(
                raw_values[valid_mask_np], dtype=torch.float32,
                device=compute_device
            ).unsqueeze(-1)
            raw_yvar = model_yvar[valid_mask]
            if raw_x.shape[0] == 0:
                raise ValueError(
                    f"No valid observations for constrained objective '{objective_name}'."
                )
            raw_model = build_single_model(raw_x, raw_y, raw_yvar)
            raw_mll = ExactMarginalLogLikelihood(
                raw_model.likelihood, raw_model
            ).to(compute_device)
            try:
                fit_gpytorch_mll(
                    raw_mll,
                    optimizer_kwargs={
                        "options": {"maxiter": 75},
                        "callback": lambda *_: check_control(),
                        "timeout_sec": _remaining_transfer_seconds(deadline_monotonic),
                    },
                )
                check_control()
            except (TransferCalculationCancelled, TransferGPFitTimeout):
                raise
            except Exception as exc:
                raise TransferGPFitError(
                    f"Transfer feasibility model fitting failed for "
                    f"{objective_name}: {exc}"
                ) from exc
            raw_model.eval()
            feasibility_models[objective_name] = raw_model.to(OPT_DEVICE)

        acq_func = ProbabilityOfFeasibilityAcquisition(
            acq_func,
            feasibility_models,
            objective_constraints,
            logarithmic=acq_is_logarithmic,
        ).to(OPT_DEVICE)

    # Setup optimization bounds & constraints
    bounds_opt = bounds_t.to(OPT_DEVICE)

    # Generate feasible candidates
    num_restarts = 8 if model_x.shape[0] <= 400 else 5
    raw_samples = 192 if model_x.shape[0] <= 400 else 96

    try:
        check_control()
        candidates_raw, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=bounds_opt,
            q=batch_size,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            timeout_sec=_remaining_transfer_seconds(deadline_monotonic),
        )
        check_control()
    except (TransferCalculationCancelled, TransferGPFitTimeout):
        raise
    except Exception as e:
        print(f"Acquisition optimization fallback: {e}")
        candidates_raw = draw_sobol_samples(bounds=bounds_opt, n=1, q=batch_size).squeeze(0)

    # -------------------------------------------------------------
    # 7. Decode Candidates to Feature Values
    # -------------------------------------------------------------
    def decode_cand(pt):
        cand_dict = {}
        for idx_f, f in enumerate(features_config):
            raw_val = pt[idx_f].item()
            if f['type'] == 'categorical':
                choices = get_categorical_choices(f)
                v = choices[max(0, min(len(choices) - 1, int(round(raw_val))))]
            elif f['type'] in ('discrete', 'regular'):
                matches = re.findall(r"[-+]?\d*\.?\d+", f['range'])
                if f['type'] == 'discrete':
                    d_vals = sorted([float(x.strip()) for x in f['range'].replace('[','').replace(']','').split(',') if x.strip()])
                elif len(matches) >= 3:
                    lo, hi, st = sorted([float(matches[0]), float(matches[1])]) + [int(float(matches[2]))]
                    d_vals = np.linspace(lo, hi, st).tolist()
                elif len(matches) == 2:
                    d_vals = np.linspace(float(matches[0]), float(matches[1]), 10).tolist()
                else:
                    d_vals = []
                v = min(d_vals, key=lambda x: abs(x - raw_val)) if d_vals else raw_val
                decimals = feature_precision.get(f['name'], 2)
                v = round(float(v), decimals)
            else:
                decimals = feature_precision.get(f['name'], 2)
                v = round(float(raw_val), decimals)
            cand_dict[f['name']] = v
        return cand_dict

    suggestions = [decode_cand(pt) for pt in candidates_raw]

    # Filter duplicate suggestions against historical data if avoid_reval is True
    if avoid_reval and not df_clean.empty:
        # Check against df_clean feature columns
        pass

    # -------------------------------------------------------------
    # 8. Compute GP Model Fit Quality (R2, MSE)
    # -------------------------------------------------------------
    performance = {}
    with torch.no_grad():
        try:
            check_control()
            pred_posterior = model.posterior(model_x.to(OPT_DEVICE))
            pred_mean = pred_posterior.mean.cpu().numpy()
            actual_y = model_y.cpu().numpy()

            for idx, obj in enumerate(objectives_config):
                obj_name = obj['name']
                y_act = actual_y[:, idx]
                y_pr = pred_mean[:, idx] if pred_mean.ndim > 1 else pred_mean

                # Invert minimize/target transformation for reporting
                if obj['type'] == 'minimize':
                    y_act = -y_act
                    y_pr = -y_pr

                ss_res = np.sum((y_act - y_pr) ** 2)
                ss_tot = np.sum((y_act - np.mean(y_act)) ** 2)
                r2 = 1.0 - (ss_res / (ss_tot + 1e-9)) if ss_tot > 1e-9 else 1.0
                mse = float(np.mean((y_act - y_pr) ** 2))

                performance[obj_name] = {
                    "r2": round(float(r2), 3),
                    "mse": round(mse, 4),
                    "actual": y_act.tolist(),
                    "predicted": y_pr.tolist()
                }
        except Exception as e:
            print(f"Performance calculation error: {e}")

    constraint_status = {
        'constrained': bool(objective_constraints),
        'all_likely_feasible': True,
        'candidates': [],
    }
    if objective_constraints and suggestions:
        encoded_suggestions = []
        for suggestion in suggestions:
            encoded = []
            for feature in features_config:
                value = suggestion[feature['name']]
                if feature['type'] == 'categorical':
                    choices = get_categorical_choices(feature)
                    value = choices.index(str(value))
                encoded.append(float(value))
            encoded_suggestions.append(encoded)
        report_tensor = torch.tensor(
            encoded_suggestions, dtype=torch.float32, device=OPT_DEVICE
        )
        constraint_status = candidate_feasibility_report(
            feasibility_models,
            objective_constraints,
            report_tensor,
        )
        if not constraint_status['all_likely_feasible']:
            best_probability = max(
                (item['probability'] for item in constraint_status['candidates']),
                default=0.0,
            )
            constraint_status['message'] = (
                '⚠️ Feasibility Warning: at least one recommended candidate '
                'does not confidently satisfy every objective constraint '
                f'(best predicted feasibility: {best_probability:.1%}).'
            )

    return suggestions, performance, transfer_diagnostics, constraint_status


def run_transfer_bo_estimation(
    df,
    input_values,
    features_config,
    objectives_config,
    priors_config,
    tweaks,
    device=torch.device('cpu'),
    compute_device=torch.device('cpu'),
    prior_combine="multiply",
    max_gp_rows=None,
    cancel_event=None,
    deadline_monotonic=None,
):
    """
    Evaluates point estimation for the transfer-weighted Gaussian Process model.
    """
    def check_control():
        _transfer_control_check(cancel_event, deadline_monotonic)

    check_control()
    df_clean = clean_dataframe(df, features_config, objectives_config)
    if df_clean.empty:
        raise ValueError("No valid data available for estimation.")

    # Calculate transfer weights using estimator Prior inputs when supplied.
    estimation_priors = []
    for prior in priors_config:
        check_control()
        configured = dict(prior)
        name = configured.get('name')
        if input_values and input_values.get(name) not in (None, ''):
            configured['target_value'] = input_values[name]
        estimation_priors.append(configured)
    relevance = compute_combined_prior_relevance(
        df_clean,
        estimation_priors,
        method=prior_combine,
    )
    combined_weights = relevance['combined_weights']
    eligible_mask = relevance['eligible_mask']

    base_noise_var = 1e-4 if tweaks.get('noiseless', False) else 1e-2
    yvar_vals = np.clip(base_noise_var / (combined_weights + 1e-4), base_noise_var, 50.0)
    train_yvar = torch.tensor(yvar_vals, dtype=torch.float32, device=compute_device).unsqueeze(-1)

    # Encode training features and objectives
    train_x_list = []
    bounds = []
    categorical_features = []
    for i, f in enumerate(features_config):
        name = f['name']
        f_range = f['range']
        if f['type'] == 'continuous':
            matches = re.findall(r"[-+]?\d*\.?\d+", f_range)
            low, high = sorted([float(matches[0]), float(matches[1])]) if len(matches) >= 2 else (0.0, 1.0)
            bounds.append([low, high])
            train_x_list.append(pd.to_numeric(df_clean[name], errors='coerce').fillna(0).values)
        elif f['type'] in ('discrete', 'regular'):
            matches = re.findall(r"[-+]?\d*\.?\d+", f_range)
            low = float(matches[0]) if matches else 0.0
            high = float(matches[1]) if len(matches) >= 2 else 1.0
            bounds.append([low, high])
            train_x_list.append(pd.to_numeric(df_clean[name], errors='coerce').fillna(low).values)
        elif f['type'] == 'categorical':
            encoded, _ = encode_categorical_training_series(df_clean[name], f)
            choices = get_categorical_choices(f)
            train_x_list.append(encoded)
            bounds.append([0, len(choices) - 1])
            categorical_features.append(i)

    train_x = torch.tensor(np.stack(train_x_list, axis=1), dtype=torch.float32).to(compute_device)
    bounds_t = torch.tensor(bounds, dtype=torch.float32).T.to(compute_device)

    # Encode test point
    test_x_list = []
    for i, f in enumerate(features_config):
        raw_val = input_values.get(f['name'])
        if f['type'] == 'categorical':
            choices = get_categorical_choices(f)
            if str(raw_val) not in choices:
                raise ValueError(f"Feature '{f['name']}' has unknown categorical value {raw_val!r}.")
            idx = choices.index(str(raw_val))
            test_x_list.append(float(idx))
        else:
            test_x_list.append(float(raw_val) if raw_val is not None else 0.0)
    test_x = torch.tensor([test_x_list], dtype=torch.float32).to(compute_device)

    # Encode objectives
    train_y_list = []
    for obj in objectives_config:
        vals = pd.to_numeric(df_clean[obj['name']], errors='coerce').values
        train_y_list.append(vals)
    train_y = torch.tensor(np.stack(train_y_list, axis=1), dtype=torch.float32).to(compute_device)

    # Use the same explicit Prior eligibility and row ceiling as optimization.
    requested_max_rows = int(max_gp_rows or tweaks.get('max_gp_rows') or MAX_GP_ROWS)
    reused_training = _reuse_training_data(
        train_x,
        train_y,
        train_yvar,
        df_clean.index.to_numpy(),
        tweaks.get('reuse_training_row_indices'),
    )
    if reused_training is not None:
        model_x, model_y, model_yvar, model_indices = reused_training
        reused_training_rows = True
    else:
        model_x, model_y, model_yvar, _, model_indices = _subsample_training_data(
            train_x,
            train_y,
            train_yvar,
            combined_weights,
            max_rows=requested_max_rows,
            eligible_mask=eligible_mask,
        )
        reused_training_rows = False
    eligible_rows = relevance['eligible_rows']
    if model_x.shape[0] < MIN_GP_TRAINING_ROWS:
        raise ValueError(
            'Too few rows satisfy the configured Prior rules for estimation. '
            'Broaden a numerical tolerance or SMILES similarity range, or use Average mode.'
        )

    def get_kernel(k_name):
        if k_name == 'rbf':      return ScaleKernel(RBFKernel()).to(compute_device)
        if k_name == 'matern32': return ScaleKernel(MaternKernel(nu=1.5)).to(compute_device)
        if k_name == 'matern12': return ScaleKernel(MaternKernel(nu=0.5)).to(compute_device)
        return ScaleKernel(MaternKernel(nu=2.5)).to(compute_device)

    def build_single_model(tx, ty_slice, tyvar_slice):
        all_dims = list(range(tx.shape[1]))
        cont_dims = [idx for idx in all_dims if idx not in categorical_features]
        input_tf = (
            Normalize(d=tx.shape[1], bounds=bounds_t, indices=cont_dims).to(compute_device)
            if cont_dims else None
        )
        if categorical_features and len(cont_dims) > 0:
            return MixedSingleTaskGP(
                tx.to(compute_device),
                ty_slice.to(compute_device),
                cat_dims=categorical_features,
                train_Yvar=tyvar_slice.to(compute_device),
                outcome_transform=Standardize(m=1),
                input_transform=input_tf
            ).to(compute_device)
        else:
            return SingleTaskGP(
                tx.to(compute_device),
                ty_slice.to(compute_device),
                train_Yvar=tyvar_slice.to(compute_device),
                covar_module=get_kernel(tweaks.get('kernel', 'matern52')).to(compute_device),
                outcome_transform=Standardize(m=1),
                input_transform=input_tf
            ).to(compute_device)

    if model_y.shape[1] > 1:
        models = []
        for idx in range(model_y.shape[1]):
            check_control()
            y_slice = model_y[:, idx:idx+1]
            non_nan_mask = ~torch.isnan(y_slice.squeeze(-1))
            tx_i = model_x[non_nan_mask]
            ty_i = y_slice[non_nan_mask]
            tyvar_i = model_yvar[non_nan_mask]
            if tx_i.shape[0] == 0:
                raise ValueError(f"No valid observations for objective '{objectives_config[idx]['name']}'.")
            models.append(build_single_model(tx_i, ty_i, tyvar_i))
        model = ModelListGP(*models).to(compute_device)
        mll = SumMarginalLogLikelihood(model.likelihood, model).to(compute_device)
    else:
        model = build_single_model(model_x, model_y, model_yvar).to(compute_device)
        mll = ExactMarginalLogLikelihood(model.likelihood, model).to(compute_device)

    try:
        check_control()
        fit_gpytorch_mll(
            mll,
            optimizer_kwargs={
                "options": {"maxiter": 40},
                "callback": lambda *_: check_control(),
                "timeout_sec": _remaining_transfer_seconds(deadline_monotonic),
            },
        )
        check_control()
    except (TransferCalculationCancelled, TransferGPFitTimeout):
        raise
    except Exception as exc:
        raise TransferGPFitError(
            f"Transfer estimation GP model fitting failed: {exc}"
        ) from exc

    check_control()
    with torch.no_grad():
        post = model.posterior(test_x)
        means = post.mean.squeeze(0).cpu().numpy().tolist()
        stds = torch.sqrt(post.variance).squeeze(0).cpu().numpy().tolist()

    predictions = {}
    results = {}
    for idx, obj in enumerate(objectives_config):
        objective_name = obj['name']
        prediction = round(float(means[idx]), 3)
        predictions[objective_name] = prediction
        results[obj['name']] = {
            "mean": prediction,
            "std": round(float(stds[idx]), 3),
        }

    _, violations = check_objective_constraints(
        predictions, tweaks.get('objective_constraints', [])
    )
    violated_objectives = {item['objective'] for item in violations}
    scores = []
    weights = []
    for obj in objectives_config:
        name = obj['name']
        value = predictions[name]
        train_values = pd.to_numeric(df_clean[name], errors='coerce').dropna().values
        if len(train_values) == 0:
            score = 0.0
        else:
            objective_type = obj.get('type', 'maximize')
            if objective_type == 'minimize':
                scored_value = -value
                scored_train = -train_values
            elif objective_type == 'target':
                target = float(obj.get('target', 0) or 0)
                scored_value = -abs(value - target)
                scored_train = -np.abs(train_values - target)
            else:
                scored_value = value
                scored_train = train_values
            minimum, maximum = scored_train.min(), scored_train.max()
            normalized = (
                (scored_value - minimum) / (maximum - minimum)
                if maximum > minimum else 1.0
            )
            score = float(np.clip(normalized, 0.0, 1.0) * 100.0)
        if name in violated_objectives:
            score = 0.0
        results[name]['success'] = round(score, 1)
        scores.append(score / 100.0)
        weights.append(float(obj.get('importance', 1) or 1))

    global_score = (
        float(np.average(scores, weights=weights)) * 100.0 if scores else 0.0
    )
    workload = {
        'requested_gp_rows': requested_max_rows,
        'model_rows_used': int(model_x.shape[0]),
        'rows_downsampled': int(train_x.shape[0] - model_x.shape[0]),
        'eligible_rows': eligible_rows,
        'permitted_max_rows': int(len(df)),
        'deterministic_subsample': True,
        'reused_training_rows': reused_training_rows,
        'training_row_indices': [int(df_clean.index[index]) for index in model_indices],
    }
    return results, round(global_score, 1), violations, workload


def run_transfer_bo_surface(
    df,
    features_config,
    objectives_config,
    priors_config,
    tweaks,
    x_feat,
    y_feat,
    z_axis=None,
    color_obj=None,
    fixed_values=None,
    resolution=25,
    device=torch.device('cpu'),
    compute_device=torch.device('cpu'),
    prior_combine="multiply",
    max_gp_rows=None,
    cancel_event=None,
    progress_callback=None,
):
    """
    Computes a 2D prediction grid (response surface) using the transfer-weighted GP surrogate.
    Prior columns are excluded from axis options (context variables).
    """
    def check_cancelled():
        if cancel_event is not None and cancel_event.is_set():
            raise SurfaceCalculationCancelled('Response-surface calculation cancelled by user.')

    def report(phase, percent):
        if progress_callback is not None:
            progress_callback(phase, percent)

    check_cancelled()
    report('Preparing transfer response-surface data', 10)
    # Auto-cap resolution for large datasets to maintain fast rendering
    if resolution > 20 and len(df) > 2000:
        resolution = 20
    resolution = max(5, min(int(resolution), 50))

    # Determine which objectives to map
    obj_names = [o['name'] for o in objectives_config if o['name'] in df.columns]
    if not obj_names:
        raise ValueError("No valid objectives found for response surface.")

    if not color_obj or color_obj not in obj_names:
        color_obj = obj_names[0]

    num_feat_names = [
        f['name'] for f in features_config
        if f['name'] in df.columns and f.get('type', 'continuous') != 'categorical'
    ]
    valid_z_options = obj_names + [f for f in num_feat_names if f != x_feat and f != y_feat]
    if not z_axis or z_axis not in valid_z_options:
        z_axis = color_obj

    needed_objs = [{'name': color_obj, 'type': 'maximize'}]
    if z_axis in obj_names and z_axis != color_obj:
        needed_objs.append({'name': z_axis, 'type': 'maximize'})

    # Clean dataframe specifically ensuring needed objectives are present and not NaN
    df_clean = clean_dataframe(df, features_config, needed_objs)
    if df_clean.empty:
        raise ValueError("No valid data points left after filtering missing features/objectives.")

    # Calculate the same authoritative Prior profile used by optimization.
    check_cancelled()
    relevance = compute_combined_prior_relevance(
        df_clean,
        priors_config,
        method=prior_combine,
    )
    combined_weights = relevance['combined_weights']
    eligible_mask = relevance['eligible_mask']

    base_noise_var = 1e-4 if tweaks.get('noiseless', False) else 1e-2
    yvar_vals = np.clip(base_noise_var / (combined_weights + 1e-4), base_noise_var, 50.0)
    train_yvar = torch.tensor(yvar_vals, dtype=torch.float32, device=compute_device).unsqueeze(-1)

    # Setup features and grid
    train_x_list = []
    bounds = []
    categorical_features = []
    feat_map = {}
    feat_ranges = {}

    for i, f in enumerate(features_config):
        name = f['name']
        f_range = f['range']
        feat_map[name] = (i, f)

        if f['type'] == 'continuous':
            matches = re.findall(r"[-+]?\d*\.?\d+", f_range)
            low, high = sorted([float(matches[0]), float(matches[1])]) if len(matches) >= 2 else (0.0, 1.0)
            bounds.append([low, high])
            train_x_list.append(pd.to_numeric(df_clean[name], errors='coerce').fillna(0).values)
            feat_ranges[name] = (low, high, 'continuous', None)
        elif f['type'] in ('discrete', 'regular'):
            matches = re.findall(r"[-+]?\d*\.?\d+", f_range)
            low = float(matches[0]) if matches else 0.0
            high = float(matches[1]) if len(matches) >= 2 else 1.0
            bounds.append([low, high])
            train_x_list.append(pd.to_numeric(df_clean[name], errors='coerce').fillna(low).values)
            feat_ranges[name] = (low, high, f['type'], None)
        elif f['type'] == 'categorical':
            encoded, _ = encode_categorical_training_series(df_clean[name], f)
            choices = get_categorical_choices(f)
            train_x_list.append(encoded)
            bounds.append([0, len(choices) - 1])
            categorical_features.append(i)
            feat_ranges[name] = (0, len(choices) - 1, 'categorical', choices)

    train_x = torch.tensor(np.stack(train_x_list, axis=1), dtype=torch.float32).to(compute_device)
    bounds_t = torch.tensor(bounds, dtype=torch.float32).T.to(compute_device)

    # Build meshgrid for x_feat and y_feat
    x_idx, x_cfg = feat_map[x_feat]
    y_idx, y_cfg = feat_map[y_feat]

    x_low, x_high = bounds[x_idx]
    y_low, y_high = bounds[y_idx]

    x_vals = np.linspace(x_low, x_high, resolution)
    y_vals = np.linspace(y_low, y_high, resolution)
    xx, yy = np.meshgrid(x_vals, y_vals)

    grid_pts = []
    fixed_values = fixed_values or {}
    for xi, yi in zip(xx.ravel(), yy.ravel()):
        pt = []
        for i, f in enumerate(features_config):
            fname = f['name']
            if fname == x_feat:
                pt.append(xi)
            elif fname == y_feat:
                pt.append(yi)
            elif fname in fixed_values:
                val = fixed_values[fname]
                if f.get('type') == 'categorical':
                    choices = get_categorical_choices(f)
                    if str(val) not in choices:
                        raise ValueError(f"Feature '{f['name']}' has unknown categorical value {val!r}.")
                    idx = choices.index(str(val))
                    pt.append(float(idx))
                else:
                    try:
                        pt.append(float(val))
                    except (ValueError, TypeError):
                        pt.append(float(bounds[i][0]))
            else:
                if f.get('type') == 'categorical':
                    pt.append(0.0)
                else:
                    pt.append(float(bounds[i][0] + bounds[i][1]) / 2.0)
        grid_pts.append(pt)

    grid_tensor = torch.tensor(grid_pts, dtype=torch.float32).to(compute_device)
    check_cancelled()
    report('Building transfer response-surface grid', 25)
    requested_max_rows = int(max_gp_rows or tweaks.get('max_gp_rows') or MAX_GP_ROWS)
    eligible_rows = relevance['eligible_rows']
    model_rows_used = min(requested_max_rows, eligible_rows)
    if model_rows_used < MIN_GP_TRAINING_ROWS:
        raise ValueError(
            'Too few rows satisfy the configured Prior rules for plotting. '
            'Broaden a numerical tolerance or SMILES similarity range, or use Average mode.'
        )

    # Helper to fit a transfer GP and evaluate on grid_tensor
    def fit_and_eval_target(target_name, position, total):
        check_cancelled()
        report(
            f'Fitting transfer surrogate for {target_name}',
            35 + int(45 * (position / max(total, 1))),
        )
        train_y_vals = pd.to_numeric(df_clean[target_name], errors='coerce').fillna(0).values
        train_y = torch.tensor(train_y_vals, dtype=torch.float32, device=compute_device).unsqueeze(-1)

        model_x, model_y, model_yvar, _, _ = _subsample_training_data(
            train_x,
            train_y,
            train_yvar,
            combined_weights,
            max_rows=requested_max_rows,
            eligible_mask=eligible_mask,
        )

        cont_dims = [idx for idx in range(model_x.shape[1]) if idx not in categorical_features]
        input_tf = Normalize(d=model_x.shape[1], bounds=bounds_t, indices=cont_dims).to(compute_device) if cont_dims else None

        if categorical_features and len(cont_dims) > 0:
            m = MixedSingleTaskGP(
                model_x, model_y,
                cat_dims=categorical_features,
                train_Yvar=model_yvar,
                outcome_transform=Standardize(m=1),
                input_transform=input_tf
            ).to(compute_device)
        else:
            m = SingleTaskGP(
                model_x, model_y,
                train_Yvar=model_yvar,
                outcome_transform=Standardize(m=1),
                input_transform=input_tf
            ).to(compute_device)

        mll = ExactMarginalLogLikelihood(m.likelihood, m)
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
        with torch.no_grad():
            post = m.posterior(grid_tensor)
            means = post.mean.cpu().numpy().squeeze(-1)
            stds = torch.sqrt(post.variance).cpu().numpy().squeeze(-1)
        return means.reshape(resolution, resolution), stds.reshape(resolution, resolution)

    predicted_objectives = [color_obj]
    if z_axis in obj_names and z_axis != color_obj:
        predicted_objectives.append(z_axis)

    color_grid, color_std_grid = fit_and_eval_target(
        color_obj, 0, len(predicted_objectives)
    )

    if z_axis in obj_names:
        if z_axis == color_obj:
            z_grid = color_grid
            std_grid = color_std_grid
        else:
            z_grid, std_grid = fit_and_eval_target(
                z_axis, 1, len(predicted_objectives)
            )
    elif z_axis in feat_map:
        z_i, _ = feat_map[z_axis]
        z_flat = grid_tensor[:, z_i].cpu().numpy()
        z_grid = z_flat.reshape(resolution, resolution)
        std_grid = np.zeros_like(z_grid)
    else:
        z_grid = color_grid
        std_grid = np.zeros_like(color_grid)

    # Plot observations must use the same transfer-relevance filter as the
    # surrogate. Otherwise a large, unrelated history can overwhelm the visual
    # even though those rows did not contribute to the fitted response surface.
    eligible_indices = np.flatnonzero(eligible_mask)

    # Legacy observation overlay fields are retained for API compatibility.
    # The richer experiments payload below is used by the browser.
    n_obs = len(eligible_indices)
    if n_obs > 2000:
        obs_probs = combined_weights[eligible_indices] + 1e-3
        obs_probs = obs_probs / np.sum(obs_probs)
        rng = np.random.default_rng(42)
        obs_idx = np.sort(rng.choice(eligible_indices, size=2000, replace=False, p=obs_probs))
        x_obs = df_clean[x_feat].values[obs_idx].tolist()
        y_obs = df_clean[y_feat].values[obs_idx].tolist()
        z_obs = df_clean[z_axis].values[obs_idx].tolist() if z_axis in df_clean.columns else df_clean[color_obj].values[obs_idx].tolist()
        color_obs = df_clean[color_obj].values[obs_idx].tolist()
        w_obs = combined_weights[obs_idx].tolist()
    else:
        x_obs = df_clean[x_feat].values[eligible_indices].tolist()
        y_obs = df_clean[y_feat].values[eligible_indices].tolist()
        z_obs = (df_clean[z_axis].values[eligible_indices].tolist()
                 if z_axis in df_clean.columns
                 else df_clean[color_obj].values[eligible_indices].tolist())
        color_obs = df_clean[color_obj].values[eligible_indices].tolist()
        w_obs = combined_weights[eligible_indices].tolist()

    # Match EDOS's response-surface context: experiments at the requested slice
    # are solid; other experiments are outlined. Keep rendering bounded for very
    # large datasets while sampling deterministically so the overlay is stable.
    report('Classifying experiments for the selected slice', 85)
    remaining_feats = [
        f for f in features_config if f['name'] != x_feat and f['name'] != y_feat
    ]
    slice_targets = {}
    for f in remaining_feats:
        fname = f['name']
        f_inf = feat_ranges.get(fname, (0.0, 1.0, 'continuous', None))
        raw_fixed = fixed_values.get(fname, None)
        if f_inf[2] == 'categorical':
            choices = f_inf[3] or []
            selected = (
                str(raw_fixed)
                if raw_fixed is not None and str(raw_fixed) in choices
                else (choices[0] if choices else '')
            )
            slice_targets[fname] = {
                'type': 'categorical',
                'target_val': selected,
                'label': selected,
            }
            continue

        try:
            selected = float(raw_fixed) if raw_fixed is not None else (f_inf[0] + f_inf[1]) / 2.0
        except (TypeError, ValueError):
            selected = (f_inf[0] + f_inf[1]) / 2.0
        values = pd.to_numeric(df_clean[fname], errors='coerce').dropna().unique()
        nearest = float(values[np.argmin(np.abs(values - selected))]) if len(values) else selected
        slice_targets[fname] = {
            'type': 'numerical',
            'target_val': nearest,
            'label': f'{nearest:g}',
        }

    in_slice_indices = []
    off_slice_indices = []
    for eligible_position, row_index in enumerate(eligible_indices):
        if eligible_position % 256 == 0:
            check_cancelled()
        row = df_clean.iloc[row_index]
        in_slice = True
        for f in remaining_feats:
            fname = f['name']
            target = slice_targets[fname]
            if target['type'] == 'categorical':
                if str(row[fname]) != target['target_val']:
                    in_slice = False
                    break
            else:
                try:
                    if abs(float(row[fname]) - target['target_val']) > 1e-6:
                        in_slice = False
                        break
                except (TypeError, ValueError):
                    in_slice = False
                    break
        (in_slice_indices if in_slice else off_slice_indices).append(row_index)

    def evenly_spaced(indices, limit):
        if len(indices) <= limit:
            return indices
        positions = np.linspace(0, len(indices) - 1, limit, dtype=int)
        return [indices[position] for position in positions]

    selected_in_slice = evenly_spaced(in_slice_indices, 750)
    selected_off_slice = evenly_spaced(off_slice_indices, 1250)

    def experiment_payload(row_index, in_slice):
        row = df_clean.iloc[row_index]
        conditions = []
        for f in remaining_feats:
            fname = f['name']
            if slice_targets[fname]['type'] == 'categorical':
                conditions.append(f'{fname}={row[fname]}')
            else:
                try:
                    conditions.append(f'{fname}={float(row[fname]):g}')
                except (TypeError, ValueError):
                    conditions.append(f'{fname}={row[fname]}')
        color_value = pd.to_numeric(pd.Series([row[color_obj]]), errors='coerce').iloc[0]
        z_source = row[z_axis] if z_axis in df_clean.columns else row[color_obj]
        z_value = pd.to_numeric(pd.Series([z_source]), errors='coerce').iloc[0]
        return {
            'exp_id': int(row_index + 1),
            'x': float(row[x_feat]),
            'y': float(row[y_feat]),
            'z': float(z_value) if pd.notna(z_value) else None,
            'color': float(color_value) if pd.notna(color_value) else None,
            'weight': float(combined_weights[row_index]),
            'in_slice': in_slice,
            'conditions': ', '.join(conditions),
        }

    experiments = [
        experiment_payload(index, True) for index in selected_in_slice
    ] + [
        experiment_payload(index, False) for index in selected_off_slice
    ]

    check_cancelled()
    report('Finalizing transfer response surface', 95)

    return {
        "x": x_vals.tolist(),
        "y": y_vals.tolist(),
        "z": z_grid.tolist(),
        "color": color_grid.tolist(),
        "std": std_grid.tolist(),
        "x_label": x_feat,
        "y_label": y_feat,
        "z_label": z_axis,
        "color_label": color_obj,
        "obs_x": x_obs,
        "obs_y": y_obs,
        "obs_z": z_obs,
        "obs_color": color_obs,
        "obs_weights": w_obs,
        "experiments": experiments,
        "slice_targets": {name: target['label'] for name, target in slice_targets.items()},
        "overlay": {
            "transfer_filtered_total": int(len(eligible_indices)),
            "in_slice_total": len(in_slice_indices),
            "off_slice_total": len(off_slice_indices),
            "in_slice_shown": len(selected_in_slice),
            "off_slice_shown": len(selected_off_slice),
        },
        "workload": {
            "requested_gp_rows": requested_max_rows,
            "model_rows_used": int(model_rows_used),
            "rows_downsampled": int(max(0, train_x.shape[0] - model_rows_used)),
            "eligible_rows": eligible_rows,
            "permitted_max_rows": int(len(df)),
            "deterministic_subsample": True,
        },
    }
