"""
TransEDOS - REST API Routes Blueprint
======================================
Defines all HTTP endpoints for file uploading, DoE, standard BO, Statistical Analysis,
and the new Transfer Learning BO engine (/transfer_optimize, /detect_prior_types, /transfer_surface).
"""

import os
import json
import sys
import math
import gc
import threading
import time
import torch
import numpy as np
import pandas as pd
from flask import Blueprint, Response, request, jsonify, render_template

# Core modules
from src.core.doe import generate_doe
from src.core.bo import (
    SurfaceCalculationCancelled,
    validate_exploration,
    run_bo_optimization,
    run_bo_estimation,
    run_bo_surface,
)
from src.core.bo_policy import (
    load_bo_budget,
    normalized_objective_weights,
    select_bo_policy,
    validate_bo_data_mode,
    validate_multi_objective_method,
)
from src.api.bo_worker import (
    BOChildFailure,
    BOProcessCancelled,
    BOProcessTimeout,
    run_bo_in_subprocess,
)
from src.core.sa import run_sa, estimate_sa, AnalysisCancelled
from src.core.constraints import (
    validate_feature_constraints,
    validate_feature_ranges,
    validate_objective_constraints,
)
from src.core.setup_csv import parse_setup_csv, serialize_setup_csv
from src.core.prior_utils import detect_prior_column_type, assert_rdkit
from src.core.transfer_bo import (
    TransferCalculationCancelled,
    TransferGPFitError,
    run_transfer_bo_optimization,
    run_transfer_bo_estimation,
    run_transfer_bo_surface,
    preview_transfer_selection,
)
from src.api.jobs import (
    cancel_background_job as _cancel_background_job,
    cleanup_expired_jobs as _cleanup_sa_jobs,
    get_background_job_response as _background_job_response,
    get_cancel_event,
    queue_background_job as _queue_background_job,
    update_background_job as _update_sa_job,
)

api_bp = Blueprint('api', __name__)

# Device selection (CUDA > CPU). DirectML is excluded because BoTorch
# quasirandom sampling and acquisition optimizers do not support DirectML private device kernels.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COMPUTE_DEVICE = torch.device("cpu")
# Response surfaces fit exact Gaussian processes. Their fitting cost grows
# cubically with the number of observations, so accepting an arbitrary upload
# here can freeze the only analysis worker for a long time.
BO_SURFACE_MAX_ROWS = 650
_REQUEST_JOB_IDS = {}
_REQUEST_JOB_IDS_LOCK = threading.Lock()
_TRANSFER_WATCHDOG_JOBS = set()
_TRANSFER_WATCHDOG_JOBS_LOCK = threading.Lock()


def _transfer_attempt_timeout_seconds():
    """Bound one exact-GP attempt so reduced-row recovery can take over."""
    try:
        configured = float(os.environ.get('TRANSFER_GP_ATTEMPT_TIMEOUT_SECONDS', '75'))
    except (TypeError, ValueError):
        configured = 75.0
    return max(5.0, min(configured, 600.0))


def _analysis_cancel_grace_seconds():
    try:
        configured = float(os.environ.get('ANALYSIS_JOB_CANCEL_GRACE_SECONDS', '30'))
    except (TypeError, ValueError):
        configured = 30.0
    return max(0.1, min(configured, 300.0))


def _release_job_slot_after_grace(job_id, grace_seconds=None):
    """Release capacity if a cooperatively cancelled worker never exits."""
    time.sleep(
        _analysis_cancel_grace_seconds()
        if grace_seconds is None else max(0.0, grace_seconds)
    )
    status = _background_job_response(job_id)
    if status and status.get('status') == 'cancelling':
        _update_sa_job(
            job_id,
            status='failed',
            phase='Cancellation timed out',
            error=(
                'The worker did not stop after cancellation; its analysis '
                'slot was released.'
            ),
            finished_at=time.time(),
        )


def _start_cancel_release_watchdog(job_id):
    threading.Thread(
        target=_release_job_slot_after_grace,
        args=(job_id,),
        name=f'cancel-watchdog-{job_id[:8]}',
        daemon=True,
    ).start()


def _watch_transfer_job(job_id):
    """Bound the complete retry sequence, including a wedged native call."""
    default_timeout = (_transfer_attempt_timeout_seconds() * 4) + 60.0
    try:
        hard_timeout = float(os.environ.get(
            'TRANSFER_JOB_HARD_TIMEOUT_SECONDS', default_timeout
        ))
    except (TypeError, ValueError):
        hard_timeout = default_timeout
    time.sleep(max(5.0, min(hard_timeout, 3600.0)))
    status = _background_job_response(job_id)
    if status and status.get('status') in {'queued', 'running'}:
        cancellation = _cancel_background_job(job_id)
        if cancellation and cancellation.get('status') == 'cancelling':
            _start_cancel_release_watchdog(job_id)


def _start_transfer_job_watchdog(job_id):
    with _TRANSFER_WATCHDOG_JOBS_LOCK:
        if job_id in _TRANSFER_WATCHDOG_JOBS:
            return
        _TRANSFER_WATCHDOG_JOBS.add(job_id)

    def watch_and_release():
        try:
            _watch_transfer_job(job_id)
        finally:
            with _TRANSFER_WATCHDOG_JOBS_LOCK:
                _TRANSFER_WATCHDOG_JOBS.discard(job_id)

    threading.Thread(
        target=watch_and_release,
        name=f'transfer-watchdog-{job_id[:8]}',
        daemon=True,
    ).start()


def _queue_idempotent_background_job(kind, payload, target, initial_phase):
    """Return the same job when a proxy retries an accepted submission."""
    request_id = (request.headers.get('X-Analysis-Request-ID') or '').strip()
    if len(request_id) > 128:
        raise ValueError('Analysis request identifier is too long.')
    if not request_id:
        return _queue_background_job(kind, payload, target, initial_phase)

    with _REQUEST_JOB_IDS_LOCK:
        existing_id = _REQUEST_JOB_IDS.get(request_id)
        if existing_id:
            if _background_job_response(existing_id) is not None:
                return existing_id
            _REQUEST_JOB_IDS.pop(request_id, None)

        job_id = _queue_background_job(kind, payload, target, initial_phase)
        if job_id is not None:
            _REQUEST_JOB_IDS[request_id] = job_id
        return job_id


def _as_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in ('1', 'true', 'yes', 'on')
    return bool(value)


def _has_configured_transfer_prior(priors):
    """Return true when a TransBO surface has an actual target condition."""
    if not isinstance(priors, list):
        return False
    for prior in priors:
        if not isinstance(prior, dict) or not prior.get('name'):
            continue
        target = prior.get('target_value')
        # Zero is a valid numerical target; only blank values are incomplete.
        if target is not None and str(target).strip() != '':
            return True
    return False


def _validate_transfer_surface_prerequisites(priors):
    if not _has_configured_transfer_prior(priors):
        raise ValueError(
            'Configure at least one Prior target before creating a TransBO '
            'response surface. The plot uses only rows filtered by the '
            'selected transfer conditions.'
        )


def _is_hosted_environment():
    environment = os.environ.get('ENVIRONMENT', '').strip().lower()
    hostname = (request.host or '').split(':', 1)[0].strip().lower()
    return (
        environment in ('production', 'prod')
        or bool(os.environ.get('SPACE_ID'))
        or bool(os.environ.get('HF_SPACE'))
        or hostname.endswith('.hf.space')
    )


def _validate_bo_surface_row_count(data):
    if isinstance(data, list) and len(data) > BO_SURFACE_MAX_ROWS:
        raise ValueError(
            f'BO response-surface plots are limited to {BO_SURFACE_MAX_ROWS:,} '
            f'dataset rows for reliable completion; this dataset has {len(data):,}. '
            'Filter or reduce the dataset before creating a contour or 3D plot.'
        )


def _is_recoverable_transfer_failure(exc):
    """Return whether retrying a smaller exact-GP fit is worth attempting."""
    if isinstance(exc, (MemoryError, TransferGPFitError)):
        return True
    message = str(exc).lower()
    recoverable_markers = (
        'out of memory', 'memory allocation', 'allocator', 'cholesky',
        'not positive definite', 'singular', 'linear algebra', 'linalg',
        'failed to converge', 'optimization failed',
    )
    return any(marker in message for marker in recoverable_markers)


def _run_transfer_with_row_recovery(
    run_operation,
    requested_rows,
    enabled,
    job_id=None,
    cancel_event=None,
    resolve_effective_rows=None,
):
    """Optionally retry a recoverable exact-GP failure with smaller row limits.

    A successful recovery is always returned with metadata; this helper never
    silently substitutes a smaller training set. A process killed by the host
    cannot be recovered because no Python code remains running to retry it.
    """
    requested = max(3, int(requested_rows or 3))
    try:
        return run_operation(requested), None
    except Exception as first_error:
        if not enabled or not _is_recoverable_transfer_failure(first_error):
            raise

        effective_requested = requested
        if resolve_effective_rows is not None:
            try:
                resolved_rows = int(resolve_effective_rows())
                if resolved_rows >= 3:
                    effective_requested = min(requested, resolved_rows)
            except Exception:
                # The original fitting failure remains authoritative. Recovery
                # can still proceed from the configured cap if previewing fails.
                effective_requested = requested

        attempted_limits = [effective_requested]
        current_limit = effective_requested
        last_error = first_error
        for attempt_number in range(1, 4):
            if cancel_event is not None and cancel_event.is_set():
                raise AnalysisCancelled('Transfer calculation cancelled by user.')
            retry_limit = max(3, (current_limit + 1) // 2)
            if retry_limit >= current_limit:
                break
            attempted_limits.append(retry_limit)
            if job_id:
                _update_sa_job(
                    job_id,
                    phase=(
                        f'Reduced-row recovery {attempt_number}/3: retrying exact GP '
                        f'with at most {retry_limit:,} rows'
                    ),
                    percent=50,
                )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            try:
                result = run_operation(retry_limit)
                return result, {
                    'used_reduced_rows': True,
                    'configured_max_gp_rows': requested,
                    'requested_gp_rows': effective_requested,
                    'retry_max_gp_rows': retry_limit,
                    'attempted_limits': attempted_limits,
                    'reason': 'A recoverable exact-GP fitting or resource error occurred.',
                }
            except Exception as exc:
                if not _is_recoverable_transfer_failure(exc):
                    raise
                last_error = exc
                current_limit = retry_limit

        raise last_error


def _with_recovery_metadata(workload, recovery):
    if not recovery:
        return workload
    metadata = dict(workload or {})
    metadata['requested_gp_rows'] = recovery.get(
        'configured_max_gp_rows', recovery['requested_gp_rows']
    )
    metadata['effective_requested_gp_rows'] = recovery['requested_gp_rows']
    metadata['recovery'] = recovery
    return metadata


def _transfer_tweaks_for_attempt(tweaks, row_limit, priors_config=None):
    """Adapt public transfer options without changing either numerical engine.

    The public LCB minimizes a loss L=-U. For a maximize-oriented utility U,
    argmin(μ_L-βσ) equals argmax(μ_U+βσ), the transfer engine's UCB branch.
    Leave LCB unchanged when there are no Priors: that path calls EDOS BO,
    whose own policy maps public LCB to qUCB.
    """
    attempt_tweaks = dict(tweaks or {})
    if priors_config and attempt_tweaks.get('acq_type') == 'LCB':
        attempt_tweaks['acq_type'] = 'UCB'
    requested = int(attempt_tweaks.get('max_gp_rows') or row_limit)
    if int(row_limit) != requested:
        attempt_tweaks.pop('reuse_training_row_indices', None)
    return attempt_tweaks


def _effective_transfer_rows(
    dataframe,
    features,
    objectives,
    priors,
    prior_combine,
    tweaks,
    input_values=None,
):
    """Return the authoritative initial fit size for recovery decisions."""
    effective_priors = []
    for prior in priors or []:
        configured = dict(prior)
        name = configured.get('name')
        if input_values and input_values.get(name) not in (None, ''):
            configured['target_value'] = input_values[name]
        effective_priors.append(configured)
    preview = preview_transfer_selection(
        df=dataframe,
        features_config=features,
        objectives_config=objectives,
        priors_config=effective_priors,
        prior_combine=prior_combine,
        objective_constraints=tweaks.get('objective_constraints', []),
        requested_rows=tweaks.get('max_gp_rows'),
    )
    return int(preview['rows_to_fit'])


def _validated_tweaks(raw_tweaks, features_config, objectives_config=None):
    """Copy and validate constraint-bearing request options."""
    tweaks = dict(raw_tweaks or {})
    constraints = dict(tweaks.get('constraints') or {})
    numerical = constraints.get('numerical') or []
    if features_config and all('range' in feature for feature in features_config):
        validate_feature_ranges(features_config)
    validate_feature_constraints(numerical, features_config or [])
    constraints['numerical'] = numerical
    constraints['disallowed'] = constraints.get('disallowed') or []
    tweaks['constraints'] = constraints
    tweaks['objective_constraints'] = validate_objective_constraints(
        tweaks.get('objective_constraints') or [], objectives_config or []
    )
    tweaks['allow_reduced_row_recovery'] = _as_bool(
        tweaks.get('allow_reduced_row_recovery', False)
    )
    tweaks.pop('min_prior_relevance', None)
    if tweaks.get('max_gp_rows') not in (None, ''):
        try:
            max_gp_rows = int(tweaks['max_gp_rows'])
        except (TypeError, ValueError):
            raise ValueError('Maximum GP training rows must be a whole number.')
        if max_gp_rows < 1:
            raise ValueError('Maximum GP training rows must be at least 1.')
        tweaks['max_gp_rows'] = max_gp_rows
    acquisition_type = str(tweaks.get('acq_type', 'EI')).upper()
    if acquisition_type not in {'EI', 'LCB'}:
        raise ValueError("Acquisition must be either 'EI' or 'LCB'.")
    tweaks['acq_type'] = acquisition_type
    tweaks['multi_objective_method'] = validate_multi_objective_method(
        tweaks.get('multi_objective_method', 'auto')
    )
    tweaks['data_mode'] = validate_bo_data_mode(tweaks.get('data_mode', 'auto'))
    strategy = str(tweaks.get('optimization_strategy', 'gradient_based'))
    if strategy not in {'gradient_based', 'exhaustive_grid'}:
        raise ValueError('Unknown Bayesian optimization strategy.')
    tweaks['optimization_strategy'] = strategy
    tweaks['exploration'] = validate_exploration(tweaks.get('exploration', 0.5))
    if objectives_config:
        weights = [
            objective.get('importance', 1) for objective in objectives_config
        ]
        normalized_objective_weights(weights)
        select_bo_policy(
            len(objectives_config),
            weights,
            acquisition_type,
            load_bo_budget(is_cuda=False).nehvi_max_objectives,
            requested_method=tweaks['multi_objective_method'],
        )
    return tweaks


def _validated_transfer_optimize_tweaks(raw_tweaks, features_config, objectives_config):
    """Refuse TransBO options that its fitted GP can silently ignore."""
    tweaks = _validated_tweaks(raw_tweaks, features_config, objectives_config)
    constraints = tweaks['constraints']
    if constraints['numerical'] or constraints['disallowed']:
        raise ValueError(
            'TransBO cannot enforce feature constraints throughout its fitted GP '
            'suggestion path. Remove them or use standard BO.'
        )
    if _as_bool(tweaks.get('avoid_reval', False)):
        raise ValueError(
            'TransBO cannot guarantee duplicate avoidance in its fitted GP '
            'suggestion path. Disable No Re-evaluation or use standard BO.'
        )
    tweaks['avoid_reval'] = False
    return tweaks


def _validated_priors(raw_priors):
    """Validate and normalize the type-specific Prior controls."""
    if raw_priors is None:
        return []
    if not isinstance(raw_priors, list):
        raise ValueError('Priors must be supplied as a list.')
    validated = []
    for raw_prior in raw_priors:
        if not isinstance(raw_prior, dict):
            raise ValueError('Each Prior configuration must be an object.')
        prior = dict(raw_prior)
        name = str(prior.get('name') or '').strip()
        if not name:
            raise ValueError('Each Prior configuration requires a column name.')
        prior['name'] = name
        col_type = str(prior.get('col_type') or 'categorical').lower().strip()
        if col_type not in {'categorical', 'numerical', 'smiles'}:
            raise ValueError(
                f"Unsupported Prior type for '{name}'. Choose categorical, numerical, or SMILES."
            )
        prior['col_type'] = col_type

        if col_type == 'numerical' and prior.get('tolerance') not in (None, ''):
            try:
                tolerance = float(prior['tolerance'])
            except (TypeError, ValueError):
                raise ValueError(f"Numerical tolerance for '{name}' must be a number.")
            if not math.isfinite(tolerance) or tolerance < 0:
                raise ValueError(
                    f"Numerical tolerance for '{name}' must be zero or greater."
                )
            prior['tolerance'] = tolerance
        if col_type == 'smiles':
            try:
                minimum = float(prior.get('min_similarity', 0.5))
            except (TypeError, ValueError):
                raise ValueError(
                    f"Minimum molecular similarity for '{name}' must be a number."
                )
            if not math.isfinite(minimum) or not 0.0 <= minimum <= 1.0:
                raise ValueError(
                    f"Minimum molecular similarity for '{name}' must be between 0 and 100%."
                )
            prior['min_similarity'] = minimum

        try:
            prior_weight = float(prior.get('prior_weight', 100.0))
        except (TypeError, ValueError):
            raise ValueError(f"Prior weight for '{name}' must be a number.")
        if not math.isfinite(prior_weight) or prior_weight < 0:
            raise ValueError(f"Prior weight for '{name}' must be zero or greater.")
        prior['prior_weight'] = prior_weight
        prior.pop('sigma', None)
        validated.append(prior)
    return validated


def _validated_prior_combine(value):
    method = str(value or 'multiply').lower().strip()
    if method not in {'multiply', 'average'}:
        raise ValueError('Prior combination rule must be Multiply or Average.')
    return method


def _format_sa_result(sa_output):
    (
        num_corr_matrix,
        cat_interactions,
        results,
        cat_features,
        num_features,
        extended_obj_names,
        metadata,
    ) = sa_output
    return clean_json_data({
        'numerical_correlation': num_corr_matrix,
        'cat_interactions': cat_interactions,
        'results': results,
        'cat_features': cat_features,
        'num_features': num_features,
        'extended_obj_names': extended_obj_names,
        'metadata': metadata,
    })


def _format_bo_result(result):
    """Preserve the public BO response while adding bounded-run diagnostics."""

    serialized = result.to_dict() if hasattr(result, "to_dict") else dict(result)
    response = {
        "performance": serialized.get("performance") or {},
        "constraint_status": serialized.get("constraint_status") or {},
        "diagnostics": serialized.get("diagnostics") or {},
    }
    if serialized.get("suggestions") is None:
        response["exhausted"] = True
    else:
        response["suggestions"] = serialized.get("suggestions")
    return clean_json_data(response)


def _run_bo_job(job_id, payload):
    """Supervise one process-isolated BO calculation from a lightweight thread."""

    cancel_event = get_cancel_event(job_id)
    if cancel_event is None:
        return

    def progress_callback(phase, percent, details):
        if cancel_event.is_set():
            _update_sa_job(
                job_id,
                status="cancelling",
                phase="Stopping optimization",
                percent=percent,
                details=details,
            )
            return
        updates = {
            "status": "running",
            "phase": phase,
            "percent": percent,
            "details": details,
        }
        if details.get("algorithm"):
            updates["algorithm"] = details["algorithm"]
        _update_sa_job(job_id, **updates)

    _update_sa_job(
        job_id,
        status="running",
        phase="Starting isolated BO worker",
        percent=1,
        started_at=time.time(),
    )
    try:
        result = run_bo_in_subprocess(payload, cancel_event, progress_callback)
        diagnostics = result.get("diagnostics") or {}
        _update_sa_job(
            job_id,
            status="completed",
            phase="Completed",
            percent=100,
            algorithm=diagnostics.get("algorithm"),
            details={
                "elapsed_seconds": diagnostics.get("elapsed_seconds"),
                "algorithm": diagnostics.get("algorithm"),
                "algorithm_label": diagnostics.get("algorithm_label"),
            },
            result=_format_bo_result(result),
            finished_at=time.time(),
        )
    except BOProcessCancelled as exc:
        _update_sa_job(
            job_id,
            status="cancelled",
            phase="Cancelled",
            error=str(exc),
            error_code="cancelled",
            finished_at=time.time(),
        )
    except BOProcessTimeout as exc:
        _update_sa_job(
            job_id,
            status="failed",
            phase="Stopped at hard time limit",
            error=str(exc),
            error_code="timeout",
            finished_at=time.time(),
        )
    except BOChildFailure as exc:
        _update_sa_job(
            job_id,
            status="failed",
            phase="Optimization failed",
            error=str(exc),
            error_code=exc.error_code,
            finished_at=time.time(),
        )
    except Exception:
        import traceback

        traceback.print_exc()
        _update_sa_job(
            job_id,
            status="failed",
            phase="Optimization failed",
            error="The BO supervisor failed unexpectedly. Please retry.",
            error_code="supervisor_error",
            finished_at=time.time(),
        )


def _run_sa_job(job_id, payload):
    cancel_event = get_cancel_event(job_id)
    if cancel_event is None:
        return

    def progress_callback(phase, percent):
        _update_sa_job(job_id, phase=phase, percent=percent, updated_at=time.time())

    _update_sa_job(
        job_id,
        status='running',
        phase='Preparing analysis',
        percent=1,
        started_at=time.time(),
    )
    try:
        dataframe = pd.DataFrame(payload['data'], columns=payload['columns'])
        output = run_sa(
            df=dataframe,
            features_config=payload['features'],
            objectives_config=payload['objectives'],
            tweaks=payload['tweaks'],
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )
        _update_sa_job(
            job_id,
            status='completed',
            phase='Completed',
            percent=100,
            result=_format_sa_result(output),
            finished_at=time.time(),
        )
    except AnalysisCancelled as exc:
        _update_sa_job(
            job_id,
            status='cancelled',
            phase='Cancelled',
            error=str(exc),
            finished_at=time.time(),
        )
    except Exception as exc:
        import traceback
        traceback.print_exc()
        _update_sa_job(
            job_id,
            status='failed',
            phase='Failed',
            error=str(exc),
            finished_at=time.time(),
        )


def _run_transfer_optimize_job(job_id, payload):
    cancel_event = get_cancel_event(job_id)
    if cancel_event is None:
        return
    _update_sa_job(job_id, status='running', phase='Computing prior weights', percent=15)
    try:
        dataframe = pd.DataFrame(payload['data'], columns=payload['columns'])
        _update_sa_job(job_id, phase='Fitting transfer surrogate models', percent=45)
        operation_result, recovery = _run_transfer_with_row_recovery(
            lambda row_limit: run_transfer_bo_optimization(
                df=dataframe,
                features_config=payload['features'],
                objectives_config=payload['objectives'],
                priors_config=payload['priors'],
                tweaks=_transfer_tweaks_for_attempt(payload['tweaks'], row_limit, payload['priors']),
                device=DEVICE,
                compute_device=COMPUTE_DEVICE,
                prior_combine=payload['prior_combine'],
                max_gp_rows=row_limit,
                cancel_event=cancel_event,
                deadline_monotonic=(
                    time.monotonic() + _transfer_attempt_timeout_seconds()
                ),
            ),
            payload['tweaks'].get('max_gp_rows'),
            payload['tweaks'].get('allow_reduced_row_recovery', False),
            job_id,
            cancel_event,
            resolve_effective_rows=lambda: _effective_transfer_rows(
                dataframe,
                payload['features'],
                payload['objectives'],
                payload['priors'],
                payload['prior_combine'],
                payload['tweaks'],
            ),
        )
        suggestions, performance, transfer, constraint_status = operation_result
        transfer = _with_recovery_metadata(transfer, recovery)
        if cancel_event.is_set():
            raise AnalysisCancelled('Transfer optimization cancelled by user.')
        result = {
            'performance': performance,
            'transfer': transfer,
            'constraint_status': constraint_status,
        }
        if suggestions is None:
            result['exhausted'] = True
        else:
            result['suggestions'] = suggestions
        _update_sa_job(
            job_id, status='completed', phase='Completed', percent=100,
            result=clean_json_data(result), finished_at=time.time(),
        )
    except (AnalysisCancelled, TransferCalculationCancelled) as exc:
        _update_sa_job(
            job_id, status='cancelled', phase='Cancelled', error=str(exc),
            finished_at=time.time(),
        )
    except Exception as exc:
        import traceback
        traceback.print_exc()
        _update_sa_job(
            job_id, status='failed', phase='Failed', error=str(exc),
            finished_at=time.time(),
        )


def _run_transfer_estimate_job(job_id, payload):
    cancel_event = get_cancel_event(job_id)
    if cancel_event is None:
        return
    _update_sa_job(job_id, status='running', phase='Fitting transfer surrogate', percent=35)
    try:
        dataframe = pd.DataFrame(payload['data'], columns=payload['columns'])
        operation_result, recovery = _run_transfer_with_row_recovery(
            lambda row_limit: run_transfer_bo_estimation(
                df=dataframe,
                input_values=payload['inputs'],
                features_config=payload['features'],
                objectives_config=payload['objectives'],
                priors_config=payload['priors'],
                tweaks=_transfer_tweaks_for_attempt(payload['tweaks'], row_limit, payload['priors']),
                device=DEVICE,
                compute_device=COMPUTE_DEVICE,
                prior_combine=payload['prior_combine'],
                max_gp_rows=row_limit,
                cancel_event=cancel_event,
                deadline_monotonic=(
                    time.monotonic() + _transfer_attempt_timeout_seconds()
                ),
            ),
            payload['tweaks'].get('max_gp_rows'),
            payload['tweaks'].get('allow_reduced_row_recovery', False),
            job_id,
            cancel_event,
            resolve_effective_rows=lambda: _effective_transfer_rows(
                dataframe,
                payload['features'],
                payload['objectives'],
                payload['priors'],
                payload['prior_combine'],
                payload['tweaks'],
                input_values=payload['inputs'],
            ),
        )
        results, global_score, violations, workload = operation_result
        workload = _with_recovery_metadata(workload, recovery)
        if cancel_event.is_set():
            raise AnalysisCancelled('Transfer estimation cancelled by user.')
        result = clean_json_data({
            'results': results,
            'global_success': global_score,
            'constraint_violated': bool(violations),
            'violations': violations,
            'workload': workload,
        })
        _update_sa_job(
            job_id, status='completed', phase='Completed', percent=100,
            result=result, finished_at=time.time(),
        )
    except (AnalysisCancelled, TransferCalculationCancelled) as exc:
        _update_sa_job(job_id, status='cancelled', phase='Cancelled', error=str(exc), finished_at=time.time())
    except Exception as exc:
        import traceback
        traceback.print_exc()
        _update_sa_job(job_id, status='failed', phase='Failed', error=str(exc), finished_at=time.time())


def _run_bo_surface_job(job_id, payload):
    cancel_event = get_cancel_event(job_id)
    if cancel_event is None:
        return
    _update_sa_job(job_id, status='running', phase='Preparing response surface', percent=5)
    try:
        dataframe = pd.DataFrame(payload['data'], columns=payload['columns'])
        if dataframe.empty:
            raise ValueError('Cannot generate surface with no data points.')
        result = run_bo_surface(
            df=dataframe,
            features_config=payload['features'],
            objectives_config=payload['objectives'],
            tweaks=payload['tweaks'],
            x_feat=payload.get('x_feat'),
            y_feat=payload.get('y_feat'),
            z_axis=payload.get('z_axis'),
            color_obj=payload.get('color_obj'),
            fixed_values=payload.get('fixed_values', {}),
            resolution=payload.get('resolution', 25),
            device=DEVICE,
            compute_device=COMPUTE_DEVICE,
            cancel_event=cancel_event,
            progress_callback=lambda phase, percent: _update_sa_job(
                job_id, status='running', phase=phase, percent=percent
            ),
        )
        if cancel_event.is_set():
            raise SurfaceCalculationCancelled('Response-surface calculation cancelled by user.')
        _update_sa_job(
            job_id, status='completed', phase='Completed', percent=100,
            result=clean_json_data(result), finished_at=time.time(),
        )
    except SurfaceCalculationCancelled as exc:
        _update_sa_job(
            job_id, status='cancelled', phase='Cancelled', error=str(exc),
            finished_at=time.time(),
        )
    except Exception as exc:
        import traceback
        traceback.print_exc()
        _update_sa_job(
            job_id, status='failed', phase='Failed', error=str(exc),
            finished_at=time.time(),
        )


def _run_transfer_surface_job(job_id, payload):
    cancel_event = get_cancel_event(job_id)
    if cancel_event is None:
        return
    _update_sa_job(job_id, status='running', phase='Preparing transfer response surface', percent=5)
    try:
        dataframe = pd.DataFrame(payload['data'], columns=payload['columns'])
        if dataframe.empty:
            raise ValueError('Cannot generate surface with no data points.')
        result = run_transfer_bo_surface(
            df=dataframe,
            features_config=payload['features'],
            objectives_config=payload['objectives'],
            priors_config=payload.get('priors', []),
            tweaks=payload['tweaks'],
            x_feat=payload.get('x_feat'),
            y_feat=payload.get('y_feat'),
            z_axis=payload.get('z_axis'),
            color_obj=payload.get('color_obj'),
            fixed_values=payload.get('fixed_values', {}),
            resolution=payload.get('resolution', 25),
            device=DEVICE,
            compute_device=COMPUTE_DEVICE,
            prior_combine=payload.get('prior_combine', 'multiply'),
            max_gp_rows=payload['tweaks'].get('max_gp_rows'),
            cancel_event=cancel_event,
            progress_callback=lambda phase, percent: _update_sa_job(
                job_id, status='running', phase=phase, percent=percent
            ),
        )
        if cancel_event.is_set():
            raise SurfaceCalculationCancelled('Response-surface calculation cancelled by user.')
        _update_sa_job(
            job_id, status='completed', phase='Completed', percent=100,
            result=clean_json_data(result), finished_at=time.time(),
        )
    except SurfaceCalculationCancelled as exc:
        _update_sa_job(
            job_id, status='cancelled', phase='Cancelled', error=str(exc),
            finished_at=time.time(),
        )
    except Exception as exc:
        import traceback
        traceback.print_exc()
        _update_sa_job(
            job_id, status='failed', phase='Failed', error=str(exc),
            finished_at=time.time(),
        )


def _run_sa_estimate_job(job_id, payload):
    cancel_event = get_cancel_event(job_id)
    if cancel_event is None:
        return
    _update_sa_job(job_id, status='running', phase='Fitting estimator model', percent=35)
    try:
        dataframe = pd.DataFrame(payload['data'], columns=payload['columns'])
        predictions, std_errors, scores, global_score, violations = estimate_sa(
            df=dataframe,
            input_values=payload['inputs'],
            features_config=payload['features'],
            objectives_config=payload['objectives'],
            model_name=payload['model'],
            params=payload['params'],
            tweaks=payload['tweaks'],
        )
        if cancel_event.is_set():
            raise AnalysisCancelled('SA estimation cancelled by user.')
        result = clean_json_data({
            'predictions': predictions,
            'std_errors': std_errors,
            'success_scores': scores,
            'global_success': global_score,
            'constraint_violated': bool(violations),
            'violations': violations,
        })
        _update_sa_job(
            job_id, status='completed', phase='Completed', percent=100,
            result=result, finished_at=time.time(),
        )
    except AnalysisCancelled as exc:
        _update_sa_job(job_id, status='cancelled', phase='Cancelled', error=str(exc), finished_at=time.time())
    except Exception as exc:
        import traceback
        traceback.print_exc()
        _update_sa_job(job_id, status='failed', phase='Failed', error=str(exc), finished_at=time.time())


@api_bp.app_errorhandler(413)
def request_entity_too_large(e):
    """Return JSON (not HTML) for 413 Payload Too Large so the frontend can parse it."""
    return jsonify({'error': 'Payload too large. The dataset exceeds the maximum allowed size.'}), 413




def clean_json_data(data):
    """Recursively cleans JSON data converting NaNs, Infs, and numpy types to JSON-safe values."""
    if isinstance(data, dict):
        return {k: clean_json_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_json_data(v) for v in data]
    elif isinstance(data, float):
        if math.isnan(data) or math.isinf(data):
            return None
        return data
    elif isinstance(data, np.floating):
        if np.isnan(data) or np.isinf(data):
            return None
        return float(data)
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.ndarray):
        return clean_json_data(data.tolist())
    else:
        return data


def _shutdown_is_allowed():
    """Return True only for a local request in a non-hosted environment."""
    client_ip = request.remote_addr
    is_local = client_ip in ('127.0.0.1', '::1')
    return is_local and not _is_hosted_environment()


@api_bp.route('/')
def index():
    """Serves the main single-page application (SPA)."""
    return render_template(
        'index.html',
        show_shutdown=_shutdown_is_allowed(),
        hosted_environment=_is_hosted_environment(),
    )


@api_bp.route('/guide')
def guide():
    """Serves the user guide and documentation."""
    return render_template('guide.html')


@api_bp.route('/shutdown', methods=['POST'])
def shutdown():
    """Gracefully shuts down the local Flask server (local development only)."""
    if not _shutdown_is_allowed():
        return jsonify({'error': 'Shutdown endpoint is restricted to local development.'}), 403

    def _stop():
        import signal
        time.sleep(0.5)
        os.kill(os.getpid(), signal.SIGTERM)

    threading.Thread(target=_stop, daemon=True).start()
    return jsonify({'status': 'shutting down'})


@api_bp.route('/upload', methods=['POST'])
def upload():
    """Accepts CSV file upload and returns parsed JSON columns and data."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        df = pd.read_csv(file)
        data = df.to_dict(orient='split')
        return jsonify(clean_json_data(data))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api_bp.route('/setup_csv/export', methods=['POST'])
def export_setup_csv():
    """Download the current Setup, separate from experiment data."""
    try:
        payload = request.get_json(silent=True) or {}
        module = payload.get('module')
        csv_text = serialize_setup_csv(payload.get('setup') or {}, module)
        label = {'doe': 'DoE', 'tbo': 'TransBO'}.get(module, str(module).upper())
        return Response(csv_text, mimetype='text/csv', headers={
            'Content-Disposition': f'attachment; filename="CEDOS_{label}_setup.csv"',
        })
    except (TypeError, ValueError) as exc:
        return jsonify({'error': str(exc)}), 400


@api_bp.route('/setup_csv/import', methods=['POST'])
def import_setup_csv():
    """Parse a setup file without persisting data or changing experiment rows."""
    try:
        file = request.files.get('file')
        if file is None or not file.filename:
            raise ValueError('Select a setup CSV file.')
        raw = file.read(2_000_001)
        if len(raw) > 2_000_000:
            raise ValueError('The setup CSV must be smaller than 2 MB.')
        try:
            content = raw.decode('utf-8-sig')
        except UnicodeDecodeError as exc:
            raise ValueError('The setup CSV must use UTF-8 text.') from exc
        columns = json.loads(request.form.get('columns', '[]'))
        if not isinstance(columns, list) or any(not isinstance(col, str) for col in columns):
            raise ValueError('Dataset column names are invalid.')
        return jsonify(parse_setup_csv(content, request.form.get('module'), columns))
    except (TypeError, ValueError) as exc:
        return jsonify({'error': str(exc)}), 400


@api_bp.route('/detect_prior_types', methods=['POST'])
def detect_prior_types():
    """
    Lightweight endpoint called when a column is assigned the Prior role.
    Detects whether the column is 'smiles', 'numerical', or 'categorical'.
    """
    try:
        req_data = request.get_json()
        data = req_data.get('data', [])
        columns = req_data.get('columns', [])
        prior_cols = req_data.get('prior_cols', [])

        df = pd.DataFrame(data, columns=columns)
        detected = {}

        for col in prior_cols:
            if col in df.columns:
                detected[col] = detect_prior_column_type(df[col].values)
            else:
                detected[col] = "categorical"

        return jsonify({
            "types": detected,
            "rdkit_available": True
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@api_bp.route('/transfer_optimize_submit', methods=['POST'])
def transfer_optimize_submit():
    """Queue Transfer BO so reverse-proxy request limits cannot terminate it."""
    try:
        req_data = request.get_json() or {}
        features = req_data.get('features', [])
        objectives = req_data.get('objectives', [])
        data = req_data.get('data')
        columns = req_data.get('columns')
        if not isinstance(data, list) or not isinstance(columns, list):
            return jsonify({'error': 'A valid dataset and columns are required.'}), 400
        if not features or not objectives:
            return jsonify({'error': 'At least one feature and objective are required.'}), 400
        tweaks = _validated_transfer_optimize_tweaks(
            req_data.get('tweaks', {}), features, objectives
        )
        priors = _validated_priors(req_data.get('priors', []))
        prior_combine = _validated_prior_combine(req_data.get('prior_combine'))
        payload = {
            'data': data,
            'columns': columns,
            'features': features,
            'objectives': objectives,
            'priors': priors,
            'prior_combine': prior_combine,
            'tweaks': tweaks,
        }
        job_id = _queue_idempotent_background_job(
            'transfer-optimization', payload, _run_transfer_optimize_job,
            'Queued transfer optimization',
        )
        if job_id is None:
            return jsonify({'error': 'Another analysis job is already running. Wait for it to finish or cancel it first.'}), 429
        _start_transfer_job_watchdog(job_id)
        return jsonify({'job_id': job_id, 'status': 'queued'}), 202
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400


@api_bp.route('/transfer_selection_preview', methods=['POST'])
def transfer_selection_preview():
    """Return the authoritative GP capacity and prior-relevance row counts."""
    try:
        req_data = request.get_json() or {}
        data = req_data.get('data')
        columns = req_data.get('columns')
        features = req_data.get('features', [])
        objectives = req_data.get('objectives', [])
        if not isinstance(data, list) or not isinstance(columns, list):
            return jsonify({'error': 'A valid dataset and columns are required.'}), 400
        if not features or not objectives:
            return jsonify({'error': 'At least one feature and objective are required.'}), 400
        tweaks = _validated_tweaks(req_data.get('tweaks', {}), features, objectives)
        priors = _validated_priors(req_data.get('priors', []))
        prior_combine = _validated_prior_combine(req_data.get('prior_combine'))
        dataframe = pd.DataFrame(data, columns=columns)
        preview = preview_transfer_selection(
            df=dataframe,
            features_config=features,
            objectives_config=objectives,
            priors_config=priors,
            prior_combine=prior_combine,
            objective_constraints=tweaks.get('objective_constraints', []),
            requested_rows=tweaks.get('max_gp_rows'),
        )
        return jsonify(clean_json_data(preview))
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(exc)}), 500


@api_bp.route('/transfer_optimize', methods=['POST'])
def transfer_optimize():
    """
    Transfer Learning Bayesian Optimization endpoint.
    Weights historical observations based on similarity to specified target priors.
    """
    try:
        req_data = request.get_json()
        data = req_data.get('data')
        columns = req_data.get('columns')
        features_config = req_data.get('features', [])
        objectives_config = req_data.get('objectives', [])
        priors_config = _validated_priors(req_data.get('priors', []))
        prior_combine = _validated_prior_combine(req_data.get('prior_combine'))
        tweaks = _validated_transfer_optimize_tweaks(
            req_data.get('tweaks', {}), features_config, objectives_config
        )

        df = pd.DataFrame(data, columns=columns)

        operation_result, recovery = _run_transfer_with_row_recovery(
            lambda row_limit: run_transfer_bo_optimization(
                df=df,
                features_config=features_config,
                objectives_config=objectives_config,
                priors_config=priors_config,
                tweaks=_transfer_tweaks_for_attempt(tweaks, row_limit, priors_config),
                device=DEVICE,
                compute_device=COMPUTE_DEVICE,
                prior_combine=prior_combine,
                max_gp_rows=row_limit,
            ),
            tweaks.get('max_gp_rows'),
            tweaks.get('allow_reduced_row_recovery', False),
            resolve_effective_rows=lambda: _effective_transfer_rows(
                df,
                features_config,
                objectives_config,
                priors_config,
                prior_combine,
                tweaks,
            ),
        )
        suggestions, performance_results, transfer_diag, constraint_status = operation_result
        transfer_diag = _with_recovery_metadata(transfer_diag, recovery)

        if suggestions is None:
            return jsonify(clean_json_data({
                'exhausted': True,
                'performance': performance_results,
                'transfer': transfer_diag,
                'constraint_status': constraint_status
            }))

        return jsonify(clean_json_data({
            'suggestions': suggestions,
            'performance': performance_results,
            'transfer': transfer_diag,
            'constraint_status': constraint_status
        }))

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@api_bp.route('/transfer_estimate_submit', methods=['POST'])
def transfer_estimate_submit():
    """Queue a potentially expensive transfer-surrogate point estimate."""
    try:
        req_data = request.get_json() or {}
        features = req_data.get('features', [])
        objectives = req_data.get('objectives', [])
        data = req_data.get('data')
        columns = req_data.get('columns')
        if not isinstance(data, list) or not isinstance(columns, list):
            return jsonify({'error': 'A valid dataset and columns are required.'}), 400
        tweaks = _validated_tweaks(req_data.get('tweaks', {}), features, objectives)
        priors = _validated_priors(req_data.get('priors', []))
        prior_combine = _validated_prior_combine(req_data.get('prior_combine'))
        payload = {
            'data': data,
            'columns': columns,
            'inputs': req_data.get('inputs', {}),
            'features': features,
            'objectives': objectives,
            'priors': priors,
            'prior_combine': prior_combine,
            'tweaks': tweaks,
        }
        job_id = _queue_idempotent_background_job(
            'transfer-estimation', payload, _run_transfer_estimate_job,
            'Queued transfer estimation',
        )
        if job_id is None:
            return jsonify({'error': 'Another analysis job is already running. Wait for it to finish or cancel it first.'}), 429
        _start_transfer_job_watchdog(job_id)
        return jsonify({'job_id': job_id, 'status': 'queued'}), 202
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400


@api_bp.route('/transfer_estimate', methods=['POST'])
def transfer_estimate():
    """Predicts objective values using the transfer-weighted BO GP model."""
    try:
        req_data = request.get_json()
        data = req_data.get('data')
        columns = req_data.get('columns')
        input_values = req_data.get('inputs', {})
        features_config = req_data.get('features', [])
        objectives_config = req_data.get('objectives', [])
        priors_config = _validated_priors(req_data.get('priors', []))
        prior_combine = _validated_prior_combine(req_data.get('prior_combine'))
        tweaks = _validated_tweaks(
            req_data.get('tweaks', {}), features_config, objectives_config
        )

        df = pd.DataFrame(data, columns=columns)
        if df.empty:
            return jsonify({'error': 'Cannot estimate with no data points.'}), 400

        operation_result, recovery = _run_transfer_with_row_recovery(
            lambda row_limit: run_transfer_bo_estimation(
                df=df,
                input_values=input_values,
                features_config=features_config,
                objectives_config=objectives_config,
                priors_config=priors_config,
                tweaks=_transfer_tweaks_for_attempt(tweaks, row_limit, priors_config),
                device=DEVICE,
                compute_device=COMPUTE_DEVICE,
                prior_combine=prior_combine,
                max_gp_rows=row_limit,
            ),
            tweaks.get('max_gp_rows'),
            tweaks.get('allow_reduced_row_recovery', False),
            resolve_effective_rows=lambda: _effective_transfer_rows(
                df,
                features_config,
                objectives_config,
                priors_config,
                prior_combine,
                tweaks,
                input_values=input_values,
            ),
        )
        results, global_score, violations, workload = operation_result
        workload = _with_recovery_metadata(workload, recovery)

        return jsonify(clean_json_data({
            'results': results,
            'global_success': global_score,
            'constraint_violated': bool(violations),
            'violations': violations,
            'workload': workload,
        }))

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@api_bp.route('/transfer_surface', methods=['POST'])
def transfer_surface():
    """Computes a response surface grid using the transfer-weighted GP model."""
    try:
        req_data = request.get_json()
        data = req_data.get('data')
        columns = req_data.get('columns')
        features_config = req_data.get('features', [])
        objectives_config = req_data.get('objectives', [])
        priors_config = _validated_priors(req_data.get('priors', []))
        _validate_transfer_surface_prerequisites(priors_config)
        prior_combine = _validated_prior_combine(req_data.get('prior_combine'))
        tweaks = _validated_tweaks(
            req_data.get('tweaks', {}), features_config, objectives_config
        )
        x_feat = req_data.get('x_feat')
        y_feat = req_data.get('y_feat')
        z_axis = req_data.get('z_axis')
        color_obj = req_data.get('color_obj')
        fixed_values = req_data.get('fixed_values', {})
        resolution = req_data.get('resolution', 25)

        df = pd.DataFrame(data, columns=columns)
        if df.empty:
            return jsonify({'error': 'Cannot generate surface with no data points.'}), 400

        result = run_transfer_bo_surface(
            df=df,
            features_config=features_config,
            objectives_config=objectives_config,
            priors_config=priors_config,
            tweaks=tweaks,
            x_feat=x_feat,
            y_feat=y_feat,
            z_axis=z_axis,
            color_obj=color_obj,
            fixed_values=fixed_values,
            resolution=resolution,
            device=DEVICE,
            compute_device=COMPUTE_DEVICE,
            prior_combine=prior_combine,
            max_gp_rows=tweaks.get('max_gp_rows')
        )

        return jsonify(clean_json_data(result))

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@api_bp.route('/transfer_surface_submit', methods=['POST'])
def transfer_surface_submit():
    """Queue a cancellable TransBO response-surface calculation."""
    try:
        req_data = request.get_json() or {}
        data = req_data.get('data')
        columns = req_data.get('columns')
        features = req_data.get('features', [])
        objectives = req_data.get('objectives', [])
        if not isinstance(data, list) or not isinstance(columns, list):
            return jsonify({'error': 'A valid dataset and columns are required.'}), 400
        if len(features) < 2 or not objectives:
            return jsonify({'error': 'At least two features and one objective are required.'}), 400
        priors = _validated_priors(req_data.get('priors', []))
        _validate_transfer_surface_prerequisites(priors)
        tweaks = _validated_tweaks(req_data.get('tweaks', {}), features, objectives)
        prior_combine = _validated_prior_combine(req_data.get('prior_combine'))
        payload = {
            'data': data,
            'columns': columns,
            'features': features,
            'objectives': objectives,
            'priors': priors,
            'prior_combine': prior_combine,
            'x_feat': req_data.get('x_feat'),
            'y_feat': req_data.get('y_feat'),
            'z_axis': req_data.get('z_axis'),
            'color_obj': req_data.get('color_obj'),
            'fixed_values': req_data.get('fixed_values', {}),
            'resolution': req_data.get('resolution', 25),
            'tweaks': tweaks,
        }
        job_id = _queue_idempotent_background_job(
            'transfer-surface', payload, _run_transfer_surface_job,
            'Queued transfer response surface',
        )
        if job_id is None:
            return jsonify({'error': 'Another analysis job is already running. Wait for it to finish or cancel it first.'}), 429
        return jsonify({'job_id': job_id, 'status': 'queued'}), 202
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(exc)}), 500


# -------------------------------------------------------------
# Preserved EDOS Endpoints: Standard BO, DoE, and SA
# -------------------------------------------------------------

@api_bp.route('/optimize', methods=['POST'])
def optimize():
    """Backward-compatible synchronous endpoint for small local/API jobs."""
    try:
        req_data = request.get_json()
        data = req_data.get('data')
        columns = req_data.get('columns')
        features_config = req_data.get('features')
        objectives_config = req_data.get('objectives')
        tweaks = _validated_tweaks(
            req_data.get('tweaks', {}), features_config, objectives_config
        )
        tweaks['_allow_unsafe_full'] = not _is_hosted_environment()

        raw_batch_size = float(tweaks.get('batch_size', 1))
        if not raw_batch_size.is_integer():
            raise ValueError('Batch size must be a whole number.')
        batch_size = int(raw_batch_size)
        sync_max_objectives = max(1, int(os.environ.get('BO_SYNC_MAX_OBJECTIVES', '2')))
        sync_max_batch = max(1, int(os.environ.get('BO_SYNC_MAX_BATCH_SIZE', '3')))
        if _is_hosted_environment() and (
            len(objectives_config or []) > sync_max_objectives
            or batch_size > sync_max_batch
        ):
            return jsonify({
                'error': (
                    'This workload must use the background BO endpoint so it is '
                    'observable and is not interrupted by the web gateway.'
                ),
                'error_code': 'async_required',
            }), 422

        df = pd.DataFrame(data, columns=columns)
        result = run_bo_optimization(
            df=df,
            features_config=features_config,
            objectives_config=objectives_config,
            tweaks=tweaks,
            device=DEVICE,
            compute_device=COMPUTE_DEVICE,
            compute_performance=bool(req_data.get('compute_performance', False)),
        )
        return jsonify(_format_bo_result(result))

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': 'Bayesian optimization failed unexpectedly. Review the server log.',
            'error_code': 'optimization_error',
        }), 500


@api_bp.route('/bo_submit', methods=['POST'])
def submit_bo_job():
    """Validate and queue a process-isolated Bayesian optimization job."""

    try:
        req_data = request.get_json() or {}
        data = req_data.get('data')
        columns = req_data.get('columns')
        features = req_data.get('features', [])
        objectives = req_data.get('objectives', [])
        if not isinstance(data, list) or not isinstance(columns, list):
            return jsonify({'error': 'A valid dataset and columns are required.'}), 400
        if any(not isinstance(row, (list, tuple)) or len(row) != len(columns) for row in data):
            return jsonify({'error': 'Every dataset row must match the columns.'}), 400
        if not features or not objectives:
            return jsonify({'error': 'At least one feature and objective are required.'}), 400
        tweaks = _validated_tweaks(req_data.get('tweaks', {}), features, objectives)
        # This is server-owned. A request cannot opt a hosted worker out of the
        # exact-GP safety boundary.
        tweaks['_allow_unsafe_full'] = not _is_hosted_environment()
        # Parse here so malformed and out-of-range requests fail before reserving a slot.
        raw_batch_size = float(tweaks.get('batch_size', 1))
        if not raw_batch_size.is_integer():
            raise ValueError('Batch size must be a whole number.')
        batch_size = int(raw_batch_size)
        max_batch_size = max(1, int(os.environ.get('BO_MAX_BATCH_SIZE', '20')))
        if not 1 <= batch_size <= max_batch_size:
            raise ValueError(
                f'Batch size must be between 1 and {max_batch_size}.'
            )
        try:
            random_seed = int(req_data.get('random_seed', 1729))
        except (TypeError, ValueError) as exc:
            raise ValueError('The BO random seed must be an integer.') from exc
        if not 0 <= random_seed < 2**63:
            raise ValueError('The BO random seed must be between 0 and 2^63 - 1.')
        payload = {
            'data': data,
            'columns': columns,
            'features': features,
            'objectives': objectives,
            'tweaks': tweaks,
            'compute_performance': bool(req_data.get('compute_performance', False)),
            'random_seed': random_seed,
        }
        job_id = _queue_idempotent_background_job(
            'bo-optimization', payload, _run_bo_job, 'Queued Bayesian optimization'
        )
        if job_id is None:
            return jsonify({
                'error': (
                    'Another analysis is already running. Wait for it to finish '
                    'or cancel it before starting BO.'
                ),
                'error_code': 'busy',
            }), 429
        return jsonify({'job_id': job_id, 'status': 'queued'}), 202
    except ValueError as exc:
        return jsonify({'error': str(exc), 'error_code': 'validation'}), 400
    except Exception:
        import traceback

        traceback.print_exc()
        return jsonify({
            'error': 'Unable to queue Bayesian optimization.',
            'error_code': 'submission_error',
        }), 500


@api_bp.route('/doe', methods=['POST'])
def run_doe():
    """Generate a validated Design of Experiments (DoE) table."""
    try:
        req_data = request.get_json()
        features_config = req_data.get('features', [])
        tweaks = _validated_tweaks(req_data.get('tweaks', {}), features_config)

        suggested_table, metrics = generate_doe(
            features_config=features_config,
            tweaks=tweaks
        )

        return jsonify(clean_json_data({'suggestions': suggested_table, 'metrics': metrics}))

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@api_bp.route('/estimate_bo', methods=['POST'])
def estimate_bo():
    """Standard BO GP point estimation endpoint (unchanged from EDOS)."""
    try:
        req_data = request.get_json()
        data = req_data.get('data')
        columns = req_data.get('columns')
        input_values = req_data.get('inputs')
        features_config = req_data.get('features', [])
        objectives_config = req_data.get('objectives', [])
        tweaks = _validated_tweaks(
            req_data.get('tweaks', {}), features_config, objectives_config
        )

        df = pd.DataFrame(data, columns=columns)
        if df.empty:
            return jsonify({'error': 'Cannot estimate with no data points.'}), 400

        raw_predictions, raw_std_errors, success_scores, global_score, violations = run_bo_estimation(
            df=df,
            input_values=input_values,
            features_config=features_config,
            objectives_config=objectives_config,
            tweaks=tweaks,
            device=DEVICE,
            compute_device=COMPUTE_DEVICE
        )

        return jsonify(clean_json_data({
            'predictions': raw_predictions,
            'std_errors': raw_std_errors,
            'success_scores': success_scores,
            'global_success': global_score,
            'constraint_violated': bool(violations),
            'violations': violations
        }))

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@api_bp.route('/bo_surface', methods=['POST'])
def bo_surface():
    """Standard BO response surface endpoint (unchanged from EDOS)."""
    try:
        req_data = request.get_json()
        data = req_data.get('data')
        _validate_bo_surface_row_count(data)
        columns = req_data.get('columns')
        features_config = req_data.get('features', [])
        objectives_config = req_data.get('objectives', [])
        tweaks = _validated_tweaks(
            req_data.get('tweaks', {}), features_config, objectives_config
        )
        x_feat = req_data.get('x_feat')
        y_feat = req_data.get('y_feat')
        z_axis = req_data.get('z_axis')
        color_obj = req_data.get('color_obj')
        fixed_values = req_data.get('fixed_values', {})
        resolution = req_data.get('resolution', 25)

        df = pd.DataFrame(data, columns=columns)
        if df.empty:
            return jsonify({'error': 'Cannot generate surface with no data points.'}), 400

        result = run_bo_surface(
            df=df,
            features_config=features_config,
            objectives_config=objectives_config,
            tweaks=tweaks,
            x_feat=x_feat,
            y_feat=y_feat,
            z_axis=z_axis,
            color_obj=color_obj,
            fixed_values=fixed_values,
            resolution=resolution,
            device=DEVICE,
            compute_device=COMPUTE_DEVICE
        )

        return jsonify(clean_json_data(result))

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@api_bp.route('/bo_surface_submit', methods=['POST'])
def bo_surface_submit():
    """Queue a cancellable BO response-surface calculation."""
    try:
        req_data = request.get_json() or {}
        data = req_data.get('data')
        columns = req_data.get('columns')
        features = req_data.get('features', [])
        objectives = req_data.get('objectives', [])
        if not isinstance(data, list) or not isinstance(columns, list):
            return jsonify({'error': 'A valid dataset and columns are required.'}), 400
        _validate_bo_surface_row_count(data)
        if len(features) < 2 or not objectives:
            return jsonify({'error': 'At least two features and one objective are required.'}), 400
        tweaks = _validated_tweaks(req_data.get('tweaks', {}), features, objectives)
        payload = {
            'data': data,
            'columns': columns,
            'features': features,
            'objectives': objectives,
            'x_feat': req_data.get('x_feat'),
            'y_feat': req_data.get('y_feat'),
            'z_axis': req_data.get('z_axis'),
            'color_obj': req_data.get('color_obj'),
            'fixed_values': req_data.get('fixed_values', {}),
            'resolution': req_data.get('resolution', 25),
            'tweaks': tweaks,
        }
        job_id = _queue_idempotent_background_job(
            'bo-surface', payload, _run_bo_surface_job,
            'Queued response surface',
        )
        if job_id is None:
            return jsonify({'error': 'Another analysis job is already running. Wait for it to finish or cancel it first.'}), 429
        return jsonify({'job_id': job_id, 'status': 'queued'}), 202
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(exc)}), 500


@api_bp.route('/sa_submit', methods=['POST'])
def submit_sa_job():
    """Queue a long-running SA calculation and return immediately."""
    try:
        _cleanup_sa_jobs()
        req_data = request.get_json() or {}
        features_config = req_data.get('features', [])
        objectives_config = req_data.get('objectives', [])
        data = req_data.get('data')
        columns = req_data.get('columns')
        if not isinstance(data, list) or not isinstance(columns, list):
            return jsonify({'error': 'A valid dataset and columns are required.'}), 400
        if not features_config or not objectives_config:
            return jsonify({'error': 'At least one feature and objective are required.'}), 400
        tweaks = _validated_tweaks(
            req_data.get('tweaks', {}), features_config, objectives_config
        )

        payload = {
            'data': data,
            'columns': columns,
            'features': features_config,
            'objectives': objectives_config,
            'tweaks': tweaks,
        }
        job_id = _queue_idempotent_background_job(
            'sa-analysis', payload, _run_sa_job, 'Queued statistical analysis'
        )
        if job_id is None:
            return jsonify({
                'error': 'Another analysis job is already running. '
                         'Wait for it to finish or cancel it first.'
            }), 429
        return jsonify({'job_id': job_id, 'status': 'queued'}), 202
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400


@api_bp.route('/sa_status/<job_id>', methods=['GET'])
def sa_job_status(job_id):
    response = _background_job_response(job_id)
    if response is None:
        return jsonify({'error': 'Analysis job not found or expired.'}), 404
    return jsonify(clean_json_data(response))


@api_bp.route('/sa_cancel/<job_id>', methods=['POST'])
def cancel_sa_job(job_id):
    response = _cancel_background_job(job_id)
    if response is None:
        return jsonify({'error': 'Analysis job not found or expired.'}), 404
    if response.get('status') == 'cancelling':
        _start_cancel_release_watchdog(job_id)
    return jsonify(response)


@api_bp.route('/job_status/<job_id>', methods=['GET'])
def background_job_status(job_id):
    response = _background_job_response(job_id)
    if response is None:
        return jsonify({'error': 'Background job not found or expired.'}), 404
    return jsonify(clean_json_data(response))


@api_bp.route('/job_cancel/<job_id>', methods=['POST'])
def cancel_background_job(job_id):
    response = _cancel_background_job(job_id)
    if response is None:
        return jsonify({'error': 'Background job not found or expired.'}), 404
    if response.get('status') == 'cancelling':
        _start_cancel_release_watchdog(job_id)
    return jsonify(response)


@api_bp.route('/sa', methods=['POST'])
def run_sa_endpoint():
    """Backward-compatible synchronous Statistical Analysis endpoint."""
    try:
        req_data = request.get_json()
        data = req_data.get('data')
        columns = req_data.get('columns')
        features_config = req_data.get('features', [])
        objectives_config = req_data.get('objectives', [])
        tweaks = _validated_tweaks(
            req_data.get('tweaks', {}), features_config, objectives_config
        )

        df = pd.DataFrame(data, columns=columns)

        output = run_sa(
            df=df,
            features_config=features_config,
            objectives_config=objectives_config,
            tweaks=tweaks
        )

        return jsonify(_format_sa_result(output))

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@api_bp.route('/sa_estimate_submit', methods=['POST'])
def sa_estimate_submit():
    """Queue SA simulation/estimation to avoid proxy timeouts for MLP fits."""
    try:
        req_data = request.get_json() or {}
        features = req_data.get('features', [])
        objectives = req_data.get('objectives', [])
        data = req_data.get('data')
        columns = req_data.get('columns')
        if not isinstance(data, list) or not isinstance(columns, list):
            return jsonify({'error': 'A valid dataset and columns are required.'}), 400
        tweaks = _validated_tweaks(req_data.get('tweaks', {}), features, objectives)
        payload = {
            'data': data,
            'columns': columns,
            'inputs': req_data.get('inputs', {}),
            'features': features,
            'objectives': objectives,
            'model': req_data.get('model', 'linear'),
            'params': req_data.get('params', {}),
            'tweaks': tweaks,
        }
        job_id = _queue_idempotent_background_job(
            'sa-estimation', payload, _run_sa_estimate_job,
            'Queued SA estimation',
        )
        if job_id is None:
            return jsonify({'error': 'Another analysis job is already running. Wait for it to finish or cancel it first.'}), 429
        return jsonify({'job_id': job_id, 'status': 'queued'}), 202
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400


@api_bp.route('/estimate', methods=['POST'])
def estimate_sa_endpoint():
    """Statistical Analysis point estimation endpoint (unchanged from EDOS)."""
    try:
        req_data = request.get_json()
        data = req_data.get('data')
        columns = req_data.get('columns')
        input_values = req_data.get('inputs')
        features_config = req_data.get('features', [])
        objectives_config = req_data.get('objectives', [])
        model_name = req_data.get('model', 'linear')
        params = req_data.get('params', {})
        tweaks = _validated_tweaks(
            req_data.get('tweaks', {}), features_config, objectives_config
        )

        df = pd.DataFrame(data, columns=columns)

        predictions, std_errors, success_feedback, global_score, violations = estimate_sa(
            df=df,
            input_values=input_values,
            features_config=features_config,
            objectives_config=objectives_config,
            model_name=model_name,
            params=params,
            tweaks=tweaks
        )

        return jsonify(clean_json_data({
            'predictions': predictions,
            'std_errors': std_errors,
            'success_scores': success_feedback,
            'global_success': global_score,
            'constraint_violated': bool(violations),
            'violations': violations
        }))

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@api_bp.errorhandler(500)
def handle_500_error(e):
    return jsonify({'error': f'Internal server error: {getattr(e, "description", str(e))}'}), 500


@api_bp.errorhandler(504)
def handle_504_error(e):
    return jsonify({'error': 'The server operation timed out. Please reduce calculation intensity.'}), 504
