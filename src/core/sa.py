import numpy as np
import os
import pandas as pd
import sklearn.linear_model as lm
import sklearn.ensemble as ensemble
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import ShuffleSplit
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import shap
from src.core.bo import infer_decimal_precision
from src.core.constraints import check_objective_constraints


class AnalysisCancelled(RuntimeError):
    """Raised when a queued SA job receives a cooperative cancellation request."""


def _check_cancel(cancel_event):
    if cancel_event is not None and cancel_event.is_set():
        raise AnalysisCancelled("Analysis aborted by user.")


def _report_progress(callback, phase, percent):
    if callback is not None:
        callback(phase, int(percent))


def _deterministic_sample_indices(target_values, max_samples):
    """Select representative rows reproducibly across the target distribution."""
    values = np.asarray(target_values, dtype=float).reshape(-1)
    if len(values) <= max_samples:
        return np.arange(len(values), dtype=int)
    ordered = np.argsort(values, kind='stable')
    positions = np.linspace(0, len(ordered) - 1, max_samples, dtype=int)
    return np.sort(ordered[positions])


def _cross_validated_r2(build_model, X_values, y_values):
    """Return deterministic held-out R² mean/std; never substitute training fit."""
    n_samples = len(y_values)
    if n_samples < 6:
        return None, None
    test_count = max(2, int(np.ceil(n_samples * 0.2)))
    if test_count >= n_samples:
        return None, None
    splitter = ShuffleSplit(n_splits=5, test_size=test_count, random_state=42)
    scores = []
    for train_idx, test_idx in splitter.split(X_values):
        model = build_model()
        model.fit(X_values[train_idx], y_values[train_idx])
        score = model.score(X_values[test_idx], y_values[test_idx])
        if np.isfinite(score):
            scores.append(float(score))
    if not scores:
        return None, None
    return float(np.mean(scores)), float(np.std(scores))


def _deterministic_kernel_shap(explainer, values, nsamples):
    """Run Kernel SHAP with an isolated reproducible NumPy random state."""
    random_state = np.random.get_state()
    np.random.seed(42)
    try:
        return explainer.shap_values(values, nsamples=nsamples, silent=True)
    finally:
        np.random.set_state(random_state)

def robust_numeric_series(col_series):
    """Safely converts a series to numeric, attempting string-level cleaning if needed."""
    num_col = pd.to_numeric(col_series, errors='coerce')
    if num_col.notna().sum() == len(col_series.dropna()):
        return num_col
    # Attempt cleaning percentages, currency symbols, and European decimal commas
    s_clean = col_series.astype(str).str.strip().str.rstrip('%').str.lstrip('$').str.replace(',', '.', regex=False)
    num_clean = pd.to_numeric(s_clean, errors='coerce')
    return num_clean if num_clean.notna().sum() > num_col.notna().sum() else num_col


def clean_dataframe_sa(df, features_config, objectives_config):
    df_clean = df.copy()
    valid_feat_mask = pd.Series(True, index=df_clean.index)

    # 1. Feature filtering and intelligent type correction
    for f in features_config:
        name = f['name']
        if name not in df_clean.columns:
            continue
        col_series = df_clean[name]
        non_empty = col_series.notna() & (col_series.astype(str).str.strip() != "")
        non_empty_cnt = non_empty.sum()

        num_col = robust_numeric_series(col_series)
        numeric_cnt = num_col.notna().sum()

        # If declared as continuous/discrete/regular but mostly non-numeric strings, auto-correct to categorical
        if f.get('type') in ['continuous', 'discrete', 'regular']:
            if non_empty_cnt > 0 and (numeric_cnt / non_empty_cnt) < 0.5:
                f['type'] = 'categorical'

        if f.get('type') in ['continuous', 'discrete', 'regular']:
            if numeric_cnt == 0:
                # Column has no valid numbers; fill with 0 so it doesn't wipe out the entire dataset
                df_clean[name] = 0.0
            else:
                valid_feat_mask = valid_feat_mask & num_col.notna()
        else:
            # Categorical feature: missing or empty values become 'None' instead of dropping rows
            df_clean[name] = df_clean[name].fillna('None').astype(str).str.strip()
            df_clean[name] = df_clean[name].replace({'': 'None', 'nan': 'None', 'None': 'None', 'null': 'None'})

    # If applying valid_feat_mask leaves valid rows, apply it; otherwise fallback to median imputation
    if valid_feat_mask.sum() > 0:
        df_clean = df_clean[valid_feat_mask].reset_index(drop=True).copy()
    else:
        for f in features_config:
            if f.get('type') in ['continuous', 'discrete', 'regular']:
                med = robust_numeric_series(df_clean[f['name']]).median()
                df_clean[f['name']] = robust_numeric_series(df_clean[f['name']]).fillna(med if pd.notna(med) else 0.0)

    if not objectives_config:
        return df_clean

    # 2. Objective filtering: Differentiate single vs multi-objective with robust parsing
    obj_df_numeric = pd.DataFrame()
    for obj in objectives_config:
        if obj['name'] in df_clean.columns:
            obj_df_numeric[obj['name']] = robust_numeric_series(df_clean[obj['name']])
        else:
            obj_df_numeric[obj['name']] = pd.Series(np.nan, index=df_clean.index)

    if len(objectives_config) == 1:
        # Single-objective case: drop any row missing the objective value
        valid_obj_mask = obj_df_numeric[objectives_config[0]['name']].notna()
    else:
        # Multi-objective case: drop rows missing ALL objectives
        valid_obj_mask = obj_df_numeric.notna().any(axis=1)

    df_clean = df_clean[valid_obj_mask].reset_index(drop=True).copy()
    return df_clean


def run_sa(
    df,
    features_config,
    objectives_config,
    tweaks,
    progress_callback=None,
    cancel_event=None,
):
    """
    Performs statistical analysis of experimental data to understand
    feature-outcome relationships.
    """
    _report_progress(progress_callback, 'Preparing and validating data', 5)
    _check_cancel(cancel_event)
    # Filter the dataframe first
    df_clean = clean_dataframe_sa(df, features_config, objectives_config)

    # Ensure categorical columns are stored as strings to prevent mixed-type issues
    for f in features_config:
        if f.get('type') == 'categorical':
            df_clean[f['name']] = df_clean[f['name']].astype(str)

    feat_names = [f['name'] for f in features_config if f.get('name')]
    obj_names  = [o['name'] for o in objectives_config]

    if not feat_names or not obj_names:
        raise ValueError('Missing features or objectives for analysis.')

    if df_clean.empty:
        raise ValueError('No valid data points left after filtering missing features/objectives.')

    original_rows = len(df_clean)
    shap_nsamples = 25 if original_rows > 1000 else 40
    shap_background_cap = min(20, original_rows)

    X_raw = df_clean[feat_names].copy()
    Y_raw = df_clean[obj_names].copy()

    # STEP 1 — Goal Transformation → Normalized Success Scores
    success_data      = {}
    processed_success = []
    weights           = []

    for obj in objectives_config:
        _check_cancel(cancel_event)
        name = obj['name']
        w    = float(obj.get('importance', 100))
        weights.append(w)
        
        # Get raw objective values (keep NaNs)
        raw_y = pd.to_numeric(Y_raw[name], errors='coerce').values
        
        if   obj['type'] == 'maximize': transformed = raw_y
        elif obj['type'] == 'minimize': transformed = -raw_y
        else:
            t          = float(obj.get('target', 0))
            transformed = -np.abs(raw_y - t)

        # Normalize per-objective ignoring NaNs
        valid_trans = transformed[~np.isnan(transformed)]
        if len(valid_trans) > 0:
            s_min, s_max = valid_trans.min(), valid_trans.max()
            if s_max > s_min:
                normalized = (transformed - s_min) / (s_max - s_min)
            else:
                normalized = np.ones_like(transformed)
        else:
            normalized = np.zeros_like(transformed)

        success_data[name] = normalized
        processed_success.append(normalized)

    # Compute weighted Global Success Index when multiple objectives present
    if len(processed_success) >= 2:
        succ_matrix = np.column_stack(processed_success)
        w_arr       = np.array(weights)
        
        global_success = []
        for row_idx in range(succ_matrix.shape[0]):
            row_vals = succ_matrix[row_idx]
            nan_mask = np.isnan(row_vals)
            
            valid_vals = row_vals[~nan_mask]
            valid_weights = w_arr[~nan_mask]
            
            if len(valid_vals) > 0 and np.sum(valid_weights) > 0:
                g_succ = np.average(valid_vals, weights=valid_weights)
            elif len(valid_vals) > 0:
                g_succ = np.mean(valid_vals)
            else:
                g_succ = 0.0
            global_success.append(g_succ)
            
        success_data['global_success'] = np.array(global_success)
    else:
        global_success = None

    Y_success         = pd.DataFrame(success_data)
    extended_obj_names = list(success_data.keys())
    _report_progress(progress_callback, 'Computing correlations and encoding features', 20)
    _check_cancel(cancel_event)

    # STEP 2 — Feature Encoding and Surrogate Modeling
    X_encoded       = pd.get_dummies(X_raw, drop_first=False)
    encoded_feat_names = list(X_encoded.columns)
    X = X_encoded.fillna(0).astype(float)
    Y = Y_success.apply(pd.to_numeric, errors='coerce').astype(float) # Keep NaNs in Y!

    # Correlation matrix in success space (fill NaNs with 0 for joint corr structure, or calculate pairwise)
    corr_matrix = pd.concat([X, Y], axis=1).corr().fillna(0).to_dict()

    model_name = tweaks.get('model', 'linear')
    params     = tweaks.get('params', {})

    results = {}
    for objective_index, obj in enumerate(extended_obj_names):
        _check_cancel(cancel_event)
        objective_progress = 30 + int(
            50 * objective_index / max(1, len(extended_obj_names))
        )
        _report_progress(
            progress_callback,
            f"Training model and computing SHAP values for {obj}",
            objective_progress,
        )
        y = Y[obj]
        
        # Smart Per-Objective Splitting: Filter out rows where objective is NaN
        valid_mask = y.notna()
        X_obj = X[valid_mask].reset_index(drop=True)
        y_obj = y[valid_mask].reset_index(drop=True)
        X_raw_obj = X_raw[valid_mask].reset_index(drop=True)

        n_samples = len(y_obj)
        if n_samples < 5:
            results[obj] = {
                'reliability': None,
                'reliability_std': None,
                'reliability_method': 'insufficient data for held-out validation',
                'importance': {f_name: 0.0 for f_name in feat_names},
                'shap_data': None,
                'fit_flag': 'Poor'
            }
            continue

        def build_model():
            if model_name == 'random_forest':
                return ensemble.RandomForestRegressor(
                    n_estimators=int(params.get('n_estimators', 100)),
                    max_depth=int(params.get('max_depth', 10)) if params.get('max_depth') else None,
                    n_jobs=-1,
                    oob_score=(n_samples >= 30),
                    random_state=42
                )
            elif model_name == 'mlp':
                hidden_layers = str(params.get('mlp_layers', '100'))
                layers = [int(x.strip()) for x in hidden_layers.split(',') if x.strip()]
                base_model = MLPRegressor(
                    hidden_layer_sizes=tuple(layers),
                    max_iter=min(1000, int(params.get('mlp_iter', 500))),
                    early_stopping=True,
                    n_iter_no_change=15,
                    validation_fraction=0.15,
                    tol=1e-4,
                    random_state=42,
                )
                return make_pipeline(
                    StandardScaler(),
                    TransformedTargetRegressor(
                        regressor=base_model,
                        transformer=StandardScaler(),
                    ),
                )
            else:
                return lm.LinearRegression()

        X_data = X_obj

        CHUNK_SIZE = 2500
        if n_samples > CHUNK_SIZE:
            # Divide-and-Conquer Stratified Partitioning (100% data accounted for)
            K = int(np.ceil(n_samples / CHUNK_SIZE))
            sorted_ranks = np.argsort(y_obj.values)
            chunk_indices = [[] for _ in range(K)]
            for rank_pos, idx_val in enumerate(sorted_ranks):
                chunk_indices[rank_pos % K].append(idx_val)

            chunk_reliabilities = []
            chunk_importances = []
            full_shap_matrix = (
                None if model_name == 'mlp'
                else np.zeros((n_samples, X_data.shape[1]))
            )
            shap_row_indices = None
            mlp_shap_blocks = []
            mlp_shap_indices = []

            for k in range(K):
                _check_cancel(cancel_event)
                c_idx = np.array(chunk_indices[k])
                X_chunk = X_data.iloc[c_idx]
                y_chunk = y_obj.iloc[c_idx]
                c_model = build_model()
                c_model.fit(X_chunk.values, y_chunk.values)

                # Reliability is always held-out and deterministic.  This is
                # deliberately stricter than the in-sample fit shown by TransBO.
                c_rel, c_rel_std = _cross_validated_r2(
                    build_model, X_chunk.values, y_chunk.values
                )
                chunk_reliabilities.append((len(c_idx), c_rel, c_rel_std))

                # Importance
                if model_name == 'linear':
                    c_imp = np.abs(c_model.coef_)
                elif model_name == 'mlp':
                    imp_res = permutation_importance(c_model, X_chunk.values, y_chunk.values, n_repeats=3, random_state=42)
                    c_imp = imp_res.importances_mean
                else:
                    c_imp = c_model.feature_importances_
                chunk_importances.append((len(c_idx), c_imp))

                # SHAP on this chunk
                try:
                    if model_name == 'random_forest':
                        explainer = shap.TreeExplainer(c_model)
                        s_vals = explainer.shap_values(X_chunk.values, check_additivity=False)
                    elif model_name == 'mlp':
                        bg_idx = _deterministic_sample_indices(
                            y_chunk.values, min(20, len(X_chunk))
                        )
                        eval_idx = _deterministic_sample_indices(
                            y_chunk.values, min(150, len(X_chunk))
                        )
                        bg = X_chunk.values[bg_idx]
                        explainer = shap.KernelExplainer(c_model.predict, bg)
                        s_vals = _deterministic_kernel_shap(
                            explainer, X_chunk.values[eval_idx], shap_nsamples
                        )
                    elif model_name == 'linear':
                        explainer = shap.LinearExplainer(c_model, X_chunk.values)
                        s_vals = explainer.shap_values(X_chunk.values)

                    if isinstance(s_vals, list):
                        s_vals = s_vals[0]
                    if model_name == 'mlp':
                        mlp_shap_blocks.append(np.asarray(s_vals))
                        mlp_shap_indices.append(c_idx[eval_idx])
                    else:
                        full_shap_matrix[c_idx] = s_vals
                except Exception as e_shap:
                    print(f"Chunk SHAP error: {e_shap}")

            # Joined reliability
            valid_reliabilities = [
                (n_c, rel, std) for n_c, rel, std in chunk_reliabilities
                if rel is not None and np.isfinite(rel)
            ]
            total_n = sum(n_c for n_c, _, _ in valid_reliabilities)
            reliability = (
                sum(n_c * rel for n_c, rel, _ in valid_reliabilities) / total_n
                if total_n else None
            )
            reliability_std = (
                sum(n_c * (std or 0.0) for n_c, _, std in valid_reliabilities) / total_n
                if total_n else None
            )

            # Joined importance
            weighted_imp = np.zeros(X_data.shape[1])
            for n_c, imp in chunk_importances:
                weighted_imp += n_c * imp
            importance = weighted_imp / n_samples

            if model_name == 'mlp' and mlp_shap_blocks:
                full_shap_matrix = np.vstack(mlp_shap_blocks)
                shap_row_indices = np.concatenate(mlp_shap_indices)

        else:
            # Single pass on 100% of data
            model = build_model()
            reliability, reliability_std = _cross_validated_r2(
                build_model, X_data.values, y_obj.values
            )
            model.fit(X_data.values, y_obj.values)

            if model_name == 'linear':
                importance = np.abs(model.coef_)
            elif model_name == 'mlp':
                imp_res = permutation_importance(model, X_data.values, y_obj.values, n_repeats=3, random_state=42)
                importance = imp_res.importances_mean
            else:
                importance = model.feature_importances_

            full_shap_matrix = None
            shap_row_indices = None
            try:
                if model_name == 'random_forest':
                    explainer = shap.TreeExplainer(model)
                    s_vals = explainer.shap_values(X_data.values, check_additivity=False)
                elif model_name == 'mlp':
                    bg_idx = _deterministic_sample_indices(
                        y_obj.values, min(20, len(X_data))
                    )
                    eval_idx = _deterministic_sample_indices(
                        y_obj.values, min(250, len(X_data))
                    )
                    bg = X_data.values[bg_idx]
                    explainer = shap.KernelExplainer(model.predict, bg)
                    s_vals = _deterministic_kernel_shap(
                        explainer, X_data.values[eval_idx], shap_nsamples
                    )
                    shap_row_indices = eval_idx
                elif model_name == 'linear':
                    explainer = shap.LinearExplainer(model, X_data.values)
                    s_vals = explainer.shap_values(X_data.values)

                if isinstance(s_vals, list):
                    s_vals = s_vals[0]
                full_shap_matrix = s_vals
            except Exception as e:
                print(f"SHAP Explainer execution failed on {model_name}: {e}")

        imp_sum = np.sum(importance)
        importance_pct = (importance / imp_sum * 100).tolist() if imp_sum > 0 else [0.0] * len(importance)

        grouped_importance = {}
        for f_name in feat_names:
            rel_indices = [i for i, c in enumerate(encoded_feat_names) if c == f_name or f"{f_name}_" in c]
            grouped_importance[f_name] = round(sum(importance_pct[i] for i in rel_indices), 2)

        shap_data = None
        if full_shap_matrix is not None:
            full_shap_matrix = np.asarray(full_shap_matrix, dtype=float)
            if full_shap_matrix.ndim == 3 and full_shap_matrix.shape[-1] == 1:
                full_shap_matrix = full_shap_matrix[..., 0]
            if shap_row_indices is None:
                shap_row_indices = np.arange(len(full_shap_matrix), dtype=int)
            shap_row_indices = np.asarray(shap_row_indices, dtype=int)
            finite_rows = np.all(np.isfinite(full_shap_matrix), axis=1)
            full_shap_matrix = full_shap_matrix[finite_rows]
            shap_row_indices = shap_row_indices[finite_rows]

            grouped_shap = {
                f: np.zeros(len(full_shap_matrix), dtype=float) for f in feat_names
            }
            for f_name in feat_names:
                rel_indices = [i for i, c in enumerate(encoded_feat_names) if c == f_name or f"{f_name}_" in c]
                for idx in rel_indices:
                    grouped_shap[f_name] += full_shap_matrix[:, idx]

            grouped_abs_values = [
                np.abs(values) for values in grouped_shap.values()
                if len(values)
            ]
            all_abs = (
                np.concatenate(grouped_abs_values)
                if grouped_abs_values else np.array([])
            )
            display_limit = (
                float(np.quantile(all_abs, 0.99)) if len(all_abs) else 0.0
            )
            display_limit = max(display_limit, 1e-9)
            clipped_count = int(np.sum(all_abs > display_limit))
            sampled_raw = X_raw_obj.iloc[shap_row_indices]

            shap_data = {
                'features': feat_names,
                'shap_values': [grouped_shap[f].tolist() for f in feat_names],
                'feature_values': [sampled_raw[f].tolist() for f in feat_names],
                'sampled_rows': int(len(shap_row_indices)),
                'total_rows': int(n_samples),
                'sampling': 'deterministic_target_stratified',
                'display_limit': display_limit,
                'display_clipped_points': clipped_count,
            }

        results[obj] = {
            'reliability': None if (reliability is None or np.isnan(reliability)) else round(float(reliability), 3),
            'reliability_std': None if (reliability_std is None or np.isnan(reliability_std)) else round(float(reliability_std), 3),
            'reliability_method': '5x deterministic held-out validation',
            'importance': grouped_importance,
            'shap_data': shap_data,
            'fit_flag': 'Good' if (reliability is not None and reliability > 0.7) else ('Limited' if (reliability is not None and reliability >= 0.4) else 'Poor')
        }

        _check_cancel(cancel_event)

    # STEP 3 — Categorical Interaction Pivot Tables
    cat_features = [f['name'] for f in features_config if f.get('type') == 'categorical']
    num_features = [f['name'] for f in features_config if f.get('type') in ['continuous', 'discrete']]

    _report_progress(progress_callback, 'Computing categorical interactions', 88)
    _check_cancel(cancel_event)
    cat_interactions = {}
    for i in range(len(cat_features)):
        for j in range(i + 1, len(cat_features)):
            f1, f2   = cat_features[i], cat_features[j]
            pair_key = f"{f1} vs {f2}"
            cat_interactions[pair_key] = {}
            for obj in extended_obj_names:
                combo_df = pd.concat([df_clean[[f1, f2]], Y_success[[obj]]], axis=1)
                pivot    = combo_df.groupby([f1, f2])[obj].max().unstack().fillna(0)
                cat_interactions[pair_key][obj] = {
                    'z': pivot.values.tolist(),
                    'x': pivot.columns.tolist(),
                    'y': pivot.index.tolist()
                }

    # Numerical Pearson correlation (guarded for datasets with zero numerical features)
    if num_features:
        num_df = df_clean[num_features].apply(pd.to_numeric, errors='coerce').fillna(0)
        num_corr_matrix = (
            pd.concat([num_df, Y_success], axis=1)
            .corr().loc[extended_obj_names, num_features].fillna(0).to_dict()
        )
    else:
        num_corr_matrix = {}

    metadata = {
        'original_rows': original_rows,
        'shap_background_rows': min(original_rows, shap_background_cap),
        'shap_nsamples': shap_nsamples,
        'shap_approximated': bool(model_name == 'mlp' and original_rows > 250),
        'shap_sampling': 'deterministic_target_stratified',
    }
    _report_progress(progress_callback, 'Finalizing analysis', 100)
    return (
        num_corr_matrix,
        cat_interactions,
        results,
        cat_features,
        num_features,
        extended_obj_names,
        metadata,
    )


def estimate_sa(
    df,
    input_values,
    features_config,
    objectives_config,
    model_name,
    params,
    tweaks=None,
):
    """
    Predicts objective values at a user-specified operating point using the
    same sklearn surrogate model fitted in run_sa().
    """
    df_clean = clean_dataframe_sa(df, features_config, objectives_config)
    if df_clean.empty:
        raise ValueError('No valid data points left after filtering missing features/objectives.')

    feat_names = [f['name'] for f in features_config if f.get('name')]
    obj_names  = [o['name'] for o in objectives_config]

    # Build Training Feature Matrix
    X_raw_train    = df_clean[feat_names].copy()
    X_train_encoded = pd.get_dummies(X_raw_train)
    encoded_cols   = X_train_encoded.columns
    X_train        = X_train_encoded.fillna(0).astype(float)

    for o in obj_names:
        if df_clean[o].dtype == object or str(df_clean[o].dtype) in ['string', 'category']:
            df_clean[o] = df_clean[o].astype(str).astype('category').cat.codes
    Y = df_clean[obj_names].apply(pd.to_numeric, errors='coerce').astype(float) # Keep NaNs

    # Keep interactive estimation bounded on hosted CPU deployments.  Sampling
    # is deterministic and stratified by the first available objective.
    if model_name == 'mlp':
        estimator_cap = max(500, int(os.environ.get('SA_ESTIMATOR_MAX_ROWS', '5000')))
        if len(X_train) > estimator_cap:
            stratify_values = Y.bfill(axis=1).iloc[:, 0].fillna(0).values
            keep = _deterministic_sample_indices(stratify_values, estimator_cap)
            X_train = X_train.iloc[keep].reset_index(drop=True)
            Y = Y.iloc[keep].reset_index(drop=True)

    # Encode the Query Point
    input_data_raw = pd.DataFrame([input_values], columns=feat_names)

    for col in feat_names:
        if df_clean[col].dtype == object or str(df_clean[col].dtype) in ['string', 'category']:
            unique_train_vals  = df_clean[col].astype(str).unique()
            input_data_raw[col] = pd.Categorical(input_data_raw[col].astype(str), categories=unique_train_vals)

    input_df_encoded = pd.get_dummies(input_data_raw)
    input_df_aligned = input_df_encoded.reindex(columns=encoded_cols, fill_value=0).astype(float)

    # Scale Features
    X_train_data = X_train
    input_data   = input_df_aligned
    if model_name == 'mlp':
        scaler       = StandardScaler()
        X_train_data = pd.DataFrame(scaler.fit_transform(X_train), columns=encoded_cols)
        input_data   = pd.DataFrame(scaler.transform(input_df_aligned), columns=encoded_cols)

    # Fit Model and Predict per-objective using valid data
    predictions = {}
    std_errors  = {}
    for obj in obj_names:
        prec = infer_decimal_precision(df_clean[obj])
        y_obj = Y[obj]
        valid_mask = y_obj.notna()
        X_tr_obj = X_train_data[valid_mask]
        y_tr_obj = y_obj[valid_mask]

        if len(y_tr_obj) == 0:
            predictions[obj] = 0.0
            std_errors[obj]  = 0.0
            continue

        if model_name == 'random_forest':
            model = ensemble.RandomForestRegressor(
                n_estimators=int(params.get('n_estimators', 100)),
                max_depth=int(params.get('max_depth', 10)) if params.get('max_depth') else None,
                n_jobs=-1,
                random_state=42
            )
            model.fit(X_tr_obj, y_tr_obj)
            pred = model.predict(input_data)[0]
            input_mat = input_data.values if hasattr(input_data, 'values') else input_data
            tree_preds = [tree.predict(input_mat)[0] for tree in model.estimators_]
            std_err = float(np.std(tree_preds))
        elif model_name == 'mlp':
            hidden_layers = str(params.get('mlp_layers', '100'))
            layers = [int(x.strip()) for x in hidden_layers.split(',') if x.strip()]
            model = TransformedTargetRegressor(
                regressor=MLPRegressor(
                    hidden_layer_sizes=tuple(layers),
                    max_iter=min(1000, int(params.get('mlp_iter', 500))),
                    early_stopping=True,
                    n_iter_no_change=15,
                    validation_fraction=0.15,
                    tol=1e-4,
                    random_state=42,
                ),
                transformer=StandardScaler(),
            )
            model.fit(X_tr_obj, y_tr_obj)
            pred = model.predict(input_data)[0]
            y_pred_tr = model.predict(X_tr_obj)
            residuals = y_tr_obj.values - y_pred_tr
            dof = max(1, len(y_tr_obj) - 1)
            std_err = float(np.sqrt(np.sum(residuals**2) / dof))
        else:
            model = lm.LinearRegression()
            model.fit(X_tr_obj, y_tr_obj)
            pred = model.predict(input_data)[0]
            y_pred_tr = model.predict(X_tr_obj)
            residuals = y_tr_obj.values - y_pred_tr
            n = len(y_tr_obj)
            p = X_tr_obj.shape[1]
            dof = max(1, n - p - 1)
            s_e2 = np.sum(residuals**2) / dof
            try:
                X_mat = X_tr_obj.values
                x0 = input_data.values[0]
                XtX_inv = np.linalg.pinv(X_mat.T @ X_mat)
                leverage = float(x0 @ XtX_inv @ x0)
                std_err = float(np.sqrt(s_e2 * (1.0 + leverage)))
            except Exception:
                std_err = float(np.sqrt(s_e2))

        predictions[obj] = round(float(pred), prec)
        std_errors[obj]  = round(float(std_err), prec + 2)

    # Compute Success Scores
    success_feedback  = {}
    processed_success = []
    weights           = []

    for obj in objectives_config:
        name = obj['name']
        val  = predictions[name]
        w    = float(obj.get('importance', 100))
        weights.append(w)

        if   obj['type'] == 'maximize': s = val
        elif obj['type'] == 'minimize': s = -val
        else:
            t = float(obj.get('target', 0))
            s = -abs(val - t)

        train_raw = Y[name].values
        valid_train = train_raw[~np.isnan(train_raw)]
        
        if len(valid_train) == 0:
            success_feedback[name] = 0.0
            processed_success.append(np.nan)
            continue

        if   obj['type'] == 'maximize': train_s = valid_train
        elif obj['type'] == 'minimize': train_s = -valid_train
        else:               train_s = -np.abs(valid_train - float(obj.get('target', 0)))

        s_min, s_max = train_s.min(), train_s.max()
        norm_s = (s - s_min) / (s_max - s_min) if s_max > s_min else 1.0

        success_feedback[name] = round(float(norm_s * 100), 1)
        processed_success.append(norm_s)

    _, violations = check_objective_constraints(
        predictions, (tweaks or {}).get('objective_constraints', [])
    )
    violated_objectives = {item['objective'] for item in violations}
    for objective_name in violated_objectives:
        if objective_name in success_feedback:
            success_feedback[objective_name] = 0.0

    processed_success = [
        float(success_feedback.get(obj['name'], 0.0)) / 100.0
        for obj in objectives_config
    ]

    # Calculate global success with dynamic renormalization of weights
    if processed_success:
        succ_arr = np.array(processed_success)
        w_arr = np.array(weights)
        nan_mask = np.isnan(succ_arr)
        
        valid_succ = succ_arr[~nan_mask]
        valid_weights = w_arr[~nan_mask]
        
        if len(valid_succ) > 0 and np.sum(valid_weights) > 0:
            global_score = np.average(valid_succ, weights=valid_weights)
        elif len(valid_succ) > 0:
            global_score = np.mean(valid_succ)
        else:
            global_score = 0.0
    else:
        global_score = 0

    return (
        predictions,
        std_errors,
        success_feedback,
        round(float(global_score * 100), 1),
        violations,
    )
