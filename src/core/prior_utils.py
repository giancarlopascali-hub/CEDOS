"""
TransEDOS - Prior Columns and Transfer Learning Utilities
==========================================================
Provides heuristics for prior column type detection (Categorical, Numerical, SMILES),
chemical fingerprint calculations via RDKit, distance metrics, and row-wise
transfer learning weight combinations.
"""

import numpy as np
import pandas as pd

# Hard dependency check: RDKit must be importable
try:
    import rdkit
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    RDKIT_AVAILABLE = True
except ImportError as e:
    RDKIT_AVAILABLE = False
    raise ImportError(
        "RDKit is a required hard dependency for TransEDOS. "
        "Please ensure 'rdkit' or 'rdkit-pypi' is installed."
    ) from e


def assert_rdkit():
    """Verify that RDKit is functional."""
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is not available in the current environment.")


def detect_prior_column_type(values) -> str:
    """
    Heuristic classifier for prior columns.
    Returns: 'numerical' | 'smiles' | 'categorical'

    Rules:
    1. Filter out null / empty / NaN values.
    2. If all valid values parse as numeric -> 'numerical'.
    3. If not numeric, test first up to 10 distinct values as SMILES with RDKit.
       If >= 60% parse successfully into valid molecules -> 'smiles'.
    4. Otherwise -> 'categorical'.
    """
    if values is None:
        return "categorical"

    s = pd.Series(values).dropna()
    # Filter empty strings or whitespace-only
    s = s[s.astype(str).str.strip() != ""]
    if len(s) == 0:
        return "categorical"

    # 1. Numeric check
    numeric_s = pd.to_numeric(s, errors="coerce")
    if numeric_s.notna().all():
        return "numerical"

    # 2. SMILES check using RDKit
    assert_rdkit()
    sample_values = s.astype(str).str.strip().unique()[:10]
    valid_mol_count = 0
    total_tested = 0

    for val in sample_values:
        val_str = str(val).strip()
        if len(val_str) == 0:
            continue
        total_tested += 1
        try:
            mol = Chem.MolFromSmiles(val_str)
            if mol is not None and mol.GetNumAtoms() > 0:
                valid_mol_count += 1
        except Exception:
            pass

    if total_tested > 0 and (valid_mol_count / total_tested) >= 0.6:
        return "smiles"

    # 3. Categorical fallback
    return "categorical"


# Fingerprint generator & cache
try:
    from rdkit.Chem import rdFingerprintGenerator
    _MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
except Exception:
    _MORGAN_GEN = None

_FP_CACHE = {}


def _is_null_or_empty(val) -> bool:
    """Check if a value is None, NaN, or an empty/null placeholder string."""
    if val is None:
        return True
    if isinstance(val, (float, np.floating)) and (np.isnan(val) or np.isinf(val)):
        return True
    s = str(val).strip().lower()
    return s in ("", "none", "nan", "null", "nat", "undefined")


def _get_morgan_fp(smiles_str: str):
    """Parses SMILES and returns Morgan fingerprint with caching."""
    if _is_null_or_empty(smiles_str):
        return None
    s = str(smiles_str).strip()
    if s in _FP_CACHE:
        return _FP_CACHE[s]
    try:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            _FP_CACHE[s] = None
            return None
        if _MORGAN_GEN is not None:
            fp = _MORGAN_GEN.GetFingerprint(mol)
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        _FP_CACHE[s] = fp
        return fp
    except Exception:
        _FP_CACHE[s] = None
        return None


def smiles_tanimoto(smiles_a: str, smiles_b: str) -> float:
    """
    Computes Morgan fingerprint (radius 2, 2048 bits) Tanimoto similarity
    between two SMILES strings.

    If either string fails to parse (e.g. malformed user input),
    falls back defensively to exact string equality (1.0 if identical, 0.0 otherwise).
    """
    assert_rdkit()
    if _is_null_or_empty(smiles_a) or _is_null_or_empty(smiles_b):
        return 0.0

    str_a = str(smiles_a).strip()
    str_b = str(smiles_b).strip()

    if str_a == str_b:
        return 1.0

    try:
        fp_a = _get_morgan_fp(str_a)
        fp_b = _get_morgan_fp(str_b)

        if fp_a is None or fp_b is None:
            return 1.0 if str_a == str_b else 0.0

        sim = float(DataStructs.TanimotoSimilarity(fp_a, fp_b))
        return float(np.clip(sim, 0.0, 1.0))
    except Exception:
        return 1.0 if str_a == str_b else 0.0


def _smoothstep(values):
    """Map normalized closeness to a smooth 0-1 relevance score."""
    values = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    return values * values * (3.0 - (2.0 * values))


def _default_numerical_tolerance(source_values) -> float:
    """Default to half the observed numerical span, as shown in the UI."""
    numeric = pd.to_numeric(pd.Series(source_values), errors="coerce").dropna()
    if numeric.empty:
        return 0.0
    span = float(numeric.max() - numeric.min())
    return max(0.0, span / 2.0)


def compute_transfer_relevance(
    source_values,
    target_value,
    col_type: str,
    tolerance=None,
    min_similarity=None,
):
    """Return per-row relevance, eligibility, and the resolved prior setting.

    Eligibility is intentionally separate from relevance. Numerical and SMILES
    priors have explicit, user-visible acceptance boundaries, while relevance
    changes smoothly within those boundaries. Categorical priors remain exact
    0/1 matches.
    """
    n_rows = len(source_values)
    weights = np.zeros(n_rows, dtype=float)
    eligible = np.zeros(n_rows, dtype=bool)
    col_type = str(col_type).lower().strip()

    if n_rows == 0:
        return weights, eligible, {}

    if col_type == "categorical":
        if _is_null_or_empty(target_value):
            return weights, eligible, {}
        target_str = str(target_value).strip()
        eligible = np.array([
            not _is_null_or_empty(value) and str(value).strip() == target_str
            for value in source_values
        ], dtype=bool)
        return eligible.astype(float), eligible, {}

    if col_type == "numerical":
        if _is_null_or_empty(target_value):
            resolved = _default_numerical_tolerance(source_values)
            return weights, eligible, {"tolerance": resolved}
        try:
            target_num = float(target_value)
        except (TypeError, ValueError):
            raise ValueError("Numerical Prior target must be a valid number.")
        if not np.isfinite(target_num):
            raise ValueError("Numerical Prior target must be finite.")

        if tolerance in (None, ""):
            resolved_tolerance = _default_numerical_tolerance(source_values)
        else:
            try:
                resolved_tolerance = float(tolerance)
            except (TypeError, ValueError):
                raise ValueError("Numerical Prior tolerance must be a valid number.")
        if not np.isfinite(resolved_tolerance) or resolved_tolerance < 0:
            raise ValueError("Numerical Prior tolerance must be zero or greater.")

        numeric = pd.to_numeric(pd.Series(source_values), errors="coerce")
        valid = numeric.notna().to_numpy()
        source = numeric.fillna(0.0).to_numpy(dtype=float)
        distances = np.abs(source - target_num)
        if resolved_tolerance == 0:
            eligible = valid & np.isclose(
                source, target_num, rtol=1e-9, atol=1e-12
            )
            weights[eligible] = 1.0
        else:
            eligible = valid & (distances <= resolved_tolerance)
            closeness = 1.0 - (distances / resolved_tolerance)
            weights[eligible] = _smoothstep(closeness[eligible])
        return weights, eligible, {"tolerance": resolved_tolerance}

    if col_type == "smiles":
        if min_similarity in (None, ""):
            resolved_similarity = 0.5
        else:
            try:
                resolved_similarity = float(min_similarity)
            except (TypeError, ValueError):
                raise ValueError("SMILES minimum similarity must be a valid number.")
        if not np.isfinite(resolved_similarity) or not 0.0 <= resolved_similarity <= 1.0:
            raise ValueError("SMILES minimum similarity must be between 0 and 1.")
        if _is_null_or_empty(target_value):
            return weights, eligible, {"min_similarity": resolved_similarity}

        target_smiles = str(target_value).strip()
        similarities = np.zeros(n_rows, dtype=float)
        for index, source_value in enumerate(source_values):
            if _is_null_or_empty(source_value):
                continue
            similarities[index] = smiles_tanimoto(source_value, target_smiles)

        eligible = similarities >= resolved_similarity
        if resolved_similarity >= 1.0:
            weights[eligible] = 1.0
        else:
            closeness = (
                (similarities - resolved_similarity) / (1.0 - resolved_similarity)
            )
            weights[eligible] = _smoothstep(closeness[eligible])
        return weights, eligible, {"min_similarity": resolved_similarity}

    raise ValueError(
        f"Unsupported Prior type '{col_type}'. Choose categorical, numerical, or SMILES."
    )


def compute_transfer_weights(
    source_values,
    target_value,
    col_type: str,
    tolerance=None,
    min_similarity=None,
) -> np.ndarray:
    """Return only the smooth relevance values for a single Prior."""
    weights, _, _ = compute_transfer_relevance(
        source_values,
        target_value,
        col_type,
        tolerance=tolerance,
        min_similarity=min_similarity,
    )
    return weights


def combine_weights(weights_list: list, method: str = "multiply", prior_weights: list = None) -> np.ndarray:
    """
    Combines per-prior weight vectors into a single row-wise transfer weight vector.

    Parameters:
    - weights_list: list of 1D numpy arrays, each shape (n_rows,).
    - method: 'multiply' (strict) or 'average' (loose).
    - prior_weights: optional list of float weights (e.g. [100, 50]) used when method='average'.

    Returns:
    - 1D numpy array of shape (n_rows,) in [0.0, 1.0].
    """
    if not weights_list:
        return np.array([], dtype=float)

    n_rows = len(weights_list[0])
    if len(weights_list) == 1:
        return np.clip(weights_list[0], 0.0, 1.0).astype(float)

    method = str(method).lower().strip()

    if method == "multiply":
        combined = np.ones(n_rows, dtype=float)
        for w in weights_list:
            combined = combined * np.asarray(w, dtype=float)
        return np.clip(combined, 0.0, 1.0).astype(float)

    elif method == "average":
        arr_stack = np.vstack([np.asarray(w, dtype=float) for w in weights_list])
        if prior_weights is not None and len(prior_weights) == len(weights_list):
            p_w = np.array(prior_weights, dtype=float)
            if np.sum(p_w) > 0:
                p_w = p_w / np.sum(p_w)
                combined = np.average(arr_stack, axis=0, weights=p_w)
            else:
                combined = np.mean(arr_stack, axis=0)
        else:
            combined = np.mean(arr_stack, axis=0)
        return np.clip(combined, 0.0, 1.0).astype(float)

    else:
        # Default to multiply
        combined = np.prod(np.vstack(weights_list), axis=0)
        return np.clip(combined, 0.0, 1.0).astype(float)


def compute_combined_prior_relevance(
    dataframe,
    priors_config,
    method="multiply",
):
    """Calculate one authoritative relevance profile for all TransBO paths.

    ``multiply`` requires every configured Prior to accept a row. ``average``
    accepts a row when at least one positively weighted Prior accepts it. The
    smooth combined weight remains available for ranking and heteroscedastic GP
    noise, but it no longer acts as a second, hidden eligibility threshold.
    """
    row_count = len(dataframe)
    method = str(method or "multiply").lower().strip()
    if method not in {"multiply", "average"}:
        raise ValueError("Prior combination rule must be Multiply or Average.")

    weight_vectors = []
    eligibility_masks = []
    prior_weights = []
    prior_stats = {}

    for prior in priors_config or []:
        name = prior.get("name") if isinstance(prior, dict) else None
        if not name or name not in dataframe.columns:
            continue
        col_type = prior.get("col_type") or detect_prior_column_type(
            dataframe[name].values
        )
        weights, eligible, resolved = compute_transfer_relevance(
            dataframe[name].values,
            target_value=prior.get("target_value", ""),
            col_type=col_type,
            tolerance=prior.get("tolerance"),
            min_similarity=prior.get("min_similarity"),
        )
        try:
            importance = float(prior.get("prior_weight", 100.0))
        except (TypeError, ValueError):
            raise ValueError(f"Prior weight for '{name}' must be a valid number.")
        if not np.isfinite(importance) or importance < 0:
            raise ValueError(f"Prior weight for '{name}' must be zero or greater.")

        weight_vectors.append(weights)
        eligibility_masks.append(eligible)
        prior_weights.append(importance)
        stats = {
            "col_type": str(col_type).lower().strip(),
            "target_value": str(prior.get("target_value", "")),
            "min": round(float(np.min(weights)), 3) if len(weights) else 0.0,
            "mean": round(float(np.mean(weights)), 3) if len(weights) else 0.0,
            "max": round(float(np.max(weights)), 3) if len(weights) else 0.0,
            "n_eligible": int(np.sum(eligible)),
            "n_effective": int(np.sum(eligible)),
            "total_rows": int(len(weights)),
        }
        stats.update(resolved)
        prior_stats[name] = stats

    if not weight_vectors:
        return {
            "combined_weights": np.ones(row_count, dtype=float),
            "eligible_mask": np.ones(row_count, dtype=bool),
            "eligible_rows": int(row_count),
            "prior_stats": prior_stats,
        }

    combined_weights = combine_weights(
        weight_vectors,
        method=method,
        prior_weights=prior_weights,
    )
    if method == "multiply":
        eligible_mask = np.logical_and.reduce(eligibility_masks)
    else:
        active_masks = [
            mask for mask, importance in zip(eligibility_masks, prior_weights)
            if importance > 0
        ]
        if not active_masks:
            raise ValueError("Average mode requires at least one Prior weight above zero.")
        eligible_mask = np.logical_or.reduce(active_masks)

    # Relevance outside the explicit eligibility rules is always exactly zero.
    combined_weights = np.asarray(combined_weights, dtype=float)
    combined_weights[~eligible_mask] = 0.0
    return {
        "combined_weights": combined_weights,
        "eligible_mask": np.asarray(eligible_mask, dtype=bool),
        "eligible_rows": int(np.sum(eligible_mask)),
        "prior_stats": prior_stats,
    }
