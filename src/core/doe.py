"""Design-of-experiments generation and quality assessment."""

from __future__ import annotations

import itertools
import math
import random
import re

import numpy as np
import pydoe as doe
from scipy.linalg import qr

from src.core.bo import infer_decimal_precision
from src.core.constraints import check_constraints, get_categorical_choices


SUPPORTED_MODELS = {
    "bbdesign",
    "ccdesign",
    "fracfact",
    "fullfact",
    "def_screening",
    "lhs",
    "doptimal",
}

MODEL_LABELS = {
    "bbdesign": "Box-Behnken",
    "ccdesign": "Central Composite",
    "fracfact": "Fractional Factorial",
    "fullfact": "Full Factorial",
    "def_screening": "Definitive Screening",
    "lhs": "Latin Hypercube",
    "doptimal": "D-Optimal",
}


def _as_bool(value) -> bool:
    """Parse request booleans without treating the string ``"false"`` as true."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _parse_features(features_config):
    valid_features = []
    for feature in features_config:
        name = feature.get("name", "Unnamed")
        feature_type = feature.get("type", "continuous")
        range_text = feature.get("range", "")
        if not range_text:
            continue

        if feature_type == "categorical":
            choices = get_categorical_choices(feature)
            if choices:
                valid_features.append(
                    {
                        "name": name,
                        "choices": choices,
                        "type": feature_type,
                        "range": range_text,
                    }
                )
            continue

        numbers = [float(match) for match in re.findall(r"[-+]?\d*\.?\d+", range_text)]
        if feature_type == "discrete":
            levels = sorted(set(numbers))
            if len(levels) >= 2:
                valid_features.append(
                    {
                        "name": name,
                        "low": levels[0],
                        "high": levels[-1],
                        "levels": levels,
                        "type": feature_type,
                        "range": range_text,
                    }
                )
        elif feature_type == "regular" and len(numbers) >= 2:
            low, high = min(numbers[0], numbers[1]), max(numbers[0], numbers[1])
            steps = max(2, int(numbers[2])) if len(numbers) >= 3 else 10
            valid_features.append(
                {
                    "name": name,
                    "low": low,
                    "high": high,
                    "levels": np.linspace(low, high, steps).tolist(),
                    "type": feature_type,
                    "range": range_text,
                }
            )
        elif len(numbers) >= 2:
            low, high = min(numbers[0], numbers[1]), max(numbers[0], numbers[1])
            if high > low:
                valid_features.append(
                    {
                        "name": name,
                        "low": low,
                        "high": high,
                        "type": feature_type,
                        "range": range_text,
                    }
                )
    return valid_features


def _move_centers_to_end(design):
    """Place center rows last while preserving order within both partitions."""
    is_center = np.all(np.abs(design) < 1e-9, axis=1)
    centers = design[is_center]
    ordered = np.vstack([design[~is_center], centers]) if len(centers) else design.copy()
    return ordered, int(is_center.sum())


def _feature_codes(feature):
    if feature["type"] == "categorical":
        count = len(feature["choices"])
    elif feature["type"] in {"regular", "discrete"}:
        return 2.0 * (np.asarray(feature["levels"]) - feature["low"]) / (
            feature["high"] - feature["low"]
        ) - 1.0
    else:
        return None
    return np.linspace(-1.0, 1.0, count) if count > 1 else np.array([0.0])


def _decode_row(row, features, model_type):
    entry = {}
    for column, feature in enumerate(features):
        coded_value = float(row[column])
        if feature["type"] == "categorical":
            codes = _feature_codes(feature)
            index = int(np.argmin(np.abs(codes - coded_value)))
            entry[feature["name"]] = feature["choices"][index]
            continue

        value = feature["low"] + (coded_value + 1.0) * 0.5 * (
            feature["high"] - feature["low"]
        )
        if feature["type"] in {"regular", "discrete"}:
            value = min(feature["levels"], key=lambda candidate: abs(candidate - value))
            entry[feature["name"]] = float(value)
            continue

        precision = infer_decimal_precision(None, feature.get("range", ""))
        # Space-filling and optimal designs need enough precision to avoid
        # collapsing distinct continuous candidate points during decoding.
        if model_type in {"lhs", "doptimal"} and feature["type"] == "continuous":
            precision = max(precision, 6)
        entry[feature["name"]] = round(float(value), precision)
    return entry


def _encode_table(table, features):
    encoded = np.zeros((len(table), len(features)), dtype=float)
    for row_index, entry in enumerate(table):
        for column, feature in enumerate(features):
            value = entry[feature["name"]]
            if feature["type"] == "categorical":
                codes = _feature_codes(feature)
                encoded[row_index, column] = codes[feature["choices"].index(value)]
            else:
                encoded[row_index, column] = (
                    2.0 * (float(value) - feature["low"])
                    / (feature["high"] - feature["low"])
                    - 1.0
                )
    return encoded


def _main_effect_columns(coded_design, features):
    """Return model columns grouped by original factor for valid interactions."""
    groups = []
    for column, feature in enumerate(features):
        values = coded_design[:, column]
        if feature["type"] != "categorical":
            groups.append(values.reshape(-1, 1))
            continue

        choices = feature["choices"]
        codes = _feature_codes(feature)
        indices = np.argmin(np.abs(values[:, None] - codes[None, :]), axis=1)
        # Treatment coding (first level is the baseline) gives m-1 estimable
        # columns and works for categorical factors with any number of levels.
        groups.append(
            np.column_stack([(indices == level).astype(float) for level in range(1, len(choices))])
            if len(choices) > 1 else np.empty((len(values), 0))
        )
    return groups


def _build_model_matrix(coded_design, features, model_spec):
    coded_design = np.asarray(coded_design, dtype=float)
    groups = _main_effect_columns(coded_design, features)
    columns = [np.ones((len(coded_design), 1))]
    columns.extend(groups)

    if model_spec in {"interaction", "quadratic", "response_surface", "factorial"}:
        for first in range(len(groups)):
            for second in range(first + 1, len(groups)):
                columns.append(
                    np.einsum("ni,nj->nij", groups[first], groups[second]).reshape(
                        len(coded_design), -1
                    )
                )

    if model_spec in {"quadratic", "response_surface", "dsd", "factorial"}:
        for index, feature in enumerate(features):
            if feature["type"] != "categorical":
                if model_spec == "factorial" and len(feature.get("levels", [])) < 3:
                    continue
                columns.append((coded_design[:, index] ** 2).reshape(-1, 1))

    return np.hstack([column for column in columns if column.shape[1] > 0])


def _d_efficiency(model_matrix):
    n_runs, n_parameters = model_matrix.shape
    if n_runs == 0 or np.linalg.matrix_rank(model_matrix) < n_parameters:
        return 0.0
    sign, log_determinant = np.linalg.slogdet(model_matrix.T @ model_matrix)
    if sign <= 0:
        return 0.0
    return float(math.exp(log_determinant / n_parameters) / n_runs * 100.0)


def _orthogonality(coded_design, features):
    if len(coded_design) < 2:
        return 0.0
    groups = _main_effect_columns(coded_design, features)
    columns = np.hstack(groups) if groups else np.empty((len(coded_design), 0))
    varying = columns[:, np.ptp(columns, axis=0) > 1e-12]
    if varying.shape[1] <= 1:
        return 100.0
    correlations = np.abs(np.corrcoef(varying, rowvar=False))
    np.fill_diagonal(correlations, 0.0)
    return max(0.0, 100.0 * (1.0 - float(np.nanmax(correlations))))


def _greedy_doptimal_indices(model_matrix, n_points):
    """Select a deterministic, without-replacement sequential D-optimal subset."""
    n_candidates, n_parameters = model_matrix.shape
    if n_points > n_candidates:
        raise ValueError(
            f"Only {n_candidates} distinct feasible candidate points are available "
            f"for the requested {n_points}-run D-Optimal design."
        )
    rank = np.linalg.matrix_rank(model_matrix)

    _, _, pivots = qr(model_matrix.T, pivoting=True, mode="economic")
    selected = [int(index) for index in pivots[:min(rank, n_points)]]
    selected_set = set(selected)
    information = model_matrix[selected].T @ model_matrix[selected]

    while len(selected) < n_points:
        inverse = np.linalg.pinv(information)
        leverages = np.einsum("ij,jk,ik->i", model_matrix, inverse, model_matrix)
        leverages[list(selected_set)] = -np.inf
        best = int(np.argmax(leverages))
        selected.append(best)
        selected_set.add(best)
        information += np.outer(model_matrix[best], model_matrix[best])

    # A few Fedorov-style exchange passes materially improve the sequential
    # start while retaining unique experimental conditions.
    for _ in range(3 if rank == n_parameters and n_points >= n_parameters else 0):
        improved = False
        for position, current_index in enumerate(list(selected)):
            available = [index for index in range(n_candidates) if index not in selected_set]
            available.append(current_index)
            rows = model_matrix[available]
            inverse = np.linalg.inv(information)
            current_row = model_matrix[current_index]
            current_leverage = float(current_row @ inverse @ current_row)
            candidate_leverages = np.einsum("ij,jk,ik->i", rows, inverse, rows)
            cross_leverages = rows @ inverse @ current_row
            # Matrix determinant lemma for replacing u with v in X'X:
            # det(M - uu' + vv') / det(M).
            determinant_ratios = (
                (1.0 - current_leverage) * (1.0 + candidate_leverages)
                + cross_leverages**2
            )
            candidate_position = int(np.argmax(determinant_ratios))
            candidate_index = available[candidate_position]
            if (
                candidate_index != current_index
                and determinant_ratios[candidate_position] > 1.0 + 1e-10
            ):
                selected_set.remove(current_index)
                selected_set.add(candidate_index)
                selected[position] = candidate_index
                information += np.outer(
                    model_matrix[candidate_index], model_matrix[candidate_index]
                ) - np.outer(current_row, current_row)
                improved = True
        if not improved:
            break
    return selected


def _quality_subset(design, n_points, features, model_spec, center_count=0):
    """Select informative unique conditions before spending any budget on repeats."""
    if n_points >= len(design):
        return design.copy()
    unique, repeats = _partition_repeats(design)
    if n_points >= len(unique):
        return np.vstack([unique, repeats[:n_points - len(unique)]])
    matrix = _build_model_matrix(unique, features, model_spec)
    return unique[_greedy_doptimal_indices(matrix, n_points)]


def _partition_repeats(design):
    """Stable partition of actual, decoded/re-encoded experimental conditions."""
    seen, unique, repeats = set(), [], []
    for row in design:
        key = tuple(row)
        (repeats if key in seen else unique).append(row)
        seen.add(key)
    width = design.shape[1]
    return (np.asarray(unique, dtype=float).reshape(-1, width),
            np.asarray(repeats, dtype=float).reshape(-1, width))


def _factorial_levels(features):
    return [codes if (codes := _feature_codes(feature)) is not None
            else np.array([-1.0, 1.0]) for feature in features]


def _factorial_candidates(features, n_runs):
    """Enumerate modest grids; sample large grids without losing their true size."""
    levels = _factorial_levels(features)
    total = math.prod(map(len, levels))
    if total <= 10000 or n_runs >= total:
        if total > 100000:
            raise ValueError(
                f"The complete Full Factorial design has {total:,} runs. "
                "Request a reduced run count or use fewer factor levels."
            )
        return np.asarray(list(itertools.product(*levels)), dtype=float), total
    count = min(total, max(2000, n_runs * 5))
    rng = random.Random(42)
    # Mixed-radix row indices avoid materialising an exponentially large grid.
    indices = {0, total - 1}
    while len(indices) < count:
        indices.add(rng.randrange(total))
    rows = []
    for index in sorted(indices):
        row = []
        for values in reversed(levels):
            index, digit = divmod(index, len(values))
            row.append(values[digit])
        rows.append(row[::-1])
    return np.asarray(rows, dtype=float), total


def _fractional_resolution(design):
    """Find the shortest defining word for a regular two-level design."""
    n_factors = design.shape[1]
    for word_length in range(3, n_factors + 1):
        for columns in itertools.combinations(range(n_factors), word_length):
            product = np.prod(design[:, columns], axis=1)
            if np.allclose(np.abs(product), 1.0) and np.allclose(product, product[0]):
                return word_length
    return None


def _doptimal_candidates(features, model_order, max_candidates=1500):
    levels = []
    for feature in features:
        codes = _feature_codes(feature)
        if codes is not None:
            levels.append(codes.tolist())
        else:
            count = 9 if model_order == "quadratic" else 7
            levels.append(np.linspace(-1.0, 1.0, count).tolist())

    total = math.prod(len(values) for values in levels)
    if total <= max_candidates:
        return np.asarray(list(itertools.product(*levels)), dtype=float)

    rng = np.random.default_rng(42)
    rows = {tuple(values[0] for values in levels), tuple(values[-1] for values in levels)}
    center = tuple(min(values, key=abs) for values in levels)
    rows.add(center)
    attempts = 0
    while len(rows) < max_candidates and attempts < max_candidates * 30:
        rows.add(tuple(values[int(rng.integers(len(values)))] for values in levels))
        attempts += 1
    return np.asarray(sorted(rows), dtype=float)


def _filter_unique_candidates(candidates, features, constraints, model_type):
    numerical = constraints.get("numerical", [])
    disallowed = constraints.get("disallowed", [])
    unique = {}
    for candidate in candidates:
        decoded = _decode_row(candidate, features, model_type)
        if (numerical or disallowed) and not check_constraints(
            decoded, numerical, disallowed, features
        ):
            continue
        key = tuple(decoded[feature["name"]] for feature in features)
        unique.setdefault(key, _encode_table([decoded], features)[0])
    return np.asarray(list(unique.values()), dtype=float).reshape(-1, len(features))


def _generate_lhs(features, n_runs, seed=42):
    numeric_indices = [
        index for index, feature in enumerate(features) if feature["type"] != "categorical"
    ]
    if not numeric_indices:
        raise ValueError("Latin Hypercube Sampling requires at least one numerical factor.")
    design = np.zeros((n_runs, len(features)), dtype=float)
    numeric = doe.lhs(
        len(numeric_indices), samples=n_runs,
        criterion="maximin" if n_runs > 1 else None, seed=seed
    ) * 2.0 - 1.0
    for numeric_column, design_column in enumerate(numeric_indices):
        feature = features[design_column]
        if feature["type"] in {"regular", "discrete"}:
            codes = _feature_codes(feature)
            strata = np.floor((numeric[:, numeric_column] + 1.0) * 0.5 * len(codes)).astype(int)
            strata = np.clip(strata, 0, len(codes) - 1)
            design[:, design_column] = codes[strata]
        else:
            design[:, design_column] = numeric[:, numeric_column]

    for column, feature in enumerate(features):
        if feature["type"] != "categorical":
            continue
        codes = _feature_codes(feature)
        balanced = np.resize(codes, n_runs).copy()
        random.Random(seed + column).shuffle(balanced)
        design[:, column] = balanced
    return design


def _unique_lhs(features, n_runs):
    design = _encode_table(
        [_decode_row(row, features, "lhs") for row in _generate_lhs(features, n_runs)],
        features,
    )
    unique, _ = _partition_repeats(design)
    if len(unique) == n_runs:
        return unique
    # A numerical continuous dimension normally makes every LHS row unique.
    # For finite grids, fill missing combinations before introducing repeats.
    if all(feature["type"] != "continuous" for feature in features):
        candidates, _ = _factorial_candidates(features, n_runs)
        np.random.default_rng(42).shuffle(candidates)
        unique, _ = _partition_repeats(np.vstack([unique, candidates]))
        return unique[:n_runs]
    raise ValueError("The entered numerical precision cannot distinguish the requested LHS runs.")


def _optimal_candidates(features, model_order, n_runs, constraints):
    """Expand candidates before claiming that unique feasible conditions are exhausted."""
    finite = all(feature["type"] != "continuous" for feature in features)
    limit = max(1500, n_runs * 5)
    while True:
        candidates = _doptimal_candidates(features, model_order, max_candidates=limit)
        feasible = _filter_unique_candidates(candidates, features, constraints, "doptimal")
        if len(feasible) >= n_runs:
            return feasible
        if finite:
            total = math.prod(len(_feature_codes(feature)) for feature in features)
            if len(candidates) == total:
                if not len(feasible):
                    raise ValueError("No candidate design points satisfy the specified constraints.")
                return feasible
            if limit >= 100000:
                raise ValueError("Too few feasible combinations found; relax constraints or request fewer runs.")
            limit = min(100000, max(limit * 2, n_runs * 10))
            continue
        # The continuous search space is not exhausted just because a grid is.
        rng = np.random.default_rng(42)
        rows = {tuple(row): row for row in feasible}
        for _ in range(max(10000, n_runs * 100)):
            row = np.array([
                rng.uniform(-1, 1) if feature["type"] == "continuous"
                else rng.choice(_feature_codes(feature)) for feature in features
            ])
            decoded = _decode_row(row, features, "doptimal")
            if check_constraints(decoded, constraints["numerical"], constraints["disallowed"], features):
                actual = _encode_table([decoded], features)[0]
                rows.setdefault(tuple(actual), actual)
            if len(rows) >= n_runs:
                return np.asarray(list(rows.values()))
        raise ValueError("Too few distinct feasible combinations found; relax constraints or request fewer runs.")


CONSTRAINED_CANDIDATE_LIMIT = 100000
CONSTRAINED_LHS_BATCH_LIMIT = 40


def _filter_actual_rows(design, features, constraints, model_type):
    """Check executable (decoded and rounded) conditions; retain parent repeats."""
    actual = [_decode_row(row, features, model_type) for row in design]
    keep = [check_constraints(row, constraints.get("numerical", []),
                              constraints.get("disallowed", []), features) for row in actual]
    encoded = _encode_table(actual, features)
    return encoded[np.asarray(keep, dtype=bool)], len(keep) - sum(keep)


def _feasible_factorial_pool(features, n_runs, constraints):
    """Enumerate a bounded finite grid, or expand a reproducible sampled pool."""
    levels = _factorial_levels(features)
    total = math.prod(map(len, levels))
    if total <= CONSTRAINED_CANDIDATE_LIMIT:
        candidates = np.asarray(list(itertools.product(*levels)), dtype=float)
        feasible, excluded = _filter_actual_rows(candidates, features, constraints, "fullfact")
        return feasible, total, excluded, True
    rng = random.Random(42)
    seen, feasible = set(), []
    excluded = 0
    target = max(1500, n_runs * 5)
    while len(seen) < CONSTRAINED_CANDIDATE_LIMIT and len(feasible) < target:
        index = rng.randrange(total)
        if index in seen:
            continue
        seen.add(index)
        row = []
        for values in reversed(levels):
            index, digit = divmod(index, len(values))
            row.append(values[digit])
        candidate, rejected = _filter_actual_rows([row[::-1]], features, constraints, "fullfact")
        excluded += rejected
        if len(candidate):
            feasible.append(candidate[0])
    return np.asarray(feasible, dtype=float).reshape(-1, len(features)), len(seen), excluded, False


def _space_filling_subset(candidates, n_runs, features):
    """Deterministic farthest-point selection; categories use mismatch distance."""
    unique, _ = _partition_repeats(candidates)
    if len(unique) <= n_runs:
        return unique
    distances = np.full(len(unique), np.inf)
    selected = []
    next_index = 0
    for _ in range(n_runs):
        selected.append(next_index)
        delta = np.zeros(len(unique))
        for column, feature in enumerate(features):
            if feature["type"] == "categorical":
                delta += unique[:, column] != unique[next_index, column]
            else:
                delta += ((unique[:, column] - unique[next_index, column]) / 2.0) ** 2
        distances = np.minimum(distances, delta)
        distances[selected] = -1.0
        next_index = int(np.argmax(distances))
    return unique[selected]


def _constrained_lhs(features, n_runs, constraints):
    if not any(feature["type"] != "categorical" for feature in features):
        raise ValueError("Latin Hypercube Sampling requires at least one numerical factor.")
    if all(feature["type"] != "continuous" for feature in features):
        pool, tested, excluded, complete = _feasible_factorial_pool(features, n_runs, constraints)
        initial, rejected = _filter_actual_rows(
            _generate_lhs(features, min(n_runs, 512)), features, constraints, "lhs"
        )
        tested += min(n_runs, 512)
        excluded += rejected
        pool = np.vstack([initial, pool])
    else:
        pool = np.empty((0, len(features)))
        tested = excluded = 0
        complete = False
        batch_size = min(512, max(32, n_runs * 2))
        for batch in range(CONSTRAINED_LHS_BATCH_LIMIT):
            candidates = _generate_lhs(features, batch_size, seed=42 + batch)
            feasible, rejected = _filter_actual_rows(candidates, features, constraints, "lhs")
            tested += len(candidates)
            excluded += rejected
            pool, _ = _partition_repeats(np.vstack([pool, feasible]))
            if len(pool) >= n_runs:
                break
    return _space_filling_subset(pool, n_runs, features), tested, excluded, complete


def _validate_design_compatibility(model_type, features, constraints):
    categorical = [feature for feature in features if feature["type"] == "categorical"]

    if model_type in {"bbdesign", "ccdesign"} and categorical:
        raise ValueError(
            f"{MODEL_LABELS[model_type]} supports numerical factors only. "
            "Use D-Optimal for mixed numerical/categorical experiments."
        )
    if model_type in {"fracfact", "def_screening"}:
        invalid = [feature["name"] for feature in categorical if len(feature["choices"]) != 2]
        if invalid:
            raise ValueError(
                f"{MODEL_LABELS[model_type]} supports only two-level categorical factors; "
                f"check: {', '.join(invalid)}."
            )


def generate_doe(features_config, tweaks):
    """Generate a DoE, exhausting unique conditions before proposing repeats."""
    model_type = str(tweaks.get("model", "bbdesign"))
    if model_type not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported DoE model '{model_type}'. Select one of: "
            + ", ".join(MODEL_LABELS[model] for model in SUPPORTED_MODELS)
            + "."
        )

    try:
        requested_runs = int(float(tweaks.get("max_runs", 50)))
    except (TypeError, ValueError) as exc:
        raise ValueError("The experiment count must be a positive integer.") from exc
    if requested_runs < 1:
        raise ValueError("The experiment count must be at least 1.")

    features = _parse_features(features_config)
    n_factors = len(features)
    if n_factors < 2:
        raise ValueError(f"DoE requires at least 2 features. Found {n_factors} valid features.")
    if model_type == "bbdesign" and n_factors < 3:
        raise ValueError(f"Box-Behnken design requires at least 3 factors. Found {n_factors}.")

    constraints = dict(tweaks.get("constraints") or {})
    constraints["numerical"] = constraints.get("numerical") or []
    constraints["disallowed"] = constraints.get("disallowed") or []
    _validate_design_compatibility(model_type, features, constraints)
    filtered_design = model_type != "doptimal" and bool(
        constraints["numerical"] or constraints["disallowed"]
    )
    candidates_tested = excluded_runs = 0
    candidate_pool_complete = True
    parent_design = None
    eligible_runs = None

    exact_only = model_type in {"lhs", "doptimal"}
    use_exact_runs = exact_only or _as_bool(tweaks.get("use_exact_runs", False))
    natural_runs = requested_runs
    center_count = 0
    added_replicates = 0
    design_status = "Natural"
    design_warning = None
    fractional_resolution = None
    factorial_sampled = False

    if model_type == "bbdesign":
        design, center_count = _move_centers_to_end(doe.bbdesign(n_factors))
        model_spec = "response_surface"
    elif model_type == "ccdesign":
        # Face-centred CCD keeps every proposed experiment within the user's bounds.
        design, center_count = _move_centers_to_end(doe.ccdesign(n_factors, face="ccf"))
        model_spec = "response_surface"
    elif model_type == "fracfact":
        try:
            requested_resolution = int(tweaks.get("fracfact_resolution", 4))
        except (TypeError, ValueError) as exc:
            raise ValueError("Fractional Factorial resolution must be III, IV, or V.") from exc
        if requested_resolution not in {3, 4, 5}:
            raise ValueError("Fractional Factorial resolution must be III, IV, or V.")
        design = doe.fracfact_by_res(n_factors, requested_resolution).astype(float)
        fractional_resolution = _fractional_resolution(design)
        model_spec = "linear"
    elif model_type == "fullfact":
        if filtered_design:
            design, candidates_tested, excluded_runs, candidate_pool_complete = _feasible_factorial_pool(
                features, requested_runs, constraints
            )
            natural_runs = math.prod(map(len, _factorial_levels(features)))
        else:
            design, natural_runs = _factorial_candidates(features, requested_runs)
        factorial_sampled = len(design) < natural_runs
        model_spec = "factorial"
    elif model_type == "def_screening":
        numeric_indices = [
            index for index, feature in enumerate(features) if feature["type"] != "categorical"
        ]
        categorical_indices = [
            index for index, feature in enumerate(features) if feature["type"] == "categorical"
        ]
        grouped = doe.definitive_screening_design(
            len(numeric_indices), n_cat=len(categorical_indices), min_13=True
        ).astype(float)
        design = np.zeros((len(grouped), n_factors), dtype=float)
        for source, target in enumerate(numeric_indices + categorical_indices):
            design[:, target] = grouped[:, source]
        model_spec = "dsd"
    elif model_type == "lhs":
        if filtered_design:
            design, candidates_tested, excluded_runs, candidate_pool_complete = _constrained_lhs(
                features, requested_runs, constraints
            )
        else:
            design = _unique_lhs(features, requested_runs)
        model_spec = "linear"
    else:  # doptimal
        model_order = str(tweaks.get("dopt_model_order", "quadratic"))
        if model_order not in {"linear", "interaction", "quadratic"}:
            raise ValueError("D-Optimal model order must be linear, interaction, or quadratic.")
        candidates = _optimal_candidates(features, model_order, requested_runs, constraints)
        candidate_matrix = _build_model_matrix(candidates, features, model_order)
        selected = _greedy_doptimal_indices(candidate_matrix, min(requested_runs, len(candidates)))
        design = candidates[selected]
        model_spec = model_order

    if model_type != "fullfact":
        natural_runs = len(design)
    if filtered_design:
        if model_type not in {"lhs", "fullfact"}:
            parent_design = _encode_table([_decode_row(row, features, model_type) for row in design], features)
            candidates_tested = len(design)
            design, excluded_runs = _filter_actual_rows(design, features, constraints, model_type)
        if not len(design):
            if model_type not in {"lhs", "fullfact"}:
                raise ValueError("No feasible conditions remain in the selected parent design. "
                                 "This does not establish that the experimental region is empty.")
            if candidate_pool_complete:
                raise ValueError("No feasible conditions exist in the fully enumerated candidate grid.")
            raise ValueError("No feasible conditions found within the bounded candidate search; "
                             "the experimental region has not been proven empty.")
        eligible_runs = len(design)
    # Compare actual conditions: snapping numerical levels can create duplicates.
    design = _encode_table([_decode_row(row, features, model_type) for row in design], features)
    unique, repeats = _partition_repeats(design)
    design = np.vstack([unique, repeats])
    selection_runs = eligible_runs if filtered_design else natural_runs
    allow_replicates = not filtered_design or (
        model_type != "lhs" and candidate_pool_complete
    )
    if requested_runs < selection_runs:
        design = _quality_subset(design, requested_runs, features, model_spec)
        design_status = "Reduced"
        design_warning = (
            f"The requested {requested_runs} runs are below the natural {natural_runs}-run "
            f"{MODEL_LABELS[model_type]} design. A quality-optimized reduced subset is shown; "
            "it is not the complete classical design. Unique combinations take priority over repeats."
        )
        if factorial_sampled:
            design_warning += " Selection used a sampled candidate pool from the full factorial grid."
    elif use_exact_runs and requested_runs > selection_runs and allow_replicates:
        added_replicates = requested_runs - selection_runs
        repeated = np.asarray([unique[index % len(unique)] for index in range(added_replicates)])
        design = np.vstack([design, repeated])
        design_status = "Replicated"
        size_label = "unique feasible combinations" if exact_only else "natural design runs"
        design_warning = (
            f"The exact request exceeds the {natural_runs} {size_label}. "
            f"{added_replicates} repeat run(s) were added after all unique combinations."
        )

    # Retain the familiar centre-at-end order within the unique RSM conditions.
    unique, repeats = _partition_repeats(design)
    if model_type in {"bbdesign", "ccdesign"}:
        unique, _ = _move_centers_to_end(unique)
    design = np.vstack([unique, repeats])

    suggested_table = [_decode_row(row, features, model_type) for row in design]
    if filtered_design and not all(check_constraints(row, constraints["numerical"],
                                                     constraints["disallowed"], features)
                                   for row in suggested_table):
        raise ValueError("Final design failed constraint validation after decoding.")
    actual_coded = _encode_table(suggested_table, features)

    if model_type in {"bbdesign", "ccdesign"}:
        # Added exact-N repetitions are reported separately, even when one of
        # those repeated rows happens to be a center point.
        base_length = len(actual_coded) - added_replicates
        center_count = int(
            np.all(np.abs(actual_coded[:base_length]) < 1e-9, axis=1).sum()
        )
    else:
        center_count = 0

    if model_type == "lhs":
        orthogonality = "N/A"
        efficiency = "N/A"
        resolution = "N/A"
        curvature = "N/A"
        rank = "N/A"
        n_parameters = "N/A"
    else:
        metric_matrix = _build_model_matrix(actual_coded, features, model_spec)
        rank = int(np.linalg.matrix_rank(metric_matrix))
        n_parameters = int(metric_matrix.shape[1])
        orthogonality = round(_orthogonality(actual_coded, features), 1)
        efficiency = round(_d_efficiency(metric_matrix), 1)

        if model_type == "fullfact":
            resolution = "Full factorial" if design_status != "Reduced" else "N/A (Reduced design)"
            numeric = [feature for feature in features if feature["type"] != "categorical"]
            curved = [feature for feature in numeric if len(feature.get("levels", [])) >= 3]
            if not numeric:
                curvature = "N/A"
            elif not curved:
                curvature = "None (Two-level numerical factors)"
            else:
                curvature = "Excellent" if len(curved) == len(numeric) and rank == n_parameters else "Partial"
        elif model_type == "fracfact":
            if fractional_resolution is None:
                resolution = "Full factorial"
            elif fractional_resolution >= 5:
                resolution = f"{fractional_resolution} (V+)"
            elif fractional_resolution == 4:
                resolution = "4 (IV)"
            else:
                resolution = "3 (III)"
            curvature = "None (Linear only)"
            if design_status == "Reduced":
                resolution = "N/A (Reduced design)"
        else:
            resolution = "N/A"
            if model_type in {"bbdesign", "ccdesign", "def_screening"}:
                curvature = "Excellent" if rank == n_parameters else "Partial"
            elif model_type == "doptimal" and model_spec == "quadratic":
                curvature = "Excellent" if rank == n_parameters else "Partial"
            else:
                curvature = "None (Linear only)"

        if rank < n_parameters:
            rank_warning = (
                f" The returned model matrix has rank {rank} for {n_parameters} terms, so not "
                "all intended effects can be estimated independently."
            )
            design_warning = (design_warning or "Design quality is limited.") + rank_warning

    constraint_metadata = {}
    if filtered_design:
        shortfall = max(0, requested_runs - len(suggested_table)) if use_exact_runs else 0
        design_status = "Constrained LHS-derived" if model_type == "lhs" else "Constraint-filtered"
        if requested_runs < eligible_runs:
            design_status += " reduced"
        if added_replicates:
            design_status += " replicated"
        resolution = "N/A (constraint-modified design)"
        if model_type != "lhs":
            # Describe the actual intended model's rank, not inherited RSM quality.
            has_squares = model_spec in {"response_surface", "dsd", "quadratic"} or (
                model_spec == "factorial" and any(len(f.get("levels", [])) >= 3
                                                 for f in features if f["type"] != "categorical")
            )
            curvature = ("Full model rank" if rank == n_parameters else "Rank deficient") if has_squares else "None (no quadratic terms)"
        missing_levels = {}
        if model_type != "lhs":
            for column, feature in enumerate(features):
                if parent_design is not None:
                    before = {str(_decode_row(row, features, model_type)[feature["name"]]) for row in parent_design}
                else:
                    levels = _factorial_levels(features)
                    before = set()
                    for value in levels[column]:
                        row = np.array([values[0] for values in levels], dtype=float)
                        row[column] = value
                        before.add(str(_decode_row(row, features, model_type)[feature["name"]]))
                after = {str(row[feature["name"]]) for row in suggested_table}
                if before - after:
                    missing_levels[feature["name"]] = sorted(before - after)
        removed_centers = 0
        if parent_design is not None and model_type in {"bbdesign", "ccdesign"}:
            feasible_parent, _ = _filter_actual_rows(parent_design, features, constraints, model_type)
            removed_centers = int(np.all(np.abs(parent_design) < 1e-9, axis=1).sum()
                                  - np.all(np.abs(feasible_parent) < 1e-9, axis=1).sum())
        messages = [f"{excluded_runs} of {candidates_tested} examined candidate rows excluded by constraints. "
                    "Original design properties are not assumed for this modified design."]
        if model_type == "lhs":
            messages.append("Feasible points are selected by farthest-point spacing; strict Latin hypercube stratification is not retained.")
        if added_replicates:
            messages.append(f"{added_replicates} feasible repeats added from the surviving candidate conditions; "
                            "this does not imply exhaustion of the full experimental region.")
        if not candidate_pool_complete:
            messages.append("Candidate search was bounded; the full experimental region was not enumerated.")
        if shortfall:
            messages.append(f"Requested {requested_runs} runs; returned {len(suggested_table)} distinct feasible conditions "
                            f"with a shortfall of {shortfall}. No repeats were added to mask the shortfall.")
        if missing_levels:
            messages.append("Some parent-design factor levels are absent from the returned design.")
        if removed_centers:
            messages.append(f"{removed_centers} parent centre-point rows excluded by constraints.")
        if model_type != "lhs" and rank < n_parameters:
            messages.append(f"Model rank {rank} for {n_parameters} terms: not all intended effects can be estimated independently.")
        design_warning = " ".join(messages)
        constraint_metadata = {
            "constraint_filtered": True,
            "parent_design_label": MODEL_LABELS[model_type],
            "parent_runs": natural_runs if model_type != "lhs" else None,
            "candidates_tested": candidates_tested,
            "excluded_runs": excluded_runs,
            "eligible_runs": eligible_runs,
            "candidate_pool_complete": candidate_pool_complete,
            "shortfall_runs": shortfall,
            "missing_levels": missing_levels,
            "constraint_removed_center_runs": removed_centers,
            "model_estimable": rank == n_parameters if model_type != "lhs" else None,
        }

    metrics = {
        "orthogonality": orthogonality,
        "efficiency": efficiency,
        "resolution": resolution,
        "curvature": curvature,
        "runs": len(suggested_table),
        "model": model_type,
        "model_label": (f"{design_status}: {MODEL_LABELS[model_type]}" if filtered_design else MODEL_LABELS[model_type]),
        "design_status": design_status,
        "natural_runs": natural_runs,
        "requested_runs": requested_runs,
        "design_warning": design_warning,
        "reduction_warning": design_status == "Reduced",
        # Retained for API compatibility; it now means replicated exact-N output.
        "oversampling_warning": added_replicates > 0,
        "added_replicates": added_replicates,
        "center_replicates": center_count,
        "rank": rank,
        "parameters": n_parameters,
        "unique_runs": len(unique),
        "repeat_runs": len(repeats),
        **constraint_metadata,
    }
    return suggested_table, metrics
