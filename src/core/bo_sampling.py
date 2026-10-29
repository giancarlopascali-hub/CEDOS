"""Deterministic representative-row selection for bounded exact-GP BO."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import time
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BORepresentativeSelection:
    source_rows: int
    selected_rows: int
    selected_source_indices: list[Any]
    feature_coverage_score: float
    objective_distribution_score: float
    tradeoff_anchor_retention_score: float
    objective_validity_score: float
    fidelity_score: float
    target_fidelity_score: float
    feature_distance_mean: float
    feature_distance_p95: float
    protected_anchor_count: int
    duplicate_feature_rows: int
    deterministic_seed: int
    selection_attempts: int = 1
    fidelity_target_met: bool = True
    search_time_limit_reached: bool = False
    method: str = "extremes-and-farthest-point-v2-bounded-search"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _json_index(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _normalized_feature_arrays(
    dataframe: pd.DataFrame,
    features_config: Sequence[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    numeric_columns = []
    categorical_columns = []
    for feature in features_config:
        name = feature["name"]
        if feature.get("type", "continuous") == "categorical":
            values = dataframe[name].fillna("None").astype(str)
            categories = {value: index for index, value in enumerate(sorted(values.unique()))}
            categorical_columns.append(
                values.map(categories).to_numpy(dtype=np.int64, copy=False)
            )
            continue

        values = pd.to_numeric(dataframe[name], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(values)
        replacement = float(np.median(values[finite])) if finite.any() else 0.0
        values = np.where(finite, values, replacement)
        minimum = float(values.min()) if values.size else 0.0
        maximum = float(values.max()) if values.size else minimum
        scale = maximum - minimum
        numeric_columns.append(
            np.zeros_like(values) if scale <= 0 else (values - minimum) / scale
        )

    numeric = (
        np.stack(numeric_columns, axis=1)
        if numeric_columns
        else np.empty((len(dataframe), 0), dtype=float)
    )
    categorical = (
        np.stack(categorical_columns, axis=1)
        if categorical_columns
        else np.empty((len(dataframe), 0), dtype=np.int64)
    )
    return numeric, categorical


def _distance_squared_to_row(
    numeric: np.ndarray,
    categorical: np.ndarray,
    row_index: int,
    feature_count: int,
) -> np.ndarray:
    distance = np.zeros(numeric.shape[0], dtype=float)
    if numeric.shape[1]:
        difference = numeric - numeric[row_index]
        distance += np.sum(difference * difference, axis=1)
    if categorical.shape[1]:
        distance += np.sum(categorical != categorical[row_index], axis=1)
    return distance / max(1, feature_count)


def _benefit_objective_values(
    dataframe: pd.DataFrame,
    objectives_config: Sequence[dict[str, Any]],
) -> np.ndarray:
    columns = []
    for objective in objectives_config:
        values = pd.to_numeric(
            dataframe[objective["name"]], errors="coerce"
        ).to_numpy(dtype=float)
        objective_type = objective.get("type", "maximize")
        if objective_type == "minimize":
            values = -values
        elif objective_type == "target":
            target = float(objective.get("target", 0))
            values = -np.abs(values - target)
        columns.append(values)
    return np.stack(columns, axis=1)


def _objective_quality(
    full: pd.DataFrame,
    selected: pd.DataFrame,
    objectives_config: Sequence[dict[str, Any]],
) -> tuple[float, float]:
    quantiles = np.linspace(0.0, 1.0, 11)
    distribution_scores = []
    validity_scores = []
    for objective in objectives_config:
        name = objective["name"]
        full_values = pd.to_numeric(full[name], errors="coerce").to_numpy(dtype=float)
        selected_values = pd.to_numeric(
            selected[name], errors="coerce"
        ).to_numpy(dtype=float)
        full_finite = full_values[np.isfinite(full_values)]
        selected_finite = selected_values[np.isfinite(selected_values)]
        validity_scores.append(
            1.0
            - abs(
                (full_finite.size / max(1, full_values.size))
                - (selected_finite.size / max(1, selected_values.size))
            )
        )
        if not full_finite.size or not selected_finite.size:
            distribution_scores.append(0.0)
            continue
        scale = float(np.ptp(full_finite))
        if scale <= 0:
            distribution_scores.append(1.0)
            continue
        error = float(
            np.mean(
                np.abs(
                    np.quantile(full_finite, quantiles)
                    - np.quantile(selected_finite, quantiles)
                )
            )
            / scale
        )
        distribution_scores.append(max(0.0, 1.0 - min(1.0, error)))
    return (
        100.0 * float(np.mean(distribution_scores or [1.0])),
        100.0 * float(np.mean(validity_scores or [1.0])),
    )


def _select_representative_bo_rows_once(
    dataframe: pd.DataFrame,
    features_config: Sequence[dict[str, Any]],
    objectives_config: Sequence[dict[str, Any]],
    *,
    max_rows: int,
    target_fidelity_score: float,
    random_seed: int = 1729,
    check_runtime: Callable[[], None] | None = None,
) -> tuple[pd.DataFrame, BORepresentativeSelection]:
    """Build one real-row candidate preserving coverage, tails, and trade-offs.

    Important feature and objective anchors are retained first. Remaining slots
    are filled by deterministic farthest-point sampling in mixed feature space.
    No synthetic or interpolated observations are introduced.
    """

    source_rows = int(len(dataframe))
    limit = max(1, min(int(max_rows), source_rows))
    if source_rows <= limit:
        source_indices = [_json_index(value) for value in dataframe.index.tolist()]
        selection = BORepresentativeSelection(
            source_rows=source_rows,
            selected_rows=source_rows,
            selected_source_indices=source_indices,
            feature_coverage_score=100.0,
            objective_distribution_score=100.0,
            tradeoff_anchor_retention_score=100.0,
            objective_validity_score=100.0,
            fidelity_score=100.0,
            target_fidelity_score=float(target_fidelity_score),
            feature_distance_mean=0.0,
            feature_distance_p95=0.0,
            protected_anchor_count=source_rows,
            duplicate_feature_rows=int(
                dataframe[[feature["name"] for feature in features_config]]
                .duplicated()
                .sum()
            ),
            deterministic_seed=int(random_seed),
            fidelity_target_met=100.0 >= float(target_fidelity_score),
        )
        return dataframe.copy(), selection

    if check_runtime is not None:
        check_runtime()
    numeric, categorical = _normalized_feature_arrays(dataframe, features_config)
    feature_count = max(1, len(features_config))
    selected_positions: list[int] = []
    selected_set: set[int] = set()

    def add_anchor(position: int) -> None:
        parsed = int(position)
        if 0 <= parsed < source_rows and parsed not in selected_set and len(selected_set) < limit:
            selected_positions.append(parsed)
            selected_set.add(parsed)

    # Preserve the observed boundary of every numerical feature.
    for feature in features_config:
        name = feature["name"]
        if feature.get("type", "continuous") == "categorical":
            continue
        values = pd.to_numeric(dataframe[name], errors="coerce").to_numpy(dtype=float)
        finite_positions = np.flatnonzero(np.isfinite(values))
        if finite_positions.size:
            finite_values = values[finite_positions]
            add_anchor(finite_positions[int(np.argmin(finite_values))])
            add_anchor(finite_positions[int(np.argmax(finite_values))])

    # Keep examples of categorical levels, prioritizing rare levels. The shared
    # budget prevents a high-cardinality identifier-like column consuming the set.
    category_budget = min(64, max(4, limit // 4))
    category_added = 0
    for feature in features_config:
        if feature.get("type") != "categorical" or category_added >= category_budget:
            continue
        values = dataframe[feature["name"]].fillna("None").astype(str)
        counts = values.value_counts(dropna=False, ascending=True)
        for level in counts.index:
            positions = np.flatnonzero(values.to_numpy() == level)
            if positions.size:
                before = len(selected_set)
                add_anchor(int(positions[0]))
                category_added += int(len(selected_set) > before)
            if category_added >= category_budget or len(selected_set) >= limit:
                break

    # Objective quantiles retain tails and valid observations even with missing
    # values. They also guarantee that missingness does not erase a sparse model.
    for objective in objectives_config:
        values = pd.to_numeric(
            dataframe[objective["name"]], errors="coerce"
        ).to_numpy(dtype=float)
        finite_positions = np.flatnonzero(np.isfinite(values))
        if not finite_positions.size:
            continue
        ordered = finite_positions[np.argsort(values[finite_positions], kind="stable")]
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            add_anchor(ordered[int(round(fraction * (len(ordered) - 1)))])

    # Preserve common objective-validity patterns so per-objective missing values
    # remain represented in approximately the same proportions.
    validity = np.stack(
        [
            np.isfinite(
                pd.to_numeric(dataframe[objective["name"]], errors="coerce")
                .to_numpy(dtype=float)
            )
            for objective in objectives_config
        ],
        axis=1,
    )
    patterns, pattern_counts = np.unique(validity, axis=0, return_counts=True)
    pattern_order = np.argsort(-pattern_counts, kind="stable")
    for pattern_index in pattern_order[: min(16, len(pattern_order))]:
        positions = np.flatnonzero(np.all(validity == patterns[pattern_index], axis=1))
        if positions.size:
            add_anchor(int(positions[0]))

    # Approximate the Pareto boundary with deterministic scalarization anchors.
    # The feature-space fill below remains dominant and avoids training only on
    # apparently good historical outcomes.
    benefit_values = _benefit_objective_values(dataframe, objectives_config)
    complete_positions = np.flatnonzero(np.isfinite(benefit_values).all(axis=1))
    tradeoff_anchors: set[int] = set()
    if complete_positions.size:
        complete = benefit_values[complete_positions]
        minima = complete.min(axis=0)
        spans = complete.max(axis=0) - minima
        normalized = (complete - minima) / np.where(spans > 0, spans, 1.0)
        directions = [np.eye(len(objectives_config))[index] for index in range(len(objectives_config))]
        rng = np.random.default_rng(int(random_seed))
        directions.extend(
            rng.dirichlet(
                np.ones(len(objectives_config)),
                size=min(24, max(4, limit // 10)),
            )
        )
        for direction in directions:
            position = int(complete_positions[int(np.argmax(normalized @ direction))])
            tradeoff_anchors.add(position)
            add_anchor(position)

    protected_positions = set(selected_set)
    if not selected_positions:
        # Defensive fallback for an empty configuration; normal BO validation
        # requires at least one feature and objective.
        add_anchor(0)

    minimum_distance_squared = np.full(source_rows, np.inf, dtype=float)
    for position in selected_positions:
        minimum_distance_squared = np.minimum(
            minimum_distance_squared,
            _distance_squared_to_row(
                numeric, categorical, position, feature_count
            ),
        )
    minimum_distance_squared[list(selected_set)] = -1.0

    while len(selected_positions) < limit:
        if check_runtime is not None and len(selected_positions) % 16 == 0:
            check_runtime()
        next_position = int(np.argmax(minimum_distance_squared))
        if minimum_distance_squared[next_position] < 0:
            break
        add_anchor(next_position)
        minimum_distance_squared = np.minimum(
            minimum_distance_squared,
            _distance_squared_to_row(
                numeric, categorical, next_position, feature_count
            ),
        )
        minimum_distance_squared[list(selected_set)] = -1.0

    ordered_positions = sorted(selected_set)
    selected = dataframe.iloc[ordered_positions].copy()
    coverage_distances = np.sqrt(np.maximum(minimum_distance_squared, 0.0))
    coverage_distances[ordered_positions] = 0.0
    mean_distance = float(np.mean(coverage_distances))
    p95_distance = float(np.quantile(coverage_distances, 0.95))
    feature_score = 100.0 * max(
        0.0,
        1.0 - min(1.0, 0.5 * (0.65 * mean_distance + 0.35 * p95_distance)),
    )
    distribution_score, validity_score = _objective_quality(
        dataframe, selected, objectives_config
    )
    retained_tradeoffs = len(tradeoff_anchors.intersection(selected_set))
    tradeoff_score = 100.0 * (
        retained_tradeoffs / len(tradeoff_anchors) if tradeoff_anchors else 1.0
    )
    fidelity = (
        0.50 * feature_score
        + 0.25 * distribution_score
        + 0.20 * tradeoff_score
        + 0.05 * validity_score
    )
    source_indices = [
        _json_index(dataframe.index[position]) for position in ordered_positions
    ]
    duplicate_rows = int(
        dataframe[[feature["name"] for feature in features_config]]
        .duplicated()
        .sum()
    )
    rounded_fidelity = round(fidelity, 2)
    diagnostics = BORepresentativeSelection(
        source_rows=source_rows,
        selected_rows=len(selected),
        selected_source_indices=source_indices,
        feature_coverage_score=round(feature_score, 2),
        objective_distribution_score=round(distribution_score, 2),
        tradeoff_anchor_retention_score=round(tradeoff_score, 2),
        objective_validity_score=round(validity_score, 2),
        fidelity_score=rounded_fidelity,
        target_fidelity_score=float(target_fidelity_score),
        feature_distance_mean=round(mean_distance, 4),
        feature_distance_p95=round(p95_distance, 4),
        protected_anchor_count=len(protected_positions),
        duplicate_feature_rows=duplicate_rows,
        deterministic_seed=int(random_seed),
        fidelity_target_met=(
            rounded_fidelity >= float(target_fidelity_score)
        ),
    )
    return selected, diagnostics


def select_representative_bo_rows(
    dataframe: pd.DataFrame,
    features_config: Sequence[dict[str, Any]],
    objectives_config: Sequence[dict[str, Any]],
    *,
    max_rows: int,
    target_fidelity_score: float,
    random_seed: int = 1729,
    maximum_attempts: int = 3,
    search_time_limit_seconds: float = 12.0,
    check_runtime: Callable[[], None] | None = None,
) -> tuple[pd.DataFrame, BORepresentativeSelection]:
    """Return the best representative subset found within a bounded search.

    The target is aspirational rather than a hard acceptance gate. Candidate
    subsets use reproducible seed variations, the search stops as soon as the
    target is reached, and the highest-fidelity candidate is returned when the
    bounded search cannot reach it.
    """

    parsed_attempts = max(1, int(maximum_attempts))
    parsed_time_limit = max(0.1, float(search_time_limit_seconds))
    started = time.monotonic()
    previous_attempt_seconds = 0.0
    attempted = 0
    time_limit_reached = False
    best_selected: pd.DataFrame | None = None
    best_diagnostic: BORepresentativeSelection | None = None

    for attempt_index in range(parsed_attempts):
        if attempt_index:
            elapsed = time.monotonic() - started
            estimated_next_attempt = max(0.05, previous_attempt_seconds)
            if elapsed + estimated_next_attempt > parsed_time_limit:
                time_limit_reached = True
                break
        if check_runtime is not None:
            check_runtime()

        attempt_started = time.monotonic()
        candidate_seed = (
            int(random_seed) + attempt_index * 104_729
        ) % (2**63 - 1)
        selected, diagnostic = _select_representative_bo_rows_once(
            dataframe,
            features_config,
            objectives_config,
            max_rows=max_rows,
            target_fidelity_score=target_fidelity_score,
            random_seed=candidate_seed,
            check_runtime=check_runtime,
        )
        previous_attempt_seconds = time.monotonic() - attempt_started
        attempted += 1

        if (
            best_diagnostic is None
            or diagnostic.fidelity_score > best_diagnostic.fidelity_score
        ):
            best_selected = selected
            best_diagnostic = diagnostic
        if diagnostic.fidelity_target_met:
            break

    if best_selected is None or best_diagnostic is None:  # pragma: no cover
        raise RuntimeError("Representative-row selection produced no candidate subset.")

    final_diagnostic = replace(
        best_diagnostic,
        selection_attempts=attempted,
        fidelity_target_met=(
            best_diagnostic.fidelity_score >= float(target_fidelity_score)
        ),
        search_time_limit_reached=time_limit_reached,
    )
    return best_selected, final_diagnostic
