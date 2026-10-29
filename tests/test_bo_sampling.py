import numpy as np
import pandas as pd

from src.core.bo_sampling import select_representative_bo_rows


def _large_mixed_dataset(row_count=2_000):
    rng = np.random.default_rng(23)
    feature_names = [f"x{index}" for index in range(8)]
    values = rng.uniform(0.0, 1.0, size=(row_count, len(feature_names)))
    data = pd.DataFrame(values, columns=feature_names)
    data["catalyst"] = np.where(values[:, 0] < 0.1, "rare", "common")
    data["yield"] = np.sin(values[:, 0] * 3.0) + values[:, 1]
    data["cost"] = 2.0 - values[:, 2] + 0.1 * values[:, 3]
    data["stability"] = values[:, 4] * values[:, 5]
    data.loc[::11, "stability"] = np.nan
    features = [
        {"name": name, "type": "continuous"} for name in feature_names
    ] + [{"name": "catalyst", "type": "categorical"}]
    objectives = [
        {"name": "yield", "type": "maximize"},
        {"name": "cost", "type": "minimize"},
        {"name": "stability", "type": "maximize"},
    ]
    return data, features, objectives


def test_representative_rows_are_deterministic_real_rows_with_extremes():
    data, features, objectives = _large_mixed_dataset()
    first, first_diagnostic = select_representative_bo_rows(
        data,
        features,
        objectives,
        max_rows=180,
        target_fidelity_score=90,
        random_seed=19,
    )
    second, second_diagnostic = select_representative_bo_rows(
        data,
        features,
        objectives,
        max_rows=180,
        target_fidelity_score=90,
        random_seed=19,
    )

    assert first.index.tolist() == second.index.tolist()
    assert len(first) == 180
    assert set(first.index) <= set(data.index)
    assert first_diagnostic == second_diagnostic
    assert first_diagnostic.fidelity_score >= 90
    assert first_diagnostic.fidelity_target_met is True
    assert first_diagnostic.selection_attempts == 1
    assert "rare" in set(first["catalyst"])
    for feature in features[:-1]:
        assert first[feature["name"]].min() == data[feature["name"]].min()
        assert first[feature["name"]].max() == data[feature["name"]].max()


def test_representative_rows_preserve_missing_objective_support():
    data, features, objectives = _large_mixed_dataset(600)
    selected, diagnostic = select_representative_bo_rows(
        data,
        features,
        objectives,
        max_rows=100,
        target_fidelity_score=90,
        random_seed=7,
    )

    assert selected["stability"].notna().sum() > 0
    assert selected["stability"].isna().sum() > 0
    assert diagnostic.objective_validity_score >= 95
    assert diagnostic.tradeoff_anchor_retention_score == 100


def test_no_reduction_reports_perfect_fidelity():
    data, features, objectives = _large_mixed_dataset(30)
    selected, diagnostic = select_representative_bo_rows(
        data,
        features,
        objectives,
        max_rows=50,
        target_fidelity_score=90,
    )

    assert selected.equals(data)
    assert diagnostic.fidelity_score == 100
    assert diagnostic.fidelity_target_met is True
    assert diagnostic.selection_attempts == 1
    assert diagnostic.selected_rows == len(data)


def test_bounded_search_returns_best_candidate_when_target_is_not_reached():
    data, features, objectives = _large_mixed_dataset(800)
    base_seed = 29
    attempt_seeds = [base_seed + offset * 104_729 for offset in range(3)]
    single_attempts = [
        select_representative_bo_rows(
            data,
            features,
            objectives,
            max_rows=90,
            target_fidelity_score=100,
            random_seed=seed,
            maximum_attempts=1,
        )
        for seed in attempt_seeds
    ]

    selected, diagnostic = select_representative_bo_rows(
        data,
        features,
        objectives,
        max_rows=90,
        target_fidelity_score=100,
        random_seed=base_seed,
        maximum_attempts=3,
        search_time_limit_seconds=60,
    )

    expected_selected, expected_diagnostic = max(
        single_attempts,
        key=lambda result: result[1].fidelity_score,
    )
    assert selected.index.tolist() == expected_selected.index.tolist()
    assert diagnostic.fidelity_score == expected_diagnostic.fidelity_score
    assert diagnostic.fidelity_target_met is False
    assert diagnostic.selection_attempts == 3
    assert diagnostic.search_time_limit_reached is False
