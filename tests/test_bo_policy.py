import math

import pytest

from src.core.bo_policy import (
    assess_bo_workload,
    load_bo_budget,
    load_bo_large_data_settings,
    normalized_objective_weights,
    select_bo_policy,
)


@pytest.mark.parametrize(
    ("objective_count", "weights", "acquisition", "expected"),
    [
        (1, [1], "EI", "qLogEI"),
        (1, [1], "LCB", "qUCB"),
        (2, [1, 1], "EI", "qLogNEHVI"),
        (3, [10, 10, 10], "EI", "qLogNEHVI"),
        (4, [1, 1, 1, 1], "EI", "qLogNParEGO"),
        (10, [1] * 10, "EI", "qLogNParEGO"),
        (4, [4, 3, 2, 1], "EI", "scalarized-qLogEI"),
    ],
)
def test_bounded_acquisition_policy(
    objective_count, weights, acquisition, expected
):
    policy = select_bo_policy(objective_count, weights, acquisition, 3)
    assert policy.key == expected


def test_weight_equality_is_not_rounded_to_a_ui_tolerance():
    policy = select_bo_policy(4, [25, 25, 25, 24.99], "EI", 3)
    assert policy.key == "scalarized-qLogEI"


@pytest.mark.parametrize("weights", [[0, 1], [-1, 2], [math.nan, 1], [math.inf, 1]])
def test_invalid_weights_fail_before_model_fitting(weights):
    with pytest.raises(ValueError, match="greater than zero"):
        normalized_objective_weights(weights)


def test_cpu_budget_defaults_are_bounded(monkeypatch):
    for name in (
        "BO_MC_SAMPLES_CPU",
        "BO_NUM_RESTARTS_CPU",
        "BO_RAW_SAMPLES_CPU",
    ):
        monkeypatch.delenv(name, raising=False)
    budget = load_bo_budget(is_cuda=False)
    assert budget.mc_samples == 64
    assert budget.num_restarts == 8
    assert budget.raw_samples == 128
    assert budget.total_job_timeout_sec == 300


def test_request_cannot_supply_unbounded_internal_budget():
    budget = load_bo_budget(is_cuda=False)
    assert not hasattr(budget, "request_overrides")


def test_fidelity_target_and_bounded_search_settings_support_legacy_name(
    monkeypatch,
):
    monkeypatch.delenv("BO_SUBSET_TARGET_FIDELITY", raising=False)
    monkeypatch.setenv("BO_SUBSET_MIN_FIDELITY", "87")
    monkeypatch.setenv("BO_SUBSET_SELECTION_ATTEMPTS", "4")
    monkeypatch.setenv("BO_SUBSET_SELECTION_TIMEOUT_SEC", "15")

    legacy = load_bo_large_data_settings()
    assert legacy.target_fidelity_score == 87
    assert legacy.representative_selection_attempts == 4
    assert legacy.representative_selection_search_timeout_sec == 15

    monkeypatch.setenv("BO_SUBSET_TARGET_FIDELITY", "92")
    current = load_bo_large_data_settings()
    assert current.target_fidelity_score == 92


def test_multi_objective_method_can_be_overridden_explicitly():
    nehvi = select_bo_policy(
        5, [1] * 5, "EI", 3, requested_method="qLogNEHVI"
    )
    scalarized = select_bo_policy(
        3, [1] * 3, "EI", 3, requested_method="scalarized-qLogEI"
    )

    assert nehvi.key == "qLogNEHVI"
    assert nehvi.recommended_key == "qLogNParEGO"
    assert nehvi.is_override is True
    assert "slower" in nehvi.warning
    assert scalarized.key == "scalarized-qLogEI"
    assert scalarized.recommended_key == "qLogNEHVI"


def test_nparego_is_rejected_for_unequal_objective_importance():
    with pytest.raises(ValueError, match="unavailable.*unequal"):
        select_bo_policy(
            4, [4, 3, 2, 1], "EI", 3, requested_method="qLogNParEGO"
        )


def test_workload_guard_keeps_reported_shape_full_but_bounds_large_data():
    budget = load_bo_budget(is_cuda=False)
    reported = assess_bo_workload(
        row_count=69,
        feature_count=16,
        objective_count=7,
        batch_size=10,
        method="qLogNParEGO",
        budget=budget,
    )
    large = assess_bo_workload(
        row_count=10_000,
        feature_count=16,
        objective_count=10,
        batch_size=10,
        method="qLogNParEGO",
        budget=budget,
    )

    assert reported.tier in {"safe", "review"}
    assert reported.controls_required is False
    assert reported.recommended_training_rows == 69
    assert large.tier == "unsafe"
    assert large.controls_required is True
    assert 20 <= large.recommended_training_rows <= 250
    assert large.representative_feasible is True


def test_many_objective_nehvi_is_flagged_even_after_row_reduction():
    assessment = assess_bo_workload(
        row_count=10_000,
        feature_count=16,
        objective_count=7,
        batch_size=10,
        method="qLogNEHVI",
        budget=load_bo_budget(is_cuda=False),
    )

    assert assessment.tier == "unsafe"
    assert assessment.representative_feasible is False
