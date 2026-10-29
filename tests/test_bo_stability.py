import numpy as np
import pandas as pd
import pytest
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from botorch.optim.core import OptimizationResult, OptimizationStatus
from gpytorch.mlls import ExactMarginalLogLikelihood

import src.core.bo as bo
from src.core.bo_policy import BOModelFitError


def _finite_mll():
    train_x = torch.linspace(0.0, 1.0, 8, dtype=torch.float64).unsqueeze(-1)
    train_y = torch.sin(train_x * 3.0)
    model = SingleTaskGP(
        train_x,
        train_y,
        outcome_transform=Standardize(m=1),
    )
    return ExactMarginalLogLikelihood(model.likelihood, model)


def _result(status, message, step=1, fval=1.0):
    return OptimizationResult(
        step=step,
        fval=fval,
        status=status,
        runtime=0.01,
        message=message,
    )


def test_abnormal_fit_retries_with_larger_line_search(monkeypatch):
    calls = []
    results = iter([
        _result(OptimizationStatus.FAILURE, "ABNORMAL: "),
        _result(OptimizationStatus.SUCCESS, "CONVERGENCE"),
    ])

    def fake_fit(_mll, *, options, timeout_sec):
        calls.append((dict(options), timeout_sec))
        return next(results)

    monkeypatch.setattr(bo, "fit_gpytorch_mll_scipy", fake_fit)
    diagnostic = bo._fit_required_model(
        _finite_mll(),
        label="test objective",
        maxiter=20,
        maxls=50,
        retries=1,
        timeout_sec=10,
    )

    assert diagnostic["status"] == "success"
    assert diagnostic["attempts"] == 2
    assert diagnostic["dtype"] == "torch.float64"
    assert [call[0]["maxls"] for call in calls] == [50, 100]


def test_persistent_abnormal_fit_requires_and_passes_finite_validation(monkeypatch):
    monkeypatch.setattr(
        bo,
        "fit_gpytorch_mll_scipy",
        lambda *_args, **_kwargs: _result(
            OptimizationStatus.FAILURE, "ABNORMAL: "
        ),
    )
    diagnostic = bo._fit_required_model(
        _finite_mll(),
        label="test objective",
        maxiter=20,
        maxls=50,
        retries=1,
        timeout_sec=10,
    )

    assert diagnostic["status"] == "failure"
    assert diagnostic["degraded"] is True
    assert diagnostic["accepted_after_validation"] is True
    assert diagnostic["attempts"] == 2


def test_non_abnormal_failure_is_not_accepted(monkeypatch):
    monkeypatch.setattr(
        bo,
        "fit_gpytorch_mll_scipy",
        lambda *_args, **_kwargs: _result(
            OptimizationStatus.FAILURE, "NUMERICAL ERROR"
        ),
    )
    with pytest.raises(BOModelFitError, match="NUMERICAL ERROR"):
        bo._fit_required_model(
            _finite_mll(),
            label="test objective",
            maxiter=20,
            timeout_sec=10,
        )


def test_non_finite_optimizer_result_is_not_accepted(monkeypatch):
    monkeypatch.setattr(
        bo,
        "fit_gpytorch_mll_scipy",
        lambda *_args, **_kwargs: _result(
            OptimizationStatus.SUCCESS, "CONVERGENCE", fval=np.nan
        ),
    )
    with pytest.raises(BOModelFitError, match="non-finite objective"):
        bo._fit_required_model(
            _finite_mll(),
            label="test objective",
            maxiter=20,
            timeout_sec=10,
        )


def test_objective_diagnostics_report_missing_constant_sparse_and_redundant_data():
    data = pd.DataFrame({
        "constant": [1.0] * 6,
        "left": [1, 2, 3, 4, 5, 6],
        "right": [2, 4, 6, 8, 10, 12],
        "sparse": [1.0, np.nan, np.nan, 2.0, np.nan, np.nan],
    })
    objectives = [
        {"name": name, "type": "maximize"}
        for name in data.columns
    ]

    diagnostics, warnings = bo.analyze_objective_data(
        data, objectives, feature_count=4
    )

    assert diagnostics["objectives"]["sparse"]["finite_count"] == 2
    assert diagnostics["objectives"]["sparse"]["missing_count"] == 4
    assert diagnostics["redundant_pairs"][0]["correlation"] == pytest.approx(1.0)
    assert any("constant" in warning and "cannot influence" in warning for warning in warnings)
    assert any("sparse" in warning and "missing" in warning for warning in warnings)
    assert any("sparse" in warning and "unreliable" in warning for warning in warnings)
    assert any("almost perfectly correlated" in warning for warning in warnings)


def test_reported_69_by_16_shape_uses_double_precision_and_returns_quality_notes(
    monkeypatch,
):
    rng = np.random.default_rng(17)
    feature_names = [f"z{index}" for index in range(1, 17)]
    x_values = rng.uniform(0.0, 1.0, size=(69, 16))
    data = pd.DataFrame(x_values, columns=feature_names)
    current = 0.55 + 0.5 * x_values[:, 0] - 0.2 * x_values[:, 1]
    data["Viable?"] = 1.0
    data["i_A_cm2"] = current
    data["power_W_cm2"] = 0.55 * current
    data["eta_mt_mV"] = 10.0 + 30.0 * x_values[:, 2]
    data["eta_contact_mV"] = 40.0 + 70.0 * x_values[:, 3] ** 2
    data["flood"] = 0.04 + 0.2 * x_values[:, 4]
    data["dP_Pa"] = np.exp(2.0 + 7.0 * x_values[:, 5])

    fit_dtypes = []

    def finite_fit(mll, **_kwargs):
        mll.eval()
        mll.model.eval()
        fit_dtypes.append(mll.model.train_inputs[0].dtype)
        return {
            "label": "test",
            "status": "success",
            "degraded": False,
            "accepted_after_validation": False,
            "attempts": 1,
            "dtype": str(mll.model.train_inputs[0].dtype),
        }

    def midpoint_candidate(*, acq_function, bounds, **_kwargs):
        assert bounds.dtype == torch.float64
        assert all(parameter.dtype == torch.float64 for parameter in acq_function.model.parameters())
        return bounds.mean(dim=0, keepdim=True), bounds.new_tensor(0.0)

    monkeypatch.setattr(bo, "_fit_required_model", finite_fit)
    monkeypatch.setattr(bo, "optimize_acqf", midpoint_candidate)

    features = [
        {"name": name, "type": "continuous", "range": "[0, 1]"}
        for name in feature_names
    ]
    objective_names = [
        "Viable?",
        "i_A_cm2",
        "power_W_cm2",
        "eta_mt_mV",
        "eta_contact_mV",
        "flood",
        "dP_Pa",
    ]
    objectives = [
        {"name": name, "type": "maximize", "importance": 1}
        for name in objective_names
    ]

    result = bo.run_bo_optimization(
        data,
        features,
        objectives,
        {"batch_size": 1, "constraints": {}},
        torch.device("cpu"),
        torch.device("cpu"),
        random_seed=17,
    )

    assert len(result.suggestions) == 1
    assert fit_dtypes == [torch.float64] * len(objectives)
    assert result.diagnostics["tensor_dtype"] == "torch.float64"
    assert any(
        "Viable?" in warning and "constant" in warning
        for warning in result.diagnostics["data_warnings"]
    )
    assert any(
        "i_A_cm2" in warning and "power_W_cm2" in warning
        for warning in result.diagnostics["data_warnings"]
    )


def test_nonempty_dataset_with_no_valid_objective_is_not_treated_as_cold_start():
    data = pd.DataFrame({"x": [0.1, 0.9], "objective": [np.nan, np.nan]})
    with pytest.raises(ValueError, match="No rows contain valid"):
        bo.run_bo_optimization(
            data,
            [{"name": "x", "type": "continuous", "range": "[0, 1]"}],
            [{"name": "objective", "type": "maximize", "importance": 1}],
            {"batch_size": 1, "constraints": {}},
            torch.device("cpu"),
            torch.device("cpu"),
        )


@pytest.mark.parametrize("data_mode", ["auto", "representative"])
def test_unsafe_large_data_uses_best_subset_below_fidelity_target(
    monkeypatch, data_mode
):
    rng = np.random.default_rng(31)
    feature_values = rng.uniform(0.0, 1.0, size=(600, 2))
    data = pd.DataFrame(feature_values, columns=["x1", "x2"])
    data["y1"] = np.sin(feature_values[:, 0] * 3.0) + feature_values[:, 1]
    data["y2"] = feature_values[:, 0] - 0.5 * feature_values[:, 1]
    data["y3"] = feature_values[:, 0] * feature_values[:, 1]
    data["y4"] = (feature_values[:, 0] - 0.4) ** 2 + feature_values[:, 1]

    def finite_fit(mll, **_kwargs):
        return {
            "label": "test",
            "status": "success",
            "degraded": False,
            "accepted_after_validation": False,
            "attempts": 1,
            "dtype": str(mll.model.train_inputs[0].dtype),
        }

    monkeypatch.setattr(bo, "_fit_required_model", finite_fit)
    monkeypatch.setattr(
        bo,
        "optimize_acqf",
        lambda *, bounds, **_kwargs: (
            bounds.mean(dim=0, keepdim=True),
            bounds.new_tensor(0.0),
        ),
    )
    monkeypatch.setenv("BO_SUBSET_TARGET_FIDELITY", "100")
    monkeypatch.setenv("BO_SUBSET_SELECTION_ATTEMPTS", "2")
    monkeypatch.setenv("BO_SUBSET_SELECTION_TIMEOUT_SEC", "60")
    features = [
        {"name": "x1", "type": "continuous", "range": "[0, 1]"},
        {"name": "x2", "type": "continuous", "range": "[0, 1]"},
    ]
    objectives = [
        {
            "name": f"y{index}",
            "type": "maximize",
            "importance": importance,
        }
        for index, importance in enumerate([40, 30, 20, 10], start=1)
    ]

    result = bo.run_bo_optimization(
        data,
        features,
        objectives,
        {"batch_size": 1, "constraints": {}, "data_mode": data_mode},
        torch.device("cpu"),
        torch.device("cpu"),
        random_seed=31,
    )

    handling = result.diagnostics["data_handling"]
    assert handling["requested_mode"] == data_mode
    assert handling["effective_mode"] == "representative"
    assert handling["source_rows"] == 600
    assert handling["training_rows"] < 600
    assert handling["selection"]["fidelity_score"] < 100
    assert handling["selection"]["target_fidelity_score"] == 100
    assert handling["selection"]["fidelity_target_met"] is False
    assert handling["selection"]["selection_attempts"] == 2
    assert any(
        "Representative-subset fallback used" in warning
        and "Optimization continued" in warning
        and "target of 100.0/100" in warning
        for warning in result.diagnostics["data_warnings"]
    )
    assert result.diagnostics["workload"]["full_data"]["tier"] == "unsafe"
    assert len(result.suggestions) == 1
