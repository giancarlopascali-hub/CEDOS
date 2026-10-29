import threading
import time

import pytest

from src.api.bo_worker import (
    BOProcessCancelled,
    BOProcessTimeout,
    run_bo_in_subprocess,
)


def _successful_child(_payload, message_queue, _cancel_event):
    message_queue.put({
        "type": "progress",
        "phase": "Optimizing candidate 1 of 1",
        "percent": 50,
        "details": {"algorithm": "qLogEI"},
    })
    message_queue.put({"type": "result", "result": {"ok": True}})


def _uncooperative_child(_payload, _message_queue, _cancel_event):
    while True:
        time.sleep(0.1)


def test_supervisor_delivers_progress_and_result(monkeypatch):
    monkeypatch.setenv("BO_CANCEL_GRACE_SEC", "0.1")
    progress = []
    result = run_bo_in_subprocess(
        {"timeout_seconds": 3},
        threading.Event(),
        lambda phase, percent, details: progress.append(
            (phase, percent, details)
        ),
        _child_target=_successful_child,
    )
    assert result == {"ok": True}
    assert progress[0][2]["algorithm"] == "qLogEI"


def test_supervisor_stops_uncooperative_cancelled_child(monkeypatch):
    monkeypatch.setenv("BO_CANCEL_GRACE_SEC", "0.1")
    cancelled = threading.Event()
    cancelled.set()
    started = time.monotonic()
    with pytest.raises(BOProcessCancelled):
        run_bo_in_subprocess(
            {"timeout_seconds": 5},
            cancelled,
            _child_target=_uncooperative_child,
        )
    assert time.monotonic() - started < 4


def test_supervisor_enforces_hard_timeout(monkeypatch):
    monkeypatch.setenv("BO_CANCEL_GRACE_SEC", "0.1")
    started = time.monotonic()
    with pytest.raises(BOProcessTimeout):
        run_bo_in_subprocess(
            {"timeout_seconds": 0.2},
            threading.Event(),
            _child_target=_uncooperative_child,
        )
    assert time.monotonic() - started < 4


def test_real_child_returns_serializable_cold_start_result(monkeypatch):
    monkeypatch.setenv("BO_CANCEL_GRACE_SEC", "1")
    result = run_bo_in_subprocess(
        {
            "data": [],
            "columns": ["x", "objective"],
            "features": [
                {"name": "x", "type": "continuous", "range": "[0, 1]"}
            ],
            "objectives": [
                {"name": "objective", "type": "maximize", "importance": 1}
            ],
            "tweaks": {"batch_size": 2, "constraints": {}},
            "compute_performance": False,
            "random_seed": 17,
            "timeout_seconds": 30,
        },
        threading.Event(),
    )
    assert len(result["suggestions"]) == 2
    assert result["diagnostics"]["algorithm"] == "sobol-cold-start"
