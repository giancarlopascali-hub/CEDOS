"""Supervised process runner for long Bayesian-optimization calculations."""

from __future__ import annotations

import multiprocessing
import os
import queue
import time
from typing import Any, Callable

from src.core.bo_policy import (
    BOCalculationCancelled,
    BOModelFitError,
    BOTimeoutExceeded,
    FeasibleInitializationError,
    load_bo_budget,
)


class BOProcessCancelled(RuntimeError):
    """Raised after a cancelled BO child has been confirmed stopped."""


class BOProcessTimeout(TimeoutError):
    """Raised after a BO child exceeded its hard deadline and was stopped."""


class BOChildFailure(RuntimeError):
    """A structured failure reported by, or inferred from, the BO child."""

    def __init__(self, message: str, error_code: str = "worker_error"):
        super().__init__(message)
        self.error_code = error_code


def _put_message(message_queue, message):
    """Best-effort IPC write that never masks the original worker outcome."""

    try:
        message_queue.put(message)
    except Exception:
        pass


def _bo_child_entry(payload, message_queue, child_cancel_event):
    """Top-level spawn target; imports the numerical stack only in the child."""

    try:
        thread_count = max(1, int(os.environ.get("BO_TORCH_NUM_THREADS", "1")))
        interop_count = max(
            1, int(os.environ.get("BO_TORCH_INTEROP_THREADS", "1"))
        )
        os.environ.setdefault("OMP_NUM_THREADS", str(thread_count))
        os.environ.setdefault("MKL_NUM_THREADS", str(thread_count))

        import pandas as pd
        import torch

        torch.set_num_threads(thread_count)
        try:
            torch.set_num_interop_threads(interop_count)
        except RuntimeError:
            # PyTorch permits this setting only before inter-op work begins.
            pass

        from src.core.bo import run_bo_optimization
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        compute_device = device
        dataframe = pd.DataFrame(payload["data"], columns=payload["columns"])
        timeout_seconds = float(payload["timeout_seconds"])
        child_deadline = time.monotonic() + timeout_seconds

        def progress(phase, percent, details):
            _put_message(message_queue, {
                "type": "progress",
                "phase": phase,
                "percent": percent,
                "details": details,
            })

        result = run_bo_optimization(
            df=dataframe,
            features_config=payload["features"],
            objectives_config=payload["objectives"],
            tweaks=payload["tweaks"],
            device=device,
            compute_device=compute_device,
            progress_callback=progress,
            cancel_event=child_cancel_event,
            deadline=child_deadline,
            compute_performance=bool(payload.get("compute_performance", False)),
            random_seed=payload.get("random_seed", 1729),
        )
        _put_message(message_queue, {"type": "result", "result": result.to_dict()})
    except BOCalculationCancelled as exc:
        _put_message(message_queue, {
            "type": "error", "error_code": "cancelled", "message": str(exc)
        })
    except BOTimeoutExceeded as exc:
        _put_message(message_queue, {
            "type": "error", "error_code": "timeout", "message": str(exc)
        })
    except BOModelFitError as exc:
        _put_message(message_queue, {
            "type": "error", "error_code": "model_fit", "message": str(exc)
        })
    except FeasibleInitializationError as exc:
        _put_message(message_queue, {
            "type": "error", "error_code": "constraints", "message": str(exc)
        })
    except ValueError as exc:
        _put_message(message_queue, {
            "type": "error", "error_code": "validation", "message": str(exc)
        })
    except MemoryError:
        _put_message(message_queue, {
            "type": "error",
            "error_code": "memory",
            "message": (
                "The BO worker ran out of memory. Reduce the dataset, objectives, "
                "or batch size."
            ),
        })
    except BaseException as exc:  # Child must always report a terminal outcome.
        import traceback

        traceback.print_exc()
        _put_message(message_queue, {
            "type": "error",
            "error_code": "worker_error",
            "message": (
                "The BO worker failed unexpectedly. Please retry; if the problem "
                "persists, review the server log."
            ),
        })


def _stop_process(process, grace_seconds):
    """Terminate, then kill if needed, and do not return while compute is live."""

    if not process.is_alive():
        process.join(timeout=0.2)
        return
    process.terminate()
    process.join(timeout=max(0.1, float(grace_seconds)))
    if process.is_alive():
        kill = getattr(process, "kill", None)
        if kill is not None:
            kill()
        else:  # pragma: no cover - all supported Python versions expose kill
            process.terminate()
        process.join(timeout=max(0.1, float(grace_seconds)))
    if process.is_alive():
        raise RuntimeError("Unable to stop the BO worker process.")


def run_bo_in_subprocess(
    payload: dict[str, Any],
    cancel_event,
    progress_callback: Callable[[str, int, dict[str, Any]], None] | None = None,
    *,
    _child_target=None,
) -> dict[str, Any]:
    """Run BO under a spawned-process hard deadline and return its result dict."""

    budget = load_bo_budget(is_cuda=False)
    timeout_seconds = min(
        float(payload.get("timeout_seconds", budget.total_job_timeout_sec)),
        budget.total_job_timeout_sec,
    )
    grace_seconds = max(0.5, float(os.environ.get("BO_CANCEL_GRACE_SEC", "3")))
    child_payload = dict(payload)
    child_payload["timeout_seconds"] = timeout_seconds

    context = multiprocessing.get_context("spawn")
    message_queue = context.Queue()
    child_cancel_event = context.Event()
    process = context.Process(
        target=_child_target or _bo_child_entry,
        args=(child_payload, message_queue, child_cancel_event),
        name="autoedos-bo-worker",
        daemon=True,
    )
    process.start()
    deadline = time.monotonic() + timeout_seconds
    cancellation_started = None
    timeout_started = None
    terminal_message = None

    try:
        while True:
            now = time.monotonic()
            if cancel_event is not None and cancel_event.is_set():
                child_cancel_event.set()
                cancellation_started = cancellation_started or now
            if now >= deadline:
                child_cancel_event.set()
                timeout_started = timeout_started or now

            if (
                cancellation_started is not None
                and now - cancellation_started >= grace_seconds
                and process.is_alive()
            ):
                _stop_process(process, grace_seconds)
            if (
                timeout_started is not None
                and now - timeout_started >= grace_seconds
                and process.is_alive()
            ):
                _stop_process(process, grace_seconds)

            try:
                message = message_queue.get(timeout=0.2)
            except queue.Empty:
                message = None
            if message is not None:
                if message.get("type") == "progress":
                    if progress_callback is not None:
                        progress_callback(
                            message.get("phase", "Running optimization"),
                            int(message.get("percent", 0)),
                            dict(message.get("details") or {}),
                        )
                else:
                    terminal_message = message

            if terminal_message is not None and not process.is_alive():
                process.join(timeout=0.2)
                break
            if not process.is_alive():
                process.join(timeout=0.2)
                # Queue feeder threads may publish just after process exit, and
                # progress messages can still precede the terminal result.
                drain_deadline = time.monotonic() + 1.0
                while time.monotonic() < drain_deadline:
                    try:
                        queued_message = message_queue.get(timeout=0.1)
                    except queue.Empty:
                        if terminal_message is not None:
                            break
                        continue
                    if queued_message.get("type") == "progress":
                        if progress_callback is not None:
                            progress_callback(
                                queued_message.get("phase", "Running optimization"),
                                int(queued_message.get("percent", 0)),
                                dict(queued_message.get("details") or {}),
                            )
                    else:
                        terminal_message = queued_message
                break

        if process.is_alive():
            _stop_process(process, grace_seconds)
        if cancellation_started is not None:
            raise BOProcessCancelled("Bayesian optimization was cancelled by the user.")
        if timeout_started is not None:
            raise BOProcessTimeout(
                f"Bayesian optimization reached its {timeout_seconds:.0f}-second hard limit."
            )
        if terminal_message is None:
            raise BOChildFailure(
                f"The BO worker exited unexpectedly (exit code {process.exitcode}).",
                "worker_exit",
            )
        if terminal_message.get("type") == "result":
            return terminal_message["result"]
        error_code = terminal_message.get("error_code", "worker_error")
        message = terminal_message.get("message", "Bayesian optimization failed.")
        if error_code == "cancelled":
            raise BOProcessCancelled(message)
        if error_code == "timeout":
            raise BOProcessTimeout(message)
        raise BOChildFailure(message, error_code)
    finally:
        if process.is_alive():
            _stop_process(process, grace_seconds)
        message_queue.close()
        message_queue.join_thread()
