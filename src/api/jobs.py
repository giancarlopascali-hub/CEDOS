"""Domain-neutral in-memory background job registry.

Jobs are intentionally process-local. Deployments using this registry must run
one application process unless the registry is replaced with shared storage.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from typing import Any, Callable


_JOBS: dict[str, dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()
_JOB_TTL_SECONDS = 3600


def _run_background_target(job_id, payload, target):
    """Ensure a stopped supervisor can never leave an active slot behind."""

    try:
        target(job_id, payload)
    except BaseException:
        import traceback

        traceback.print_exc()
    finally:
        with _JOBS_LOCK:
            job = _JOBS.get(job_id)
            if job is not None and job.get("status") in {
                "queued", "running", "cancelling"
            }:
                job.update({
                    "status": "failed",
                    "phase": "Supervisor stopped unexpectedly",
                    "error": "The background supervisor stopped unexpectedly. Please retry.",
                    "error_code": "supervisor_exit",
                    "finished_at": time.time(),
                    "updated_at": time.time(),
                })


def _max_jobs() -> int:
    return max(1, int(os.environ.get(
        "MAX_CONCURRENT_ANALYSIS_JOBS",
        os.environ.get("SA_MAX_CONCURRENT_JOBS", "1"),
    )))


def cleanup_expired_jobs() -> None:
    """Remove completed jobs after their retention period."""
    cutoff = time.time() - _JOB_TTL_SECONDS
    with _JOBS_LOCK:
        for job_id in [
            job_id for job_id, job in _JOBS.items()
            if job.get("finished_at", float("inf")) < cutoff
        ]:
            _JOBS.pop(job_id, None)


def queue_background_job(
    kind: str,
    payload: dict[str, Any],
    target: Callable[[str, dict[str, Any]], None],
    initial_phase: str,
) -> str | None:
    """Start a daemon worker, or return ``None`` when capacity is occupied."""
    cleanup_expired_jobs()
    with _JOBS_LOCK:
        active_jobs = sum(
            job.get("status") in {"queued", "running", "cancelling"}
            for job in _JOBS.values()
        )
        if active_jobs >= _max_jobs():
            return None
        job_id = str(uuid.uuid4())
        _JOBS[job_id] = {
            "kind": kind,
            "status": "queued",
            "phase": initial_phase,
            "percent": 0,
            "created_at": time.time(),
            "updated_at": time.time(),
            "cancel_event": threading.Event(),
        }

    worker = threading.Thread(
        target=_run_background_target,
        args=(job_id, payload, target),
        name=f"{kind}-{job_id[:8]}",
        daemon=True,
    )
    with _JOBS_LOCK:
        if job_id in _JOBS:
            _JOBS[job_id]["supervisor_thread"] = worker
    worker.start()
    return job_id


def update_background_job(job_id: str, **updates: Any) -> None:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is not None:
            job.update(updates)
            job["updated_at"] = time.time()


def get_cancel_event(job_id: str) -> threading.Event | None:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        return job.get("cancel_event") if job is not None else None


def get_background_job_response(job_id: str) -> dict[str, Any] | None:
    cleanup_expired_jobs()
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return None
        response = {
            "job_id": job_id,
            "kind": job.get("kind"),
            "status": job.get("status"),
            "phase": job.get("phase"),
            "percent": job.get("percent", 0),
            "elapsed_seconds": round(
                max(
                    0.0,
                    (job.get("finished_at") or time.time())
                    - (job.get("started_at") or job.get("created_at", time.time())),
                ),
                1,
            ),
        }
        if job.get("algorithm"):
            response["algorithm"] = job["algorithm"]
        if job.get("details"):
            response["details"] = job["details"]
        if job.get("status") == "completed":
            response["result"] = job.get("result")
        elif job.get("status") in {"failed", "cancelled"}:
            response["error"] = job.get("error")
            if job.get("error_code"):
                response["error_code"] = job["error_code"]
        return response


def cancel_background_job(job_id: str) -> dict[str, Any] | None:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return None
        if job.get("status") not in {"completed", "failed", "cancelled"}:
            job["cancel_event"].set()
            job["status"] = "cancelling"
            job["phase"] = "Cancellation requested"
            job["updated_at"] = time.time()
        return {"job_id": job_id, "status": job.get("status")}
