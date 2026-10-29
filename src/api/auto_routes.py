"""CEDOS ↔ IMEX exchange for identified experiment batches.

The local preview runs without a token. A hosted Space requires an owner token
and one worker because campaign state is process-local. A server restart loses
the active campaign; multiuser or multiworker use needs isolated durable state.
"""

from __future__ import annotations

import io
import hashlib
import json
import math
import os
import re
import secrets
import threading
import time
from copy import deepcopy

import pandas as pd
from flask import Blueprint, jsonify, request, send_file


auto_bp = Blueprint("auto", __name__)
_AUTO_STATE_LOCK = threading.RLock()
_IDENTIFIER = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_LEASE_SECONDS = max(5, int(os.environ.get("CEDOS_PROPOSAL_LEASE_SECONDS", "30")))


@auto_bp.before_request
def require_hosted_auto_token():
    """Keep the local exchange unchanged, but protect it on a public Space.

    The blueprint also owns the unrelated /export route, which must remain
    available to all visitors. Only /auto/* uses this owner-controlled token.
    """
    if not os.environ.get("SPACE_ID") or not request.path.startswith("/auto/"):
        return None
    expected = os.environ.get("CEDOS_AUTO_TOKEN", "")
    if not expected:
        return _error("Hosted Auto is not configured.", 503)
    scheme, separator, provided = request.headers.get("Authorization", "").partition(" ")
    if separator != " " or scheme.lower() != "bearer" or not secrets.compare_digest(provided, expected):
        response, status = _error("A valid Auto access token is required.", 401)
        response.headers["WWW-Authenticate"] = 'Bearer realm="CEDOS Auto"'
        return response, status
    return None


def _initial_state():
    return {
        "campaign_id": None,
        "optimizer": None,
        "status": "idle",
        "iteration": 0,
        "proposal_id": None,
        "proposal": None,
        "proposal_fingerprint": None,
        "lease_until": 0.0,
        "results": None,
        "results_fingerprint": None,
        "completed_results": {},
    }


AUTO_STATE = _initial_state()


def _reset_auto_state():
    """Reset the in-memory exchange (used by local integration tests)."""
    with _AUTO_STATE_LOCK:
        AUTO_STATE.clear()
        AUTO_STATE.update(_initial_state())


def _error(message, status=400):
    return jsonify({"error": message}), status


def _identifier(value, label):
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{label} must be a nonempty identifier.")
    return value


def _unique_names(value, label, *, allow_empty=False):
    if not isinstance(value, list) or (not value and not allow_empty):
        raise ValueError(f"{label} must be a nonempty list.")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"{label} must contain nonempty names.")
    if len(value) != len(set(value)):
        raise ValueError(f"{label} must contain unique names.")
    return value


def _fingerprint(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("The exchange payload contains invalid values.") from exc


def _same_cell(left, right):
    """Compare CSV number strings to their in-memory numeric equivalents."""
    if left == right:
        return True
    try:
        left_number, right_number = float(left), float(right)
        return math.isfinite(left_number) and math.isfinite(right_number) and left_number == right_number
    except (TypeError, ValueError, OverflowError):
        return False


def _valid_feature_value(value):
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return False
    if str(value).strip() == "":
        return False
    if isinstance(value, (int, float)):
        try:
            return math.isfinite(float(value))
        except OverflowError:
            return False
    return True


def _proposal_payload(body):
    if not isinstance(body, dict):
        raise ValueError("A JSON proposal object is required.")
    campaign_id = _identifier(body.get("campaign_id"), "campaign_id")
    proposal_id = _identifier(body.get("proposal_id"), "proposal_id")
    optimizer = body.get("optimizer")
    if optimizer not in {"doe", "bo", "tbo"}:
        raise ValueError("optimizer must be doe, bo or tbo.")
    revision = body.get("dataset_rev")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
        raise ValueError("dataset_rev must be a nonnegative integer.")
    config_revision = body.get("config_rev")
    if isinstance(config_revision, bool) or not isinstance(config_revision, int) or config_revision < 0:
        raise ValueError("config_rev must be a nonnegative integer.")
    columns = _unique_names(body.get("columns"), "columns")
    if "row_id" in columns:
        raise ValueError("row_id is reserved for exchange identity, not a dataset column.")
    objectives = _unique_names(body.get("objective_names"), "objective_names")
    if any(name not in columns for name in objectives):
        raise ValueError("Every objective name must occur in columns.")
    suggestions = body.get("suggestions")
    features_only = body.get("features_only")
    rows = body.get("rows")
    if not isinstance(suggestions, list) or not suggestions:
        raise ValueError("suggestions must be a nonempty list.")
    if not isinstance(features_only, list) or not isinstance(rows, list):
        raise ValueError("features_only and rows must be lists.")
    if len(suggestions) != len(features_only) or len(rows) != len(suggestions):
        raise ValueError("suggestions, features_only and rows must have the same length.")
    if any(not isinstance(item, dict) for item in suggestions + features_only + rows):
        raise ValueError("Each suggested experiment must be an object.")
    row_ids = [_identifier(row.get("row_id"), "row_id") for row in rows]
    if len(row_ids) != len(set(row_ids)):
        raise ValueError("A proposal cannot repeat a row_id.")
    if any(name not in columns and name != "row_id" for row in rows for name in row):
        raise ValueError("A proposal row contains an unknown dataset column.")
    for index, feature_row in enumerate(features_only):
        if feature_row.get("row_id", row_ids[index]) != row_ids[index]:
            raise ValueError("A features_only row_id does not match its proposal row.")
        if not any(name != "row_id" for name in feature_row):
            raise ValueError("Each features_only row needs at least one feature.")
        if any(name not in columns and name != "row_id" for name in feature_row):
            raise ValueError("features_only references an unknown dataset column.")
        if any(name in objectives for name in feature_row):
            raise ValueError("features_only cannot contain objective columns.")
        for name, value in feature_row.items():
            if name == "row_id":
                continue
            if not _valid_feature_value(value):
                raise ValueError(f"Proposed feature {name} needs a finite scalar value.")
        if any(name not in rows[index] or not _same_cell(value, rows[index][name])
               for name, value in feature_row.items() if name != "row_id"):
            raise ValueError("A proposed feature differs from its pending dataset row.")
    payload = deepcopy(body)
    payload.update({
        "campaign_id": campaign_id,
        "proposal_id": proposal_id,
        "optimizer": optimizer,
        "dataset_rev": revision,
        "config_rev": config_revision,
        "columns": columns,
        "objective_names": objectives,
        "row_ids": row_ids,
    })
    _fingerprint(payload)
    return payload


def _results_payload(body, proposal):
    if not isinstance(body, dict):
        raise ValueError("Results must be an identified JSON object.")
    campaign_id = _identifier(body.get("campaign_id"), "campaign_id")
    proposal_id = _identifier(body.get("proposal_id"), "proposal_id")
    if campaign_id != proposal["campaign_id"] or proposal_id != proposal["proposal_id"]:
        raise ValueError("Results identify a different campaign or proposal.")
    rows = body.get("rows")
    if not isinstance(rows, list) or len(rows) != len(proposal["row_ids"]):
        raise ValueError("Results must contain exactly one row per proposed experiment.")
    by_id = {}
    allowed = set(proposal["columns"]) | {"row_id"}
    proposed_rows = {row["row_id"]: row for row in proposal["rows"]}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Every result row must be an object.")
        row_id = _identifier(row.get("row_id"), "row_id")
        if row_id in by_id:
            raise ValueError("Results contain a duplicate row_id.")
        if any(name not in allowed for name in row):
            raise ValueError("A result contains an unknown dataset column.")
        if row_id not in proposed_rows:
            raise ValueError("Result row IDs do not match the proposed experiments.")
        for name, value in row.items():
            if name == "row_id" or name in proposal["objective_names"]:
                continue
            if name not in proposed_rows[row_id] or not _same_cell(value, proposed_rows[row_id][name]):
                raise ValueError(f"Result row {row_id} changes proposed {name}.")
        for objective in proposal["objective_names"]:
            value = row.get(objective)
            if value is None or value == "" or isinstance(value, bool):
                raise ValueError(f"Result row {row_id} lacks numeric objective {objective}.")
            try:
                if not math.isfinite(float(value)):
                    raise ValueError
            except (TypeError, ValueError):
                raise ValueError(f"Result row {row_id} lacks numeric objective {objective}.") from None
        by_id[row_id] = row
    if set(by_id) != set(proposal["row_ids"]):
        raise ValueError("Result row IDs do not match the proposed experiments.")
    normalized = {
        "campaign_id": campaign_id,
        "proposal_id": proposal_id,
        "rows": [deepcopy(by_id[row_id]) for row_id in proposal["row_ids"]],
    }
    _fingerprint(normalized)
    return normalized


def _result_signature(body):
    """Order-independent signature for a measurement upload retry."""
    rows = body.get("rows")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Results must contain identified rows.")
    row_ids = [_identifier(row.get("row_id"), "row_id") for row in rows]
    if len(row_ids) != len(set(row_ids)):
        raise ValueError("Results contain a duplicate row_id.")
    canonical = {"campaign_id": body["campaign_id"], "proposal_id": body["proposal_id"],
                 "rows": sorted(rows, key=lambda row: row["row_id"])}
    return hashlib.sha256(_fingerprint(canonical).encode("utf-8")).hexdigest()


def _current(campaign_id, proposal_id=None):
    if AUTO_STATE["campaign_id"] != campaign_id:
        return False
    if proposal_id is not None and AUTO_STATE["proposal_id"] != proposal_id:
        return False
    return True


@auto_bp.route("/auto/proposal", methods=["POST", "GET"])
def proposal():
    if request.method == "GET":
        with _AUTO_STATE_LOCK:
            pending = AUTO_STATE["proposal"]
            if pending is None or AUTO_STATE["status"] not in {"pending", "leased"}:
                return "", 204
            now = time.monotonic()
            if AUTO_STATE["status"] == "leased" and now < AUTO_STATE["lease_until"]:
                return "", 204
            AUTO_STATE["status"] = "leased"
            AUTO_STATE["lease_until"] = now + _LEASE_SECONDS
            delivery = deepcopy(pending)
            delivery["iteration"] = AUTO_STATE["iteration"]
            delivery["lease_seconds"] = _LEASE_SECONDS
        return jsonify(delivery)

    try:
        payload = _proposal_payload(request.get_json(silent=True))
        fingerprint = _fingerprint(payload)
    except ValueError as exc:
        return _error(str(exc))
    with _AUTO_STATE_LOCK:
        same_campaign = AUTO_STATE["campaign_id"] == payload["campaign_id"]
        if same_campaign and AUTO_STATE["proposal_id"] == payload["proposal_id"]:
            if AUTO_STATE["proposal_fingerprint"] != fingerprint:
                return _error("A proposal_id cannot be reused with different contents.", 409)
            return jsonify({
                "status": "success", "campaign_id": payload["campaign_id"],
                "proposal_id": payload["proposal_id"],
                "iteration": AUTO_STATE["iteration"], "duplicate": True,
            })
        if same_campaign and AUTO_STATE["status"] not in {"idle", "merged", "cancelled"}:
            return _error("Finish or cancel the current proposal before sending another.", 409)
        if not same_campaign and AUTO_STATE["status"] not in {"idle", "merged", "cancelled"}:
            return _error("Another Auto campaign is already active in this local server.", 409)
        if not same_campaign:
            AUTO_STATE.update(_initial_state())
            AUTO_STATE["campaign_id"] = payload["campaign_id"]
        AUTO_STATE.update({
            "optimizer": payload["optimizer"],
            "status": "pending",
            "iteration": AUTO_STATE["iteration"] + 1,
            "proposal_id": payload["proposal_id"],
            "proposal": payload,
            "proposal_fingerprint": fingerprint,
            "lease_until": 0.0,
            "results": None,
            "results_fingerprint": None,
        })
        iteration = AUTO_STATE["iteration"]
    return jsonify({
        "status": "success", "campaign_id": payload["campaign_id"],
        "proposal_id": payload["proposal_id"], "iteration": iteration,
    })


@auto_bp.post("/auto/proposal/<proposal_id>/ack")
def acknowledge_proposal(proposal_id):
    body = request.get_json(silent=True) or {}
    try:
        campaign_id = _identifier(body.get("campaign_id"), "campaign_id")
    except ValueError as exc:
        return _error(str(exc))
    with _AUTO_STATE_LOCK:
        if not _current(campaign_id, proposal_id) or AUTO_STATE["proposal"] is None:
            return _error("The proposal is no longer active.", 409)
        if AUTO_STATE["status"] in {"pending", "leased"}:
            AUTO_STATE["status"] = "dispatched"
            AUTO_STATE["lease_until"] = 0.0
        elif AUTO_STATE["status"] not in {"dispatched", "results_ready", "merged"}:
            return _error("The proposal cannot be acknowledged in its current state.", 409)
        status = AUTO_STATE["status"]
    return jsonify({"status": status, "campaign_id": campaign_id, "proposal_id": proposal_id})


@auto_bp.post("/auto/proposal/<proposal_id>/release")
def release_proposal(proposal_id):
    body = request.get_json(silent=True) or {}
    try:
        campaign_id = _identifier(body.get("campaign_id"), "campaign_id")
    except ValueError as exc:
        return _error(str(exc))
    with _AUTO_STATE_LOCK:
        if not _current(campaign_id, proposal_id) or AUTO_STATE["proposal"] is None:
            return _error("The proposal is no longer active.", 409)
        if AUTO_STATE["status"] not in {"pending", "leased"}:
            return _error("A dispatched proposal cannot be released.", 409)
        AUTO_STATE["status"] = "pending"
        AUTO_STATE["lease_until"] = 0.0
    return jsonify({"status": "pending", "campaign_id": campaign_id, "proposal_id": proposal_id})


@auto_bp.route("/auto/results", methods=["POST", "GET"])
def results():
    if request.method == "GET":
        try:
            campaign_id = _identifier(request.args.get("campaign_id"), "campaign_id")
            proposal_id = _identifier(request.args.get("proposal_id"), "proposal_id")
        except ValueError as exc:
            return _error(str(exc))
        with _AUTO_STATE_LOCK:
            if not _current(campaign_id, proposal_id):
                return _error("The requested campaign or proposal is not active.", 409)
            if AUTO_STATE["status"] == "merged":
                return "", 204
            if AUTO_STATE["results"] is None:
                return jsonify({"status": "waiting", "campaign_id": campaign_id,
                                "proposal_id": proposal_id}), 202
            response = deepcopy(AUTO_STATE["results"])
        return jsonify(response)

    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return _error("An identified JSON result object is required.")
    try:
        campaign_id = _identifier(body.get("campaign_id"), "campaign_id")
        proposal_id = _identifier(body.get("proposal_id"), "proposal_id")
    except ValueError as exc:
        return _error(str(exc))
    with _AUTO_STATE_LOCK:
        if not _current(campaign_id, proposal_id) or AUTO_STATE["proposal"] is None:
            if AUTO_STATE["campaign_id"] == campaign_id:
                previous = AUTO_STATE["completed_results"].get(proposal_id)
                if previous is not None:
                    try:
                        if _result_signature(body) == previous:
                            return jsonify({"status": "success", "campaign_id": campaign_id,
                                            "proposal_id": proposal_id, "duplicate": True})
                    except ValueError:
                        pass
                    return _error("Different results have already been accepted for this proposal.", 409)
            return _error("Results do not match the current campaign and proposal.", 409)
        if AUTO_STATE["status"] not in {"dispatched", "results_ready", "merged"}:
            return _error("A proposal must be acknowledged by IMEX before results are accepted.", 409)
        try:
            normalized = _results_payload(body, AUTO_STATE["proposal"])
            fingerprint = _fingerprint(normalized)
        except ValueError as exc:
            return _error(str(exc))
        if AUTO_STATE["results"] is not None:
            if AUTO_STATE["results_fingerprint"] != fingerprint:
                return _error("Different results have already been accepted for this proposal.", 409)
            return jsonify({"status": "success", "campaign_id": campaign_id,
                            "proposal_id": proposal_id, "duplicate": True})
        AUTO_STATE["results"] = normalized
        AUTO_STATE["results_fingerprint"] = fingerprint
        AUTO_STATE["status"] = "results_ready"
    return jsonify({"status": "success", "campaign_id": campaign_id,
                    "proposal_id": proposal_id})


@auto_bp.post("/auto/results/<proposal_id>/ack")
def acknowledge_results(proposal_id):
    body = request.get_json(silent=True) or {}
    try:
        campaign_id = _identifier(body.get("campaign_id"), "campaign_id")
    except ValueError as exc:
        return _error(str(exc))
    with _AUTO_STATE_LOCK:
        if not _current(campaign_id, proposal_id):
            return _error("The proposal is no longer active.", 409)
        if AUTO_STATE["results"] is None:
            return _error("There are no results to acknowledge.", 409)
        completed = AUTO_STATE["completed_results"]
        completed[proposal_id] = _result_signature(AUTO_STATE["results"])
        while len(completed) > 16:
            completed.pop(next(iter(completed)))
        AUTO_STATE["status"] = "merged"
    return jsonify({"status": "merged", "campaign_id": campaign_id,
                    "proposal_id": proposal_id})


@auto_bp.post("/auto/stop")
def stop():
    body = request.get_json(silent=True) or {}
    try:
        campaign_id = _identifier(body.get("campaign_id"), "campaign_id")
    except ValueError as exc:
        return _error(str(exc))
    with _AUTO_STATE_LOCK:
        if not _current(campaign_id):
            return _error("The campaign is not active.", 409)
        AUTO_STATE.update({
            "status": "cancelled", "proposal": None, "proposal_id": None,
            "proposal_fingerprint": None, "lease_until": 0.0,
            "results": None, "results_fingerprint": None,
        })
    return jsonify({"status": "cancelled", "campaign_id": campaign_id})


@auto_bp.get("/auto/status")
def status():
    campaign_id = request.args.get("campaign_id")
    with _AUTO_STATE_LOCK:
        if campaign_id is not None and AUTO_STATE["campaign_id"] != campaign_id:
            return _error("The requested campaign is not active.", 409)
        return jsonify({
            "status": AUTO_STATE["status"],
            "campaign_id": AUTO_STATE["campaign_id"],
            "proposal_id": AUTO_STATE["proposal_id"],
            "optimizer": AUTO_STATE["optimizer"],
            "iteration": AUTO_STATE["iteration"],
        })


@auto_bp.post("/export")
def export_csv():
    """Download the current main dataset, separately from Setup CSV files."""
    body = request.get_json(silent=True) or {}
    columns = body.get("columns")
    data = body.get("data")
    if not isinstance(columns, list) or not isinstance(data, list):
        return _error("A valid dataset and columns are required.")
    if any(not isinstance(row, list) or len(row) != len(columns) for row in data):
        return _error("Every dataset row must match the columns.")
    try:
        frame = pd.DataFrame(data, columns=columns)
        output = io.BytesIO()
        frame.to_csv(output, index=False)
        output.seek(0)
        return send_file(output, mimetype="text/csv", as_attachment=True,
                         download_name="cedos_export.csv")
    except (TypeError, ValueError) as exc:
        return _error(str(exc))
