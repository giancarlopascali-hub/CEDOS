"""Local folder exchange for identified CEDOS Auto experiments.

Run this beside the single-process CEDOS server. A proposed batch is written
once to a CSV and acknowledged only after that file exists. Fill the matching
result example, save it with the ``.csv`` suffix, and the bridge uploads it.
The original AutoEDOS IMEX application is left untouched.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def api_request(server: str, method: str, path: str, payload=None, *, token: str | None = None):
    """Return (HTTP status, parsed JSON) for one CEDOS bridge request."""
    parsed = urllib.parse.urlsplit(server)
    if token and parsed.scheme != "https" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("A hosted Auto token may only be sent over HTTPS.")
    data = None if payload is None else json.dumps(payload, allow_nan=False).encode("utf-8")
    headers = {"Content-Type": "application/json"} if data is not None else {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        server.rstrip("/") + path,
        data=data,
        method=method,
        headers=headers,
    )
    try:
        response = urllib.request.urlopen(request, timeout=5)
    except urllib.error.HTTPError as exc:
        response = exc
    with response:
        raw = response.read()
        body = json.loads(raw) if raw else None
        return response.status, body


def _csv_text(fieldnames, rows):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _write_once(path: Path, content: str):
    """Atomically create a dispatch file, accepting byte-identical retries."""
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise ValueError(f"Existing {path.name} differs from this proposal; inspect it before retrying.")
        return False
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent,
                                         prefix=".cedos_", delete=False) as handle:
            temp_name = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        return True
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)


def _stem(campaign_id: str, proposal_id: str):
    digest = hashlib.sha256(f"{campaign_id}\0{proposal_id}".encode("utf-8")).hexdigest()[:12]
    return f"{campaign_id[:8]}_{proposal_id[:8]}_{digest}"


class FolderBridge:
    def __init__(self, server: str, proposals_dir: Path, results_dir: Path,
                 *, token: str | None = None, requester=None, logger=print):
        self.server = server
        self.proposals_dir = Path(proposals_dir)
        self.results_dir = Path(results_dir)
        self.proposals_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.requester = requester or (lambda method, path, payload=None: api_request(
            self.server, method, path, payload, token=token))
        self.log = logger
        self._reported_errors = set()

    def _request(self, method, path, payload=None):
        return self.requester(method, path, payload)

    def _report_once(self, key, message):
        if key not in self._reported_errors:
            self._reported_errors.add(key)
            self.log(message)

    def poll_once(self):
        okay = True
        try:
            self.process_proposal()
        except Exception as exc:
            okay = False
            self._report_once(("proposal", str(exc)), f"Proposal exchange: {exc}")
        try:
            _, result_errors = self.process_results()
            okay = okay and not result_errors
        except Exception as exc:
            okay = False
            self._report_once(("results", str(exc)), f"Result exchange: {exc}")
        return okay

    def process_proposal(self):
        status, proposal = self._request("GET", "/auto/proposal")
        if status == 204:
            return False
        if status != 200 or not isinstance(proposal, dict):
            raise RuntimeError(f"CEDOS returned HTTP {status} while fetching a proposal: {proposal}")
        campaign_id = proposal["campaign_id"]
        proposal_id = proposal["proposal_id"]
        stem = _stem(campaign_id, proposal_id)
        try:
            self._stage_proposal(proposal, stem)
        except Exception:
            # A failed local write must not consume the server proposal.
            try:
                self._request("POST", f"/auto/proposal/{proposal_id}/release",
                              {"campaign_id": campaign_id})
            except Exception:
                pass
            raise
        # If ACK fails after the file is written, the lease expires and the
        # same file is verified on retry, avoiding a second hardware dispatch.
        status, body = self._request("POST", f"/auto/proposal/{proposal_id}/ack",
                                     {"campaign_id": campaign_id})
        if status != 200:
            raise RuntimeError(f"CEDOS did not acknowledge proposal {proposal_id}: HTTP {status}, {body}")
        self.log(f"Proposal ready: {self.proposals_dir / ('Proposal_' + stem + '.csv')}")
        return True

    def _stage_proposal(self, proposal, stem):
        row_ids = proposal["row_ids"]
        feature_rows = proposal["features_only"]
        if len(row_ids) != len(feature_rows):
            raise ValueError("Proposal row IDs and feature rows differ in length.")
        objectives = proposal["objective_names"]
        columns = proposal["columns"]
        feature_names = [name for name in columns if name not in objectives and any(
            name in row for row in feature_rows)]
        if not feature_names:
            raise ValueError("Proposal contains no feature columns.")
        rows = []
        for row_id, feature_row in zip(row_ids, feature_rows):
            if feature_row.get("row_id", row_id) != row_id:
                raise ValueError("Proposal feature row ID mismatch.")
            rows.append({"row_id": row_id, **{name: feature_row.get(name, "")
                                               for name in feature_names}})
        csv_path = self.proposals_dir / f"Proposal_{stem}.csv"
        manifest_path = self.proposals_dir / f"Proposal_{stem}.json"
        result_name = f"Results_{stem}.csv"
        manifest = {
            "campaign_id": proposal["campaign_id"],
            "proposal_id": proposal["proposal_id"],
            "optimizer": proposal["optimizer"],
            "dataset_rev": proposal["dataset_rev"],
            "config_rev": proposal["config_rev"],
            "row_ids": row_ids,
            "objective_names": objectives,
            "columns": columns,
            "result_filename": result_name,
        }
        _write_once(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        _write_once(csv_path, _csv_text(["row_id", *feature_names], rows))
        example = self.results_dir / f"{result_name}.example"
        if not example.exists():
            _write_once(example, _csv_text(["row_id", *objectives], [
                {"row_id": row_id, **{name: "" for name in objectives}} for row_id in row_ids
            ]))

    def process_results(self):
        sent = 0
        errors = False
        for manifest_path in sorted(self.proposals_dir.glob("Proposal_*.json")):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                stem = _stem(manifest["campaign_id"], manifest["proposal_id"])
                if (manifest_path.name != f"Proposal_{stem}.json"
                        or manifest.get("result_filename") != f"Results_{stem}.csv"):
                    raise ValueError(f"{manifest_path.name} has inconsistent proposal identity.")
                result_path = self.results_dir / manifest["result_filename"]
                if not result_path.is_file():
                    continue
                rows = self._read_results(result_path, manifest)
                payload = {
                    "campaign_id": manifest["campaign_id"],
                    "proposal_id": manifest["proposal_id"],
                    "rows": rows,
                }
                status, body = self._request("POST", "/auto/results", payload)
                if status != 200:
                    raise RuntimeError(f"CEDOS rejected {result_path.name}: HTTP {status}, {body}")
                sent_path = self.results_dir / f"SENT_{result_path.name}"
                if sent_path.exists():
                    if sent_path.read_bytes() != result_path.read_bytes():
                        raise ValueError(f"{sent_path.name} already exists with different measurements.")
                    result_path.unlink()
                else:
                    os.replace(result_path, sent_path)
                self.log(f"Measurements accepted: {sent_path}")
                sent += 1
            except Exception as exc:
                errors = True
                self._report_once((str(manifest_path), str(exc)), f"Result exchange: {exc}")
        return sent, errors

    @staticmethod
    def _read_results(path, manifest):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            headers = reader.fieldnames or []
            if len(headers) != len(set(headers)):
                raise ValueError(f"{path.name} repeats a CSV column.")
            if "row_id" not in headers or any(name not in headers for name in manifest["objective_names"]):
                raise ValueError(f"{path.name} needs row_id and every objective column.")
            if any(name not in set(manifest["columns"]) | {"row_id"} for name in headers):
                raise ValueError(f"{path.name} contains an unknown dataset column.")
            rows = list(reader)
        expected = manifest["row_ids"]
        actual = [row.get("row_id") for row in rows]
        if len(rows) != len(expected) or len(set(actual)) != len(actual) or set(actual) != set(expected):
            raise ValueError(f"{path.name} row IDs must match its proposal exactly.")
        for row in rows:
            if None in row:
                raise ValueError(f"{path.name} has a row with extra CSV fields.")
            for name in manifest["objective_names"]:
                try:
                    if not math.isfinite(float(row[name])):
                        raise ValueError
                except (TypeError, ValueError):
                    raise ValueError(f"{path.name} has a missing or nonnumeric {name} value.") from None
        return rows


def main(argv=None):
    folder = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="CEDOS local folder exchange")
    parser.add_argument("--server", default="http://127.0.0.1:5000")
    parser.add_argument("--token-file", type=Path,
                        help="Read the hosted Auto exchange token from a local file")
    parser.add_argument("--proposals-dir", type=Path, default=folder / "proposals")
    parser.add_argument("--results-dir", type=Path, default=folder / "results")
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--once", action="store_true", help="Check CEDOS and result files once, then exit")
    args = parser.parse_args(argv)
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    if args.token_file:
        try:
            token = args.token_file.read_text(encoding="utf-8").strip()
        except OSError as exc:
            parser.error(f"Unable to read token file: {exc}")
        if not token:
            parser.error("The hosted Auto token file is empty")
    else:
        token = os.environ.get("CEDOS_AUTO_TOKEN", "").strip() or None
    bridge = FolderBridge(args.server, args.proposals_dir, args.results_dir, token=token)
    print(f"CEDOS folder exchange: {args.server}")
    print(f"Proposals: {bridge.proposals_dir}")
    print(f"Results:   {bridge.results_dir}")
    print("Fill a Results_*.csv.example file and save it without .example to upload measurements.")
    if args.once:
        return 0 if bridge.poll_once() else 1
    try:
        while True:
            bridge.poll_once()
            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        print("Folder exchange stopped.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
