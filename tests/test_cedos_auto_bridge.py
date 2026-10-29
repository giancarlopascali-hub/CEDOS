"""Identified local Auto/IMEX handoff and folder transport regression checks."""

import csv
from pathlib import Path

import pytest

from app import app
from imex.cedos_imex import FolderBridge
from src.api.auto_routes import _reset_auto_state


@pytest.fixture
def client():
    _reset_auto_state()
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client
    _reset_auto_state()


def proposal(campaign="campaign-1", batch="proposal-1"):
    return {
        "campaign_id": campaign,
        "proposal_id": batch,
        "optimizer": "tbo",
        "dataset_rev": 3,
        "config_rev": 4,
        "columns": ["Temperature", "Yield"],
        "objective_names": ["Yield"],
        "suggestions": [{"Temperature": 25}, {"Temperature": 35}],
        "features_only": [{"Temperature": 25}, {"Temperature": 35}],
        "rows": [
            {"row_id": "row-1", "Temperature": 25, "Yield": ""},
            {"row_id": "row-2", "Temperature": 35, "Yield": ""},
        ],
    }


def results(campaign="campaign-1", batch="proposal-1", first="0.4"):
    return {
        "campaign_id": campaign,
        "proposal_id": batch,
        "rows": [
            {"row_id": "row-2", "Yield": "0.8"},
            {"row_id": "row-1", "Yield": first},
        ],
    }


@pytest.mark.parametrize("method,path", [
    ("GET", "/auto/proposal"),
    ("POST", "/auto/proposal"),
    ("POST", "/auto/proposal/proposal-1/ack"),
    ("POST", "/auto/proposal/proposal-1/release"),
    ("GET", "/auto/results"),
    ("POST", "/auto/results"),
    ("POST", "/auto/results/proposal-1/ack"),
    ("POST", "/auto/stop"),
    ("GET", "/auto/status"),
])
def test_hosted_auto_requires_configured_token(client, monkeypatch, method, path):
    monkeypatch.setenv("SPACE_ID", "realgcp/CEDOS")
    monkeypatch.delenv("CEDOS_AUTO_TOKEN", raising=False)
    assert client.open(path, method=method).status_code == 503
    monkeypatch.setenv("CEDOS_AUTO_TOKEN", "a-long-owner-controlled-space-token")
    assert client.open(path, method=method).status_code == 401
    assert client.open(path, method=method, headers={
        "Authorization": "Bearer a-wrong-token"
    }).status_code == 401


def test_hosted_auto_token_allows_exchange_but_not_needed_for_export(client, monkeypatch):
    monkeypatch.setenv("SPACE_ID", "realgcp/CEDOS")
    monkeypatch.setenv("CEDOS_AUTO_TOKEN", "a-long-owner-controlled-space-token")
    headers = {"Authorization": "Bearer a-long-owner-controlled-space-token"}
    assert client.post("/auto/proposal", json=proposal(), headers=headers).status_code == 200
    fetched = client.get("/auto/proposal", headers=headers)
    assert fetched.status_code == 200
    assert fetched.get_json()["proposal_id"] == "proposal-1"
    assert client.get("/auto/status", headers=headers).get_json()["status"] == "leased"
    assert client.post("/export", json={"columns": ["x"], "data": [[1]]}).status_code == 200


def test_folder_round_trip_preserves_identity_and_requires_merge_ack(client, tmp_path):
    proposed = proposal()
    assert client.post("/auto/proposal", json=proposed).status_code == 200

    def requester(method, path, payload=None):
        response = client.open(path, method=method, json=payload)
        return response.status_code, response.get_json(silent=True)

    logs = []
    bridge = FolderBridge("http://unused", tmp_path / "proposals", tmp_path / "results",
                          requester=requester, logger=logs.append)
    bridge.poll_once()
    assert client.get("/auto/status?campaign_id=campaign-1").get_json()["status"] == "dispatched"
    proposal_csv = next((tmp_path / "proposals").glob("Proposal_*.csv"))
    with proposal_csv.open(newline="", encoding="utf-8") as handle:
        exported = list(csv.DictReader(handle))
    assert [row["row_id"] for row in exported] == ["row-1", "row-2"]
    assert [row["Temperature"] for row in exported] == ["25", "35"]

    example = next((tmp_path / "results").glob("Results_*.csv.example"))
    result_path = example.with_suffix("")
    with result_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "Yield"])
        writer.writeheader()
        writer.writerows(results()["rows"])
    bridge.poll_once()
    assert not result_path.exists()
    assert (tmp_path / "results" / f"SENT_{result_path.name}").exists()

    fetched = client.get("/auto/results?campaign_id=campaign-1&proposal_id=proposal-1")
    assert fetched.status_code == 200
    assert [row["row_id"] for row in fetched.get_json()["rows"]] == ["row-1", "row-2"]
    assert client.post("/auto/proposal", json=proposal(batch="proposal-2")).status_code == 409
    assert client.post("/auto/results/proposal-1/ack",
                       json={"campaign_id": "campaign-1"}).status_code == 200
    assert client.post("/auto/proposal", json=proposal(batch="proposal-2")).status_code == 200
    assert client.post("/auto/results", json=results()).get_json()["duplicate"] is True
    assert client.post("/auto/results", json=results(first="0.5")).status_code == 409


def test_lease_release_and_retry_do_not_lose_proposal(client):
    assert client.post("/auto/proposal", json=proposal()).status_code == 200
    first = client.get("/auto/proposal")
    assert first.status_code == 200
    assert first.get_json()["row_ids"] == ["row-1", "row-2"]
    assert client.get("/auto/proposal").status_code == 204
    assert client.post("/auto/proposal/proposal-1/release",
                       json={"campaign_id": "campaign-1"}).status_code == 200
    assert client.get("/auto/proposal").status_code == 200
    assert client.post("/auto/proposal/proposal-1/ack",
                       json={"campaign_id": "campaign-1"}).status_code == 200
    assert client.get("/auto/proposal").status_code == 204


def test_cross_campaign_stale_and_conflicting_measurements_are_rejected(client):
    mismatched = proposal()
    mismatched["rows"][0]["Temperature"] = 30
    assert client.post("/auto/proposal", json=mismatched).status_code == 400
    blank_feature = proposal()
    blank_feature["features_only"][0]["Temperature"] = ""
    assert client.post("/auto/proposal", json=blank_feature).status_code == 400
    assert client.post("/auto/proposal", json=proposal()).status_code == 200
    assert client.post("/auto/proposal", json=proposal(campaign="other")).status_code == 409
    assert client.post("/auto/results", json=results()).status_code == 409
    assert client.post("/auto/proposal/proposal-1/ack",
                       json={"campaign_id": "campaign-1"}).status_code == 200
    assert client.post("/auto/results", json=results(campaign="other")).status_code == 409
    incomplete = results()
    incomplete["rows"][0]["Yield"] = ""
    assert client.post("/auto/results", json=incomplete).status_code == 400
    changed_feature = results()
    changed_feature["rows"][0]["Temperature"] = 99
    assert client.post("/auto/results", json=changed_feature).status_code == 400
    assert client.post("/auto/results", json=results()).status_code == 200
    assert client.post("/auto/results", json=results()).get_json()["duplicate"] is True
    assert client.post("/auto/results", json=results(first="0.5")).status_code == 409
    assert client.get("/auto/results?campaign_id=other&proposal_id=proposal-1").status_code == 409
    assert client.post("/auto/results/proposal-1/ack",
                       json={"campaign_id": "campaign-1"}).status_code == 200
    assert client.post("/auto/stop", json={"campaign_id": "campaign-1"}).status_code == 200
    assert client.post("/auto/stop", json={"campaign_id": "campaign-1"}).status_code == 200
    assert client.post("/auto/proposal", json=proposal(campaign="other")).status_code == 200


def test_folder_keeps_bad_result_file_for_correction(client, tmp_path):
    client.post("/auto/proposal", json=proposal())

    def requester(method, path, payload=None):
        response = client.open(path, method=method, json=payload)
        return response.status_code, response.get_json(silent=True)

    logs = []
    bridge = FolderBridge("http://unused", tmp_path / "proposals", tmp_path / "results",
                          requester=requester, logger=logs.append)
    bridge.poll_once()
    example = next((tmp_path / "results").glob("Results_*.csv.example"))
    result_path = Path(str(example).removesuffix(".example"))
    result_path.write_text("row_id,Yield\nrow-1,0.5\nwrong-row,0.8\n", encoding="utf-8")
    assert bridge.poll_once() is False
    assert result_path.exists()
    assert any("row IDs must match" in line for line in logs)
    assert client.get("/auto/results?campaign_id=campaign-1&proposal_id=proposal-1").status_code == 202
