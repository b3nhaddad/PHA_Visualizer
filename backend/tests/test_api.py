"""Integration tests for the run lifecycle endpoints."""
import time

import pytest
from fastapi.testclient import TestClient


def _wait_complete(client: TestClient, run_id: str, timeout: float = 10.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/api/runs/{run_id}/status")
        data = r.json()
        if data["status"] in ("complete", "failed"):
            return data
        time.sleep(0.1)
    pytest.fail(f"Run {run_id} did not complete within {timeout}s")


def test_health(client: TestClient) -> None:
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_start_run_pure(client: TestClient, pure_request: dict) -> None:
    r = client.post("/api/runs", json=pure_request)
    assert r.status_code == 201
    body = r.json()
    assert "run_id" in body
    assert body["status"] == "queued"


def test_start_run_mixed(client: TestClient, mixed_request: dict) -> None:
    r = client.post("/api/runs", json=mixed_request)
    assert r.status_code == 201


def test_run_lifecycle_pure(client: TestClient, pure_request: dict) -> None:
    run_id = client.post("/api/runs", json=pure_request).json()["run_id"]
    status = _wait_complete(client, run_id)
    assert status["status"] == "complete"

    r = client.get(f"/api/runs/{run_id}/batches?from_batch=0")
    assert r.status_code == 200
    data = r.json()
    assert len(data["batches"]) > 0
    assert data["is_final"] is True

    # Verify schema_version on every batch
    for batch in data["batches"]:
        assert batch["schema_version"] == "0.2.0"
        assert batch["run_id"] == run_id
        assert len(batch["snapshots"]) > 0


def test_run_lifecycle_mixed(client: TestClient, mixed_request: dict) -> None:
    run_id = client.post("/api/runs", json=mixed_request).json()["run_id"]
    status = _wait_complete(client, run_id)
    assert status["status"] == "complete"

    batches = client.get(f"/api/runs/{run_id}/batches").json()["batches"]
    steps = sum(len(b["snapshots"]) for b in batches)
    assert steps == mixed_request["n_steps"]


def test_batches_from_offset(client: TestClient, pure_request: dict) -> None:
    run_id = client.post("/api/runs", json=pure_request).json()["run_id"]
    _wait_complete(client, run_id)

    all_batches = client.get(f"/api/runs/{run_id}/batches?from_batch=0").json()["batches"]
    if len(all_batches) > 1:
        partial = client.get(f"/api/runs/{run_id}/batches?from_batch=1").json()["batches"]
        assert len(partial) == len(all_batches) - 1


def test_unknown_run_404(client: TestClient) -> None:
    r = client.get("/api/runs/does-not-exist/status")
    assert r.status_code == 404


def test_invalid_model_422(client: TestClient) -> None:
    r = client.post("/api/runs", json={"model": {"type": "unknown"}, "n_steps": 10})
    assert r.status_code == 422
