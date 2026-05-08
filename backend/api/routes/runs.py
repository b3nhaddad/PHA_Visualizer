"""
Run lifecycle endpoints.

POST  /api/runs                          → start a new run
GET   /api/runs/{run_id}/status          → poll run state
GET   /api/runs/{run_id}/batches         → pull ready batches

In-memory store is fine for a single-worker dev server.
For production replace _RUNS with Redis or a database.
"""
from __future__ import annotations

import math
import threading
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from ..models import (
    BatchesResponse,
    RunStatusResponse,
    SnapshotBatch,
    StartRunRequest,
    StartRunResponse,
)

router = APIRouter(prefix="/runs")

# ── In-memory state ────────────────────────────────────────────────

_RUNS: dict[str, dict[str, Any]] = {}
# Shape of each entry:
# {
#   "status":        "queued" | "running" | "complete" | "failed",
#   "batches":       list[SnapshotBatch],
#   "request":       StartRunRequest,
#   "error":         str | None,
# }


# ── Helpers ────────────────────────────────────────────────────────

def _batch_size(N: int) -> int:
    return max(1, int(200 / math.sqrt(N)))


def _run_algorithm(run_id: str, request: StartRunRequest) -> None:
    """
    Worker thread: call the algorithm and store batches as they arrive.

    TODO: replace the stub body with a call to your algorithm module, e.g.
        from algorithm.subag import run as subag_run
        for batch in subag_run(request):
            _RUNS[run_id]["batches"].append(batch)
    """
    entry = _RUNS[run_id]
    entry["status"] = "running"
    try:
        from algorithm.subag import run as subag_run
        for batch in subag_run(request):
            entry["batches"].append(batch)

        entry["status"] = "complete"
    except Exception as exc:
        entry["status"] = "failed"
        entry["error"] = str(exc)


# ── Endpoints ──────────────────────────────────────────────────────

@router.post("", response_model=StartRunResponse, status_code=201)
def start_run(request: StartRunRequest) -> StartRunResponse:
    run_id = str(uuid.uuid4())
    _RUNS[run_id] = {"status": "queued", "batches": [], "request": request, "error": None}
    thread = threading.Thread(target=_run_algorithm, args=(run_id, request), daemon=True)
    thread.start()
    return StartRunResponse(run_id=run_id)


@router.get("/{run_id}/status", response_model=RunStatusResponse)
def get_status(run_id: str) -> RunStatusResponse:
    entry = _RUNS.get(run_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")
    return RunStatusResponse(
        run_id=run_id,
        status=entry["status"],
        batches_ready=len(entry["batches"]),
        error=entry.get("error"),
    )


@router.get("/{run_id}/batches", response_model=BatchesResponse)
def get_batches(
    run_id: str,
    from_batch: int = Query(default=0, alias="from_batch", ge=0),
) -> BatchesResponse:
    entry = _RUNS.get(run_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")
    batches = entry["batches"][from_batch:]
    return BatchesResponse(
        run_id=run_id,
        batches=batches,
        is_final=entry["status"] == "complete" and bool(batches) and batches[-1].is_final,
    )
