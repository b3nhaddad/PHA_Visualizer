"""
Public entry point called by api/routes/runs.py.

Dispatches to the registry, builds snapshots, and yields SnapshotBatch objects.
Nothing model-specific lives here — add algorithms in registry.py instead.
"""
from __future__ import annotations

import math
from typing import Generator

from algorithm.registry import dispatch
from algorithm.snapshot import build_all_snapshots
from api.models import SnapshotBatch, StartRunRequest


def run(request: StartRunRequest) -> Generator[SnapshotBatch, None, None]:
    model = request.model
    N = model.N
    k = request.algorithm.n_steps
    seed = model.seed if model.seed is not None else 42
    run_id = f"pha_{model.type}_N{N}_k{k}_seed{seed}"

    result = dispatch(model, k)
    snapshots = build_all_snapshots(result, k, N)

    batch_size = max(1, int(200 / math.sqrt(N)))
    i = 0
    batch_idx = 0
    while i < k:
        end = min(i + batch_size - 1, k - 1)
        yield SnapshotBatch(
            run_id=run_id,
            batch_index=batch_idx,
            step_range={"start": i, "end": end},
            total_steps=k,
            is_final=(end == k - 1),
            model=model,
            predictions=result.predictions,
            snapshots=snapshots[i : end + 1],
        )
        batch_idx += 1
        i = end + 1
