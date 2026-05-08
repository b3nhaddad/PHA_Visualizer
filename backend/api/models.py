"""
Pydantic v2 models — mirror of frontend/src/api/types.ts.

Keep these two files in sync.  The TypeScript file is the source of truth for
field names; the Python file is the source of truth for validation rules.
"""
from __future__ import annotations

from typing import Annotated, Literal, Union
from pydantic import BaseModel, Field


# ── Model specs ────────────────────────────────────────────────────

class PureSpinSpec(BaseModel):
    type: Literal["pure_p_spin"]
    p: Annotated[int, Field(ge=2, le=10)]
    N: Annotated[int, Field(ge=1)]
    seed: int | None = None


class MixedComponent(BaseModel):
    p: Annotated[int, Field(ge=2, le=10)]
    beta: float


class MixedSpinSpec(BaseModel):
    type: Literal["mixed_p_spin"]
    N: Annotated[int, Field(ge=1)]
    components: Annotated[list[MixedComponent], Field(min_length=1)]
    seed: int | None = None


ModelSpec = Annotated[Union[PureSpinSpec, MixedSpinSpec], Field(discriminator="type")]


# ── Predictions / proof ────────────────────────────────────────────

class RunPredictions(BaseModel):
    E_infinity: float
    E_infinity_formula: str
    spectral_edge_formula: str


class ProofAnnotation(BaseModel):
    active_lemma: str
    tag: str
    human_readable: str
    katex: str


# ── Snapshot ───────────────────────────────────────────────────────

class Snapshot(BaseModel):
    index: int
    j: int
    q: float
    radius: float
    sigma: list[float]
    v_chosen: list[float]
    energy_density: float
    lambda_min_tangent: float
    spectral_edge_target: float
    edge_target_satisfied: bool
    grad_norm_tangent: float
    step_orthogonality_residual: float
    eigvals_topK: list[float]
    full_spectrum: list[float] | None = None
    overlap_with_start: float
    overlap_with_final: float
    proof_annotation: ProofAnnotation


# ── Batch ──────────────────────────────────────────────────────────

class SnapshotBatch(BaseModel):
    schema_version: Literal["0.2.0"] = "0.2.0"
    run_id: str
    batch_index: int
    step_range: dict[str, int]          # {"start": int, "end": int}
    total_steps: int
    is_final: bool
    model: ModelSpec
    predictions: RunPredictions
    snapshots: list[Snapshot]


# ── Request / response ─────────────────────────────────────────────

class StartRunAlgorithm(BaseModel):
    n_steps: Annotated[int, Field(ge=1, le=10_000)] = 100
    epsilon_spectral: float = 0.01
    batch_size: int | None = None


class StartRunRequest(BaseModel):
    model: ModelSpec
    algorithm: StartRunAlgorithm = StartRunAlgorithm()


class StartRunResponse(BaseModel):
    run_id: str
    status: str = "queued"


class RunStatusResponse(BaseModel):
    run_id: str
    status: Literal["queued", "running", "complete", "failed"]
    batches_ready: int = 0
    error: str | None = None


class BatchesResponse(BaseModel):
    run_id: str
    batches: list[SnapshotBatch]
    is_final: bool
