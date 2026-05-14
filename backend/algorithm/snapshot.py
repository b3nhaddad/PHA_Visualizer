"""
Generic snapshot builder — converts one trajectory position into a schema-v0.2.0 Snapshot.

This module knows nothing about which model or algorithm produced the trajectory.
It only needs the functions in AlgorithmResult.
"""
from __future__ import annotations

import math

import torch

from algorithm.protocol import AlgorithmResult
from api.models import ProofAnnotation, Snapshot

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_snapshot(
    j: int,
    k: int,
    sigma: torch.Tensor,
    v: torch.Tensor,               # approximate step direction (normalized delta)
    result: AlgorithmResult,
    N: int,
    final_sigma: torch.Tensor,
) -> Snapshot:
    q = (j + 1) / k

    # ── Projected Hessian eigenvalues ──────────────────────────────
    H = result.hessian_fn(sigma)
    norm_sq = float(sigma @ sigma)
    if norm_sq > 1e-12:
        proj = torch.eye(N, device=_device) - torch.outer(sigma, sigma) / norm_sq
        H_proj = proj @ H @ proj
    else:
        H_proj = H
    eigvals = torch.linalg.eigvalsh(H_proj)   # ascending order

    lam_min = float(eigvals[0])
    edge_val = result.spectral_edge_fn(q)

    # ── Derived scalars ────────────────────────────────────────────
    v_nrm = float(torch.linalg.norm(v))
    v_unit = (v / v_nrm if v_nrm > 1e-12 else v).tolist()
    ortho = abs(float(sigma @ v)) / max(math.sqrt(norm_sq), 1e-12)

    eigvals_list = [round(float(x), 6) for x in eigvals]

    return Snapshot(
        index=j, j=j,
        q=round(q, 6),
        radius=round(math.sqrt(N * q), 6),
        sigma=[round(x, 6) for x in sigma.tolist()],
        v_chosen=[round(x, 6) for x in v_unit],
        energy_density=round(float(result.energy_fn(sigma)), 6),
        lambda_min_tangent=round(lam_min, 6),
        spectral_edge_target=round(edge_val, 6),
        edge_target_satisfied=lam_min <= edge_val + 0.01,
        grad_norm_tangent=0.0,
        step_orthogonality_residual=round(ortho, 8),
        eigvals_topK=eigvals_list[:20],
        full_spectrum=(eigvals_list if j % 5 == 0 else None),
        overlap_with_start=round(float(sigma[0]) / math.sqrt(N), 6),
        overlap_with_final=round(float(sigma @ final_sigma) / N, 6),
        proof_annotation=ProofAnnotation(
            active_lemma="lemma_3",
            tag="lemma_3_spectral_edge",
            human_readable=(
                f"λ_min={lam_min:.4f} vs edge={edge_val:.4f} — "
                f"{'satisfied' if lam_min <= edge_val + 0.01 else 'VIOLATED'}"
            ),
            katex=r"\#\{i:\lambda_i\le-2\sqrt{\xi''(q)}+\varepsilon\}\ge N\delta",
        ),
    )


def build_all_snapshots(result: AlgorithmResult, k: int, N: int) -> list[Snapshot]:
    """Convert a full trajectory into a list of k Snapshot objects."""
    trajectory = result.trajectory
    final_sigma = trajectory[k]
    steps = trajectory[1:]          # skip trajectory[0] (start = zeros)

    # Approximate step direction from consecutive position deltas
    vs: list[torch.Tensor] = []
    for i in range(k):
        delta = steps[i] - trajectory[i]
        nrm = torch.linalg.norm(delta)
        vs.append(delta / nrm if nrm > 1e-12 else delta)

    return [
        build_snapshot(j, k, steps[j], vs[j], result, N, final_sigma)
        for j in range(k)
    ]
