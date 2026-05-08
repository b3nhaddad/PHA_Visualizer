"""
Pure p=2 spherical spin glass — algorithm wrapper.

Physics reference
-----------------
Hamiltonian:   H_N(σ) = (1/2√N) σᵀJσ
               J ~ GOE: symmetric Gaussian, zero diagonal
Hessian:       ∇²H_N = J/√N   (constant — does NOT depend on σ)
E_∞:           2√((p-1)/p) = √2 ≈ 1.4142
Spectral edge: -2√(p(p-1)) = -2√2   (constant for p=2, independent of q)

What you need to implement
--------------------------
1. _make_disorder  — sample J from the GOE
2. hessian_fn      — return the constant matrix J/√N
3. energy_fn       — return H_N(σ)/N as a Python float
4. spectral_edge_fn — return -2√(p(p-1))·q^((p-2)/2)
5. Call spatial_hessian_descent_p2 with the right arguments
6. Fill in E_inf and the prediction strings

See ALGORITHM_GUIDE.md for the full contract, trajectory format, and
step-by-step walkthrough.
"""
from __future__ import annotations

import math

import torch

from algorithm.p2 import spatial_hessian_descent_p2
from algorithm.protocol import AlgorithmResult
from api.models import PureSpinSpec, RunPredictions

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_P = 2


# ── Step 1: Sample the disorder tensor ────────────────────────────────
#
# J must be an N×N symmetric Gaussian matrix with zero diagonal (GOE).
# Use `seed` so runs are reproducible.
#
# Signature to keep:  _make_disorder(N: int, seed: int) -> torch.Tensor
#
def _make_disorder(N: int, seed: int) -> torch.Tensor:
    raise NotImplementedError(
        "TODO: sample J ~ GOE(N).  "
        "Hint: randn(N,N), symmetrize with (J+J.T)/2, zero the diagonal."
    )


# ── Entry point (called by registry.py → subag.py → HTTP) ─────────────
#
# Do NOT change the signature or return type.
# The infrastructure calls build(model, k) and expects an AlgorithmResult.
#
def build(model: PureSpinSpec, k: int) -> AlgorithmResult:
    N    = model.N
    seed = model.seed if model.seed is not None else 42

    # ── Step 2: Build disorder and the constant Hessian matrix ─────────
    #
    # J  = _make_disorder(N, seed)
    # H_const = J / sqrt(N)    ← this is the full N×N Hessian (constant for p=2)
    #
    raise NotImplementedError("TODO: call _make_disorder and compute H_const")

    # ── Step 3: Define the three physics callables ──────────────────────
    #
    # hessian_fn(sigma) -> N×N torch.Tensor
    #   For p=2 the Hessian is constant (ignores sigma).
    #   Return H_const.
    #
    # energy_fn(sigma) -> float
    #   H_N(σ)/N = σᵀJσ / (2·N·√N)
    #
    # edge_fn(q) -> float   (always negative)
    #   Spectral edge of the projected Hessian at overlap q.
    #   For p=2: -2·√(p(p-1)) = -2√2  (independent of q)
    #   General formula: -2·√(p(p-1)) · q^((p-2)/2)
    #
    hessian_fn = None   # replace with your lambda / function
    energy_fn  = None
    edge_fn    = None

    # ── Step 4: Run the descent algorithm ──────────────────────────────
    #
    # spatial_hessian_descent_p2 lives in algorithm/p2.py (do not edit it).
    # It returns a list of k+1 torch.Tensor vectors:
    #   trajectory[0]   = start_position (zeros on the sphere)
    #   trajectory[1]   = position after step 1
    #   ...
    #   trajectory[k]   = final position
    #
    # trajectory = spatial_hessian_descent_p2(
    #     step_size      = 1.0 / k,
    #     dimensionality = N,
    #     start_position = torch.zeros(N, device=_device),
    #     hessian_fn     = hessian_fn,
    #     n_steps        = k,
    # )
    #
    trajectory = None   # replace with the actual call

    # ── Step 5: Compute E_∞ and assemble the result ────────────────────
    #
    # E_inf = 2·√((p-1)/p)
    # Fill the formula strings — they appear in the UI.
    #
    E_inf = None   # replace: 2.0 * math.sqrt((_P - 1) / _P)

    return AlgorithmResult(
        trajectory=trajectory,
        hessian_fn=hessian_fn,
        energy_fn=energy_fn,
        spectral_edge_fn=edge_fn,
        predictions=RunPredictions(
            E_infinity=E_inf,
            E_infinity_formula=f"2*sqrt((p-1)/p) = ???  [p={_P}]",
            spectral_edge_formula=f"-2*sqrt(p*(p-1)) = ???  [p={_P}, constant]",
        ),
    )
