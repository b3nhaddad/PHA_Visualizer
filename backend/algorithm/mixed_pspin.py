"""
Mixed p-spin spherical glass — algorithm wrapper.

Physics reference
-----------------
Hamiltonian:   H_N(σ) = Σ_p β_p H_p(σ)
               Each H_p is the pure p-spin Hamiltonian with its own disorder tensor
               and coupling constant β_p.

Covariance:    ξ(q)   = Σ_p β_p² q^p
Second deriv:  ξ''(q) = Σ_p β_p² p(p-1) q^(p-2)
Spectral edge: -2√ξ''(q)
E_∞:           ∫₀¹ √ξ''(t) dt   (numerical integration)

Hessian (sum of component Hessians):
  For each (p, β_p) component, add the p-spin contribution:
    p=2: β · J_{ab}/√N                     (constant)
    p=3: β · (1/N) Σ_k J_{abk} σ_k        (linear in σ)
    p≥4: extend the pattern (each needs its own disorder tensor)

What you need to implement
--------------------------
1. Disorder tensor builders for each p in the mixture (_make_J2, _make_J3, ...)
2. _xi_pp(components, q)       — ξ''(q) = Σ β² p(p-1) q^(p-2)
3. _spectral_edge(components, q) — -2√ξ''(q)
4. _E_inf(components)          — ∫₀¹ √ξ''(t) dt  (numerical)
5. hessian_fn                  — sum of component Hessians at σ
6. energy_fn                   — sum of component energies at σ
7. Your mixed descent algorithm call
8. Fill in E_inf and the prediction strings

model.components is a list of objects with:
  .p     — spin order (int)
  .beta  — coupling strength (float)

See ALGORITHM_GUIDE.md for the full contract, trajectory format, and
step-by-step walkthrough.
"""
from __future__ import annotations

import math

import torch

from algorithm.protocol import AlgorithmResult
from api.models import MixedSpinSpec, RunPredictions
import algorithm.p3 as _p3_module
from algorithm.p3 import spatial_hessian_descentp3
from algorithm.pure_p2 import _make_disorder
from algorithm.pure_p3 import _make_disorder_p3

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#step 0


# ── Step 1a: Disorder tensor for p=2 ──────────────────────────────────
#
# N×N symmetric Gaussian, zero diagonal.
# Signature to keep:  _make_J2(N, seed) -> torch.Tensor
#
def _make_J2(N: int, seed: int) -> torch.Tensor:
    J = _make_disorder(N, seed)
    return J.to(_device)



# ── Step 1b: Disorder tensor for p=3 ──────────────────────────────────
#
# N×N×N fully symmetric Gaussian.
# Signature to keep:  _make_J3(N, seed) -> torch.Tensor
#
def _make_J3(N: int, seed: int) -> torch.Tensor:
    J = _make_disorder_p3(N, seed)
    return J.to(_device)

#def _make_J4(N: int, seed: int) -> torch.Tensor:


# ── Add _make_J4, _make_J5, ... here if you need higher p ─────────────


# ── Step 2: ξ''(q) ────────────────────────────────────────────────────
#
# ξ''(q) = Σ_p β_p² · p·(p-1) · q^(p-2)
# `components` is model.components — each element has .p and .beta
#
def _xi_pp(components: list, q: float) -> float:
    return sum(c.beta ** 2 * c.p * (c.p - 1) * q ** (c.p - 2) for c in components)
    #raise NotImplementedError("TODO: implement ξ''(q) = Σ β² p(p-1) q^(p-2)")


# ── Step 3: Spectral edge ──────────────────────────────────────────────
#
# λ_edge(q) = -2·√ξ''(q)   (always negative)
#
def _spectral_edge(components: list, q: float) -> float:
    return -2.0 * math.sqrt(max(_xi_pp(components, q), 1e-12))
    #raise NotImplementedError("TODO: return -2.0 * math.sqrt(max(_xi_pp(components, q), 1e-12))")


# ── Step 4: E_∞ ───────────────────────────────────────────────────────
#
# E_∞ = ∫₀¹ √ξ''(t) dt   — integrate numerically.
# Simple midpoint rule with n_pts=1000 is sufficient.
#
def _E_inf(components: list, n_pts: int = 1000) -> float:
    return sum(math.sqrt(max(_xi_pp(components, (i + 0.5) / n_pts), 0)) * (1 / n_pts) for i in range(n_pts))
    '''''raise NotImplementedError(
        "TODO: sum sqrt(max(_xi_pp(components, (i+0.5)/n_pts), 0)) * (1/n_pts) for i in range(n_pts)"
    )''''


# ── Entry point (called by registry.py → subag.py → HTTP) ─────────────
#
# Do NOT change the signature or return type.
#
def build(model: MixedSpinSpec, k: int) -> AlgorithmResult:
    N          = model.N
    seed       = model.seed if model.seed is not None else 42
    components = model.components   # list of {.p, .beta}

    # ── Step 5: Build one disorder tensor per unique p in this mixture ──
    disorder: dict[int, torch.Tensor] = {}
    for i, c in enumerate(components):
        if c.p not in disorder:
            if c.p == 2:
                disorder[2] = _make_J2(N, seed + i)
            elif c.p == 3:
                disorder[3] = _make_J3(N, seed + i)
            else:
                raise NotImplementedError(f"p={c.p} disorder tensor not implemented")

    # ── Step 6: Define hessian_fn ───────────────────────────────────────
    if 3 in disorder:
        _p3_module.J = disorder[3]

    def hessian_fn(sigma: torch.Tensor) -> torch.Tensor:
        H = torch.zeros(N, N, device=_device)
        for c in components:
            if c.p == 2:
                H = H + c.beta * disorder[2] / math.sqrt(N)
            elif c.p == 3:
                H = H + c.beta * torch.einsum("ijk,k->ij", disorder[3], sigma) / N
        return H

    # ── Step 7: Define energy_fn ────────────────────────────────────────
    def energy_fn(sigma: torch.Tensor) -> float:
        e = 0.0
        for c in components:
            if c.p == 2:
                e += c.beta * float(sigma @ (disorder[2] @ sigma)) / (2.0 * N * math.sqrt(N))
            elif c.p == 3:
                e += c.beta * float(torch.einsum("ijk,i,j,k->", disorder[3], sigma, sigma, sigma)) / (6.0 * N * N)
        return e

    # ── Step 8: Define spectral edge callable ───────────────────────────
    edge_fn = lambda q: _spectral_edge(components, q)

    # ── Step 9: Run the mixed descent algorithm ──────────────────────────
    if 3 in disorder:
        trajectory = spatial_hessian_descentp3(
            step_size      = 1.0 / k,
            dimensionality = N,
            start_position = torch.zeros(N, device=_device),
            hessian_fn     = hessian_fn,
            n_steps        = k,
            J = disorder[3],
        )
    else:
        trajectory = spatial_hessian_descentp3(
            step_size=1.0 / k,
            dimensionality=N,
            start_position=torch.zeros(N, device=_device),
            hessian_fn=hessian_fn,
            n_steps=k,
        )

    # ── Step 10: Compute E_∞ and assemble the result ───────────────────
    #
    E_inf   = _E_inf(components)
    xi_str  = " + ".join(f"{c.beta:.2f}²·q^{c.p}" for c in components)

    return AlgorithmResult(
        trajectory=trajectory,
        hessian_fn=hessian_fn,
        energy_fn=energy_fn,
        spectral_edge_fn=edge_fn,
        predictions=RunPredictions(
            E_infinity=E_inf,
            E_infinity_formula=f"∫₀¹ √ξ''(t)dt ≈ {E_inf:.6f}",
            spectral_edge_formula=f"-2√ξ''(q),  ξ''(q) = {xi_str}",
        ),
    )