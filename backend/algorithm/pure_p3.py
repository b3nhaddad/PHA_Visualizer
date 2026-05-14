"""
Pure p=3 spherical spin glass — algorithm wrapper.

Physics reference
-----------------
Hamiltonian:   H_N(σ) = (1/6N) Σ_{ijk} J_{ijk} σ_i σ_j σ_k
               J ~ fully symmetric Gaussian rank-3 tensor
Hessian:       (∇²H_N)_{ab} = (1/N) Σ_k J_{abk} σ_k   (linear in σ — recomputed each step)
E_∞:           2√((p-1)/p) = 2√(2/3) ≈ 1.6330
Spectral edge: -2√(p(p-1)) · q^((p-2)/2) = -2√6 · √q

What you need to implement
--------------------------
1. _make_disorder  — sample J (N×N×N fully symmetric)
2. hessian_fn      — contract J with σ: (1/N) Σ_k J_{abk} σ_k
3. energy_fn       — return H_N(σ)/N as a Python float
4. spectral_edge_fn — return -2√(p(p-1)) · q^((p-2)/2)
5. Inject J into p3.py module global (required — see note below)
6. Call spatial_hessian_descentp3 with the right arguments
7. Fill in E_inf and the prediction strings

IMPORTANT — J injection
-----------------------
p3.py (your partner's code) references a bare `J` global on line 68
inside the descent function.  Before you call spatial_hessian_descentp3
you MUST inject the tensor:

    import algorithm.p3 as _p3_module
    _p3_module.J = J        # ← this line, every time before calling

This is already imported at the top of this file.  Do NOT remove it.

See ALGORITHM_GUIDE.md for the full contract, trajectory format, and
step-by-step walkthrough.
"""
from __future__ import annotations

import math

import torch

import algorithm.p3 as _p3_module          # needed for J injection — do not remove
from algorithm.p3 import spatial_hessian_descentp3
from algorithm.protocol import AlgorithmResult
from api.models import PureSpinSpec, RunPredictions

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_P = 3


# ── Step 1: Sample the disorder tensor ────────────────────────────────
#
# J must be an N×N×N fully symmetric Gaussian tensor.
# "Fully symmetric" means J_{ijk} = J_{ikj} = J_{jik} = ... for all permutations.
#
# Construction recipe:
#   1. Sample raw = randn(N, N, N)
#   2. Average over all 6 permutations of the 3 indices to symmetrize
#   3. Move to _device
#
# Signature to keep:  _make_disorder(N: int, seed: int) -> torch.Tensor
#
def _make_disorder_p3(N: int, seed: int) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed)
    J = torch.randn(N, N, N, generator=g)

    J = (J + J.permute(0, 2, 1) + J.permute(1, 0, 2) +
         J.permute(1, 2, 0) + J.permute(2, 0, 1) +
         J.permute(2, 1, 0)) / 6

    J = J * 6.0  # symmetrized randn has variance 1/6; need variance 6 for spectral edge -2√6√q
    return J.to(_device)


# ── Entry point (called by registry.py → subag.py → HTTP) ─────────────
#
# Do NOT change the signature or return type.
#
def build(model: PureSpinSpec, k: int) -> AlgorithmResult:
    N    = model.N
    seed = model.seed if model.seed is not None else 42

    # ── Step 2: Build the disorder tensor ──────────────────────────────
    #
    J = _make_disorder_p3(N, seed)
    #

    # ── Step 2b: Inject J into p3.py's module global ───────────────────
    #
    # p3.py line 68 uses bare `J` — it must be set before calling the descent.
    # This line is REQUIRED every time you build:
    #
    # _p3_module.J = J
    #
    _p3_module.J = J

    # ── Step 3: Define the three physics callables ──────────────────────
    #
    # hessian_fn(sigma) -> N×N torch.Tensor
    #   H_{ab} = (1/N) Σ_k J_{abk} σ_k
    #   With einsum: torch.einsum("ijk,k->ij", J, sigma) / N
    #
    # energy_fn(sigma) -> float
    #   H_N(σ)/N = (1/6N²) Σ_{ijk} J_{ijk} σ_i σ_j σ_k
    #   With einsum: torch.einsum("ijk,i,j,k->", J, sigma, sigma, sigma) / (6.0 * N * N)
    #
    # edge_fn(q) -> float   (always negative)
    #   -2·√(p(p-1)) · q^((p-2)/2)
    #   For p=3: -2·√6 · √q
    #   General formula: -2·√(p(p-1)) · max(q, 1e-9)^((p-2)/2)
    hessian_fn = lambda sigma: torch.einsum("ijk,k->ij", J, sigma) / N

    energy_fn = lambda sigma: float(torch.einsum("ijk,i,j,k->", J, sigma, sigma, sigma)) / (6.0 * N * N)

    edge_fn = lambda q: -2.0 * math.sqrt(3 * 2) * max(q, 1e-9) ** 0.5

    # ── Step 4: Run the descent algorithm ──────────────────────────────
    #
    # spatial_hessian_descentp3 lives in algorithm/p3.py (do not edit it).
    # It returns a list of k+1 torch.Tensor vectors:
    #   trajectory[0]   = start_position (zeros on the sphere)
    #   trajectory[1]   = position after step 1
    #   ...
    #   trajectory[k]   = final position
    #
    # trajectory = spatial_hessian_descentp3(
    #     step_size      = 1.0 / k,
    #     dimensionality = N,
    #     start_position = torch.zeros(N, device=_device),
    #     hessian_fn     = hessian_fn,
    #     n_steps        = k,
    # )
    #
    trajectory = spatial_hessian_descentp3(
        step_size=1.0 / k,
        dimensionality=N,
        start_position=torch.zeros(N, device=_device),
        hessian_fn=hessian_fn,
        n_steps=k,
    )

    # For p=3 the Hessian at σ=0 is identically zero, so the first step direction
    # is implementation-arbitrary. Since H is odd (H(−σ) = −H(σ)), a wrong first
    # sign maximizes energy for the entire run. Detect and correct here.
    if energy_fn(trajectory[-1]) > 0:
        trajectory = [(-sigma).clone() for sigma in trajectory]

    # ── Step 5: Compute E_∞ and assemble the result ────────────────────
    #
    # E_inf = 2·√((p-1)/p)
    # Fill the formula strings — they appear in the UI.
    #
    E_inf = 2.0 * math.sqrt((_P - 1) / _P)   # replace: 2.0 * math.sqrt((_P - 1) / _P)

    return AlgorithmResult(
        trajectory=trajectory,
        hessian_fn=hessian_fn,
        energy_fn=energy_fn,
        spectral_edge_fn=edge_fn,
        predictions=RunPredictions(
            E_infinity=E_inf,
            E_infinity_formula=f"2*sqrt((p-1)/p) = ???  [p={_P}]",
            spectral_edge_formula=f"-2*sqrt(p*(p-1))*sqrt(q)  [p={_P}]",
        ),
    )
