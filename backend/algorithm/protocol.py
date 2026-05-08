"""
AlgorithmResult — the single interface every algorithm module must return from build().

To add a new algorithm:
  1. Create algorithm/your_model.py
  2. Implement:  def build(model: ModelSpec, k: int) -> AlgorithmResult
  3. Add one entry to algorithm/registry.py  ← that's it

The rest of the stack (snapshot building, batching, HTTP) is automatic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from api.models import RunPredictions


@dataclass
class AlgorithmResult:
    # Full trajectory returned by the algorithm.
    # trajectory[0] = start (zeros), trajectory[1..k] = k descent steps.
    trajectory: list[torch.Tensor]

    # σ → N×N Hessian matrix at that position.
    # Used by snapshot builder to compute projected eigenvalues.
    hessian_fn: Callable[[torch.Tensor], torch.Tensor]

    # σ → H_N(σ)/N  (energy density, scalar float)
    energy_fn: Callable[[torch.Tensor], float]

    # q ∈ [0,1] → spectral edge value  (scalar float, negative)
    # For pure p-spin: -2√(p(p-1)) · q^((p-2)/2)
    # For mixed:       -2√(Σ β²p(p-1)q^(p-2))
    spectral_edge_fn: Callable[[float], float]

    # Human-readable predictions shown in the visualizer.
    predictions: RunPredictions
