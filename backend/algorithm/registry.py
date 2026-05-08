"""
Algorithm registry — maps model specs to build() functions.

To add a new algorithm
----------------------
1. Create algorithm/your_module.py with:
       def build(model: YourModelSpec, k: int) -> AlgorithmResult: ...

2. Import it here and add one entry to _REGISTRY:
       from algorithm import your_module
       # pure p=4:
       ('pure_p_spin', 4): your_module.build,
       # or mixed:
       ('mixed_p_spin', None): your_module.build,

That's all.  HTTP routing, batching, and snapshot building are automatic.
"""
from __future__ import annotations

from typing import Callable

from algorithm import mixed_pspin, pure_p2, pure_p3
from algorithm.protocol import AlgorithmResult
from api.models import ModelSpec, MixedSpinSpec, PureSpinSpec

# ── Registry ───────────────────────────────────────────────────────
# Key for pure models:  ('pure_p_spin', p)
# Key for mixed models: ('mixed_p_spin', None)

_REGISTRY: dict[tuple, Callable] = {
    ('pure_p_spin', 2): pure_p2.build,
    ('pure_p_spin', 3): pure_p3.build,
    # ('pure_p_spin', 4): pure_p4.build,    ← add when ready
    ('mixed_p_spin', None): mixed_pspin.build,
}


def dispatch(model: ModelSpec, k: int) -> AlgorithmResult:
    """Look up and call the right build() function for this model."""
    if isinstance(model, PureSpinSpec):
        key = ('pure_p_spin', model.p)
    elif isinstance(model, MixedSpinSpec):
        key = ('mixed_p_spin', None)
    else:
        raise ValueError(f"Unknown model type: {model.type!r}")

    build_fn = _REGISTRY.get(key)
    if build_fn is None:
        raise NotImplementedError(
            f"No algorithm registered for {key}. "
            f"Add an entry to algorithm/registry.py."
        )
    return build_fn(model, k)
