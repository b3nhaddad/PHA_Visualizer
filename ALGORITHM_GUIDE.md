# Algorithm Implementation Guide

This document explains exactly what you need to write to make the backend run, without
touching anything in the API, HTTP routing, snapshot building, or batching layers.

---

## How the backend is structured

```
frontend ──HTTP──► api/routes/runs.py          ← do not touch
                        │
                        ▼
                   algorithm/subag.py           ← do not touch
                        │
                        ▼
                   algorithm/registry.py        ← do not touch
                        │
                 ┌──────┴──────────┐
                 ▼                 ▼
           pure_p2.py         pure_p3.py        ← YOU WRITE THIS
           pure_p3.py         mixed_pspin.py    ← YOU WRITE THIS
           mixed_pspin.py
                 │
                 ▼
           algorithm/p2.py  (partner's code)    ← do not touch
           algorithm/p3.py  (partner's code)    ← do not touch
```

**Your job**: implement `build(model, k) -> AlgorithmResult` in each wrapper file.
The infrastructure handles everything else.

---

## The AlgorithmResult contract

Every `build()` function must return an `AlgorithmResult` (defined in `algorithm/protocol.py`):

```python
@dataclass
class AlgorithmResult:
    trajectory:        list[torch.Tensor]                   # length k+1
    hessian_fn:        Callable[[torch.Tensor], torch.Tensor]
    energy_fn:         Callable[[torch.Tensor], float]
    spectral_edge_fn:  Callable[[float], float]             # q → negative float
    predictions:       RunPredictions
```

### trajectory

- A Python list of `k+1` torch tensors, each of shape `(N,)`.
- `trajectory[0]` = starting position (always `torch.zeros(N)` on the sphere).
- `trajectory[i]` = position on the sphere after step `i`.
- This comes directly from the descent function return value — do not modify it.

### hessian_fn

```python
def hessian_fn(sigma: torch.Tensor) -> torch.Tensor:
    # sigma: shape (N,)
    # return: shape (N, N)  — the full Hessian matrix at sigma
```

### energy_fn

```python
def energy_fn(sigma: torch.Tensor) -> float:
    # sigma: shape (N,)
    # return: H_N(sigma) / N  as a plain Python float
```

### spectral_edge_fn

```python
def edge_fn(q: float) -> float:
    # q: overlap (0 ≤ q ≤ 1)
    # return: -2 * sqrt(xi''(q))   — always a negative number
```

### RunPredictions

```python
RunPredictions(
    E_infinity=float,            # predicted ground-state energy density
    E_infinity_formula=str,      # human-readable formula shown in UI
    spectral_edge_formula=str,   # human-readable formula shown in UI
)
```

---

## Pure p=2 spin glass  (`algorithm/pure_p2.py`)

### Physics

| Quantity | Formula |
|---|---|
| Hamiltonian | `H_N(σ) = (1/2√N) σᵀ J σ` |
| Disorder | `J` — N×N GOE matrix (symmetric Gaussian, zero diagonal) |
| Hessian | `∇²H_N = J/√N` (constant — does not depend on σ) |
| Energy | `H_N(σ)/N = σᵀJσ / (2·N·√N)` |
| Spectral edge | `-2√2` (constant, independent of q) |
| E_∞ | `2·√((p-1)/p) = √2 ≈ 1.4142` |

### Steps

**1. Sample J (GOE)**
```python
def _make_disorder(N: int, seed: int) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed)
    J = torch.randn(N, N, generator=g)
    J = (J + J.T) / 2.0
    J.fill_diagonal_(0.0)
    return J.to(_device)
```

**2. Build the constant Hessian**
```python
J = _make_disorder(N, seed)
H_const = J / math.sqrt(N)
```

**3. Define callables**
```python
hessian_fn = lambda sigma: H_const
energy_fn  = lambda sigma: float(sigma @ (J @ sigma)) / (2.0 * N * math.sqrt(N))
edge_fn    = lambda q: -2.0 * math.sqrt(_P * (_P - 1)) * max(q, 1e-9) ** ((_P - 2) / 2.0)
# For p=2, q^((p-2)/2) = q^0 = 1, so edge_fn returns -2√2 always.
```

**4. Run the descent**
```python
trajectory = spatial_hessian_descent_p2(
    step_size      = 1.0 / k,
    dimensionality = N,
    start_position = torch.zeros(N, device=_device),
    hessian_fn     = hessian_fn,
    n_steps        = k,
)
```

**5. Return**
```python
E_inf = 2.0 * math.sqrt((_P - 1) / _P)
return AlgorithmResult(
    trajectory    = trajectory,
    hessian_fn    = hessian_fn,
    energy_fn     = energy_fn,
    spectral_edge_fn = edge_fn,
    predictions   = RunPredictions(
        E_infinity           = E_inf,
        E_infinity_formula   = f"2*sqrt((p-1)/p) = {E_inf:.6f}  [p={_P}]",
        spectral_edge_formula= f"-2*sqrt(p*(p-1)) = {-2*math.sqrt(_P*(_P-1)):.6f}  [p={_P}, constant]",
    ),
)
```

---

## Pure p=3 spin glass  (`algorithm/pure_p3.py`)

### Physics

| Quantity | Formula |
|---|---|
| Hamiltonian | `H_N(σ) = (1/6N) Σ_{ijk} J_{ijk} σ_i σ_j σ_k` |
| Disorder | `J` — N×N×N fully symmetric Gaussian tensor |
| Hessian | `H_{ab}(σ) = (1/N) Σ_k J_{abk} σ_k` (linear in σ, recomputed each step) |
| Energy | `H_N(σ)/N = Σ J_{ijk} σ_i σ_j σ_k / (6·N²)` |
| Spectral edge | `-2√6 · √q` |
| E_∞ | `2·√(2/3) ≈ 1.6330` |

### The J injection requirement

`p3.py` line 68 references a bare global `J` inside the descent function.
Before calling `spatial_hessian_descentp3` you **must** write:

```python
import algorithm.p3 as _p3_module   # already imported at top of file
_p3_module.J = J                    # inject BEFORE the call, every time
```

The import is already at the top of `pure_p3.py` — do not remove it.

### Steps

**1. Sample J (fully symmetric rank-3 tensor)**
```python
def _make_disorder(N: int, seed: int) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed)
    J = torch.randn(N, N, N, generator=g)
    J = (
        J + J.permute(0,2,1) + J.permute(1,0,2)
        + J.permute(1,2,0) + J.permute(2,0,1) + J.permute(2,1,0)
    ) / 6.0
    return J.to(_device)
```

**2-5. Build callables, inject J, run descent**
```python
J = _make_disorder(N, seed)
_p3_module.J = J                    # ← required injection

hessian_fn = lambda sigma: torch.einsum("ijk,k->ij", J, sigma) / N
energy_fn  = lambda sigma: float(torch.einsum("ijk,i,j,k->", J, sigma, sigma, sigma)) / (6.0 * N * N)
edge_fn    = lambda q: -2.0 * math.sqrt(_P * (_P - 1)) * max(q, 1e-9) ** ((_P - 2) / 2.0)

trajectory = spatial_hessian_descentp3(
    step_size      = 1.0 / k,
    dimensionality = N,
    start_position = torch.zeros(N, device=_device),
    hessian_fn     = hessian_fn,
    n_steps        = k,
)
```

**6. Return** — same pattern as p=2, with `E_inf = 2.0 * math.sqrt((_P - 1) / _P)`.

---

## Mixed p-spin glass  (`algorithm/mixed_pspin.py`)

### Physics

The mixture couples multiple pure-p models with coupling constants β_p:

| Quantity | Formula |
|---|---|
| Hamiltonian | `H_N(σ) = Σ_p β_p H_p(σ)` |
| ξ(q) | `Σ_p β_p² q^p` |
| ξ''(q) | `Σ_p β_p² p(p-1) q^(p-2)` |
| Spectral edge | `-2√ξ''(q)` |
| E_∞ | `∫₀¹ √ξ''(t) dt` |

The `components` list (from `model.components`) gives you each `(p, β)` pair.

### Steps

**1. Implement the physics helpers**
```python
def _xi_pp(components, q):
    return sum(c.beta**2 * c.p * (c.p-1) * max(q,1e-9)**(c.p-2) for c in components)

def _spectral_edge(components, q):
    return -2.0 * math.sqrt(max(_xi_pp(components, q), 1e-12))

def _E_inf(components, n_pts=1000):
    dt = 1.0 / n_pts
    return sum(math.sqrt(max(_xi_pp(components, (i+0.5)*dt), 0)) * dt for i in range(n_pts))
```

**2. Build disorder tensors (one per unique p)**
```python
disorder = {}
for i, c in enumerate(components):
    if c.p not in disorder:
        if c.p == 2:
            disorder[2] = _make_J2(N, seed + i)
        elif c.p == 3:
            disorder[3] = _make_J3(N, seed + i)
        # add more p values here as needed
```

**3. Define hessian_fn and energy_fn as sums over components**
```python
def hessian_fn(sigma):
    H = torch.zeros(N, N, device=_device)
    for c in components:
        if c.p == 2:
            H = H + c.beta * disorder[2] / math.sqrt(N)
        elif c.p == 3:
            H = H + c.beta * torch.einsum("ijk,k->ij", disorder[3], sigma) / N
    return H

def energy_fn(sigma):
    e = 0.0
    for c in components:
        if c.p == 2:
            e += c.beta * float(sigma @ (disorder[2] @ sigma)) / (2.0 * N * math.sqrt(N))
        elif c.p == 3:
            e += c.beta * float(torch.einsum("ijk,i,j,k->", disorder[3], sigma, sigma, sigma)) / (6.0 * N * N)
    return e
```

**4. Write and call your descent algorithm**

Your mixed descent function can follow the same pattern as `spatial_hessian_descent_p2` /
`spatial_hessian_descentp3` — it receives a `hessian_fn` and does not need to know whether
it's pure or mixed. Place it in a new file (e.g. `algorithm/mixed_descent.py`) or inline.

```python
trajectory = your_mixed_descent(
    step_size      = 1.0 / k,
    dimensionality = N,
    start_position = torch.zeros(N, device=_device),
    hessian_fn     = hessian_fn,
    n_steps        = k,
)
```

---

## Checklist before testing

For each model you implement:

- [ ] `_make_disorder` returns a tensor on `_device` with the right shape
- [ ] `hessian_fn(sigma)` returns shape `(N, N)` on `_device`
- [ ] `energy_fn(sigma)` returns a plain Python `float` (not a tensor)
- [ ] `edge_fn(q)` returns a negative float for any `0 < q ≤ 1`
- [ ] `trajectory` is a Python list of length `k+1`, each element shape `(N,)` on `_device`
- [ ] `E_infinity` in `RunPredictions` is a finite float (not None, not NaN)
- [ ] For p=3: `_p3_module.J = J` is called before `spatial_hessian_descentp3`
- [ ] No `raise NotImplementedError` remains in the `build()` function

---

## How to add a new model type (e.g. p=4)

1. Create `algorithm/pure_p4.py` — copy the structure from `pure_p2.py` and fill in your physics.
2. Open `algorithm/registry.py` and add one line:
   ```python
   from algorithm import pure_p4
   # inside _REGISTRY:
   ('pure_p_spin', 4): pure_p4.build,
   ```
3. That's it. HTTP routing, batching, and snapshot building are automatic.

---

## Running the backend

```bat
.\run_backend.bat
```

The server starts at `http://localhost:8000`.
Do **not** add `--reload` — it causes a WinError 10013 on Windows.

After editing a wrapper file, stop the server (Ctrl+C) and restart it.

---

## Quick smoke test

With the backend running, open a browser or use curl:

```
GET http://localhost:8000/api/health
```

Should return `{"status": "ok"}`.

To trigger a real run, start the frontend (`run_frontend.bat`) and use the UI,
or POST directly:

```json
POST http://localhost:8000/api/runs
{
  "model": { "type": "pure_p_spin", "N": 50, "p": 2, "seed": 42 },
  "algorithm": { "n_steps": 100 }
}
```

A `NotImplementedError` in `build()` will return HTTP 500 with the error message —
that's expected until you fill in the TODOs.
