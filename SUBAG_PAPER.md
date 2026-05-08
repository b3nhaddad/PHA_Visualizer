# Implementing Subag's Algorithm — Paper Reference Guide

This document maps every step of the algorithm directly to Subag's paper.
Each section states what the paper says, what that means concretely, and
what you need to write.

**Primary reference**
> Subag, E. (2018). *Following the Ground States of Full-RSB Spherical Spin Glasses.*
> Communications on Pure and Applied Mathematics, 74(5), 1021–1044.
> arXiv: 1812.04588

---

## 1. Problem setup

### The spin glass Hamiltonian

For pure p-spin:

```
H_N(σ) = N^{-(p-1)/2}  Σ_{i₁ ... iₚ}  g_{i₁...iₚ} σ_{i₁} ⋯ σ_{iₚ}
```

where:
- σ ∈ ℝᴺ with |σ|² = N  (the N-sphere, radius √N)
- g_{i₁...iₚ} are i.i.d. N(0, 1) Gaussians, independently for each unordered tuple
- The sum is over **all ordered** tuples (with the symmetry built into the normalization)

### Codebase normalization

The code uses **fully symmetrized** tensors J and absorbs the symmetry factor:

| p | Paper | Code | Relation |
|---|-------|------|----------|
| 2 | `N^{-1/2} Σ_{ij} g_{ij} σ_i σ_j` | `(1/2√N) σᵀJσ` | J = symmetrized g, factor of ½ for double-counting |
| 3 | `N^{-1} Σ_{ijk} g_{ijk} σ_i σ_j σ_k` | `(1/6N) Σ J_{ijk} σ_i σ_j σ_k` | J = average over 6 permutations, factor 1/6 for triple-counting |

Both conventions give the same variance structure. The paper's `g` tensors and the code's `J` tensors are related by the same symmetrization step you implement in `_make_disorder`.

### The mixed model

```
H_N(σ) = Σ_p  β_p · H_p(σ)
```

Each component has its own independent disorder tensor and coupling constant β_p.
The covariance function is:

```
ξ(q) = E[H_N(σ₁) H_N(σ₂)] / N  =  Σ_p  β_p²  q^p       (σ₁·σ₂/N = q)
```

---

## 2. The algorithm (Section 2 of the paper)

The goal is to construct a path σ_0, σ_1, ..., σ_k on the sphere such that:

```
H_N(σ_k) / N  →  -E_∞    as N → ∞
```

where the ground-state energy is:

```
E_∞  =  ∫₀¹ √ξ''(t) dt
```

For pure p-spin: `E_∞ = 2√((p-1)/p)`.

---

### Algorithm: Greedy Hessian Descent

**Initialization**

```
σ_0  =  0  ∈ ℝᴺ
```

Start at the origin (inside the sphere, overlap q = 0).

---

**At each step i = 0, 1, ..., k-1:**

**Step A — Compute the projected Hessian**

Let σ = σ_i and q_i = i/k. The *projected* Hessian (also called the *tangent-space* Hessian) is:

```
H̃(σ)  =  M · ∇²H_N(σ) · M

where  M  =  I  −  σσᵀ / |σ|²
```

M is the orthogonal projector onto the tangent space of the sphere at σ.
This removes the radial component so we only see directions we can actually move.

In code (already in `snapshot.py` and `p3.py`):
```python
norm_sq = sigma @ sigma
M = torch.eye(N) - torch.outer(sigma, sigma) / norm_sq
H_proj = M @ hessian_fn(sigma) @ M
```

**Step B — Find the most negative eigenvector**

```
v  =  argmin_{|v|=1, v⊥σ}  vᵀ ∇²H_N(σ) v
```

This is the bottom (most negative) eigenvector of H̃(σ).
The paper says: *"we choose the direction v along which the Hessian is most negative,
since this is the direction in which the energy decreases fastest to second order."*

In code: `eigenvectors[:, 0]` after `torch.linalg.eigh(H_proj)` (eigh returns ascending order).

**Sign convention**: The paper notes that because ∇H_N(σ_q) ≈ 0 near the path, the
gradient should NOT be used to choose the sign of v (it would push toward local minima).
Instead, choose the sign of v so that energy is actually decreasing:
```
if  vᵀ ∇H_N(σ)  >  0:   v = -v        # flip to go downhill
```
The gradient of H_N is exactly what `p2.py` uses for `gradient_fn`. For `p3.py`,
the gradient at position σ is `(1/2N) Σ_{jk} J_{ajk} σ_j σ_k`, which the code
evaluates as `torch.einsum('ijk,j,k', J, position, position)`.

**Step C — Move along v**

```
σ̃_{i+1}  =  σ_i  +  √N · τ · v       where  τ = 1/k  (step_size)
```

The √N factor keeps the step size O(1) relative to the sphere radius √N.
In code:
```python
position = position + math.sqrt(N) * step_size * v
```

**Step D — Renormalize to the target overlap**

After the step, project back to the sphere of radius √(N · q_{i+1}):

```
σ_{i+1}  =  σ̃_{i+1} / |σ̃_{i+1}|  ·  √(N · q_{i+1})

where  q_{i+1} = (i+1)/k
```

In code:
```python
target_radius = math.sqrt(N * (i + 1) / k)
position = position / torch.linalg.norm(position) * target_radius
```

---

## 3. The key theoretical guarantee — Lemma 3

This is what `snapshot.py` checks at every step (`proof_annotation` field).

**Lemma 3** (Subag 2018): For any ε > 0, with high probability as N → ∞,
at overlap q = i/k the minimum eigenvalue of H̃(σ_i) satisfies:

```
λ_min(H̃(σ_q))  ≤  -2√ξ''(q)  +  ε
```

In words: the bottom of the tangent-space Hessian spectrum sits at (or below)
the *spectral edge* `-2√ξ''(q)`.

For pure p-spin:
```
ξ''(q)  =  p(p-1) q^{p-2}

Spectral edge  =  -2√(p(p-1)) · q^{(p-2)/2}
```

For p=2 this is `-2√2` (constant).
For p=3 this is `-2√6 · √q`.

The snapshot viewer shows whether this condition is satisfied at each step.
If it is NOT satisfied (VIOLATED in the UI), the algorithm is not following
the paper's construction correctly.

**`spectral_edge_fn(q)` must return this value exactly.**

---

## 4. Energy along the path

After k steps (overlap q = 1, |σ_k|² = N), the energy density satisfies:

```
H_N(σ_k) / N  →  -∫₀¹ √ξ''(t) dt  =  -E_∞
```

This is the ground-state energy density for full-RSB models.

The paper shows this is optimal: no polynomial-time algorithm can do better
(assuming the Overlap Gap Property).

---

## 5. Concrete formulas by model

### Pure p=2

```
∇²H_N         =  J / √N                    (constant matrix)
ξ''(q)        =  2·1 = 2                   (independent of q)
spectral edge =  -2√2  ≈  -2.8284
E_∞           =  √2    ≈  1.4142
```

The Hessian is the same at every step. Compute once before the loop.

### Pure p=3

```
(∇²H_N)_{ab}  =  (1/N) Σ_k J_{abk} σ_k    (changes every step)
ξ''(q)        =  6q
spectral edge =  -2√6 · √q  ≈  -4.899 · √q
E_∞           =  2√(2/3)    ≈  1.6330
```

The Hessian must be recomputed at every step because it is linear in σ.

### Mixed (p=2 and p=3 example, β₂=β₃=1)

```
ξ''(q)        =  2 + 6q
spectral edge =  -2√(2 + 6q)
E_∞           =  ∫₀¹ √(2 + 6t) dt  =  [⅓(2 + 6t)^{3/2} / 3]₀¹  ≈  ...
```

The Hessian is:
```
H̃_{ab}(σ)  =  β₂·J₂_{ab}/√N  +  β₃·(1/N) Σ_k J₃_{abk} σ_k
```

---

## 6. What to implement for the mixed descent

The descent loop for the mixed model is **identical in structure** to p2/p3 —
only `hessian_fn` differs (it now sums contributions from each component).

Your function should:

```python
def mixed_descent(step_size, dimensionality, start_position, hessian_fn, n_steps):
    position = start_position.clone()
    N = dimensionality
    trajectory = [position.clone()]

    for i in range(n_steps):
        # A: project Hessian to tangent space
        norm_sq = position @ position
        if norm_sq > 1e-12:
            M = torch.eye(N) - torch.outer(position, position) / norm_sq
            H_proj = M @ hessian_fn(position) @ M
        else:
            H_proj = hessian_fn(position)

        # B: bottom eigenvector
        eigenvalues, eigenvectors = torch.linalg.eigh(H_proj)
        v = eigenvectors[:, 0]

        # (optional sign fix — see section 2, Step B)

        # C: step
        position = position + math.sqrt(N) * step_size * v

        # D: renormalize to sphere at overlap (i+1)/n_steps
        target_radius = math.sqrt(N * (i + 1) / n_steps)
        position = position / torch.linalg.norm(position) * target_radius

        trajectory.append(position.clone())

    return trajectory
```

Save this in `algorithm/mixed_descent.py`, then call it from `mixed_pspin.py`:

```python
from algorithm.mixed_descent import mixed_descent

trajectory = mixed_descent(
    step_size      = 1.0 / k,
    dimensionality = N,
    start_position = torch.zeros(N, device=_device),
    hessian_fn     = hessian_fn,   # the summed mixed Hessian you built
    n_steps        = k,
)
```

---

## 7. Correctness checks

After implementing, these observable quantities should match the paper:

| Check | What to look for in the UI |
|---|---|
| Lemma 3 satisfied at each step | `edge_target_satisfied = True` for all snapshots |
| λ_min tracks the spectral edge | `lambda_min_tangent` should be ≤ `spectral_edge_target + 0.01` |
| Energy density converges to E_∞ | Final `energy_density` should be close to `-E_infinity` |
| Step orthogonality | `step_orthogonality_residual` should be near 0 (v ⊥ σ) |
| Overlap increases linearly | `q` increases from 0 to 1 uniformly across snapshots |

For small N (e.g. N=50), fluctuations are large. Use N ≥ 200 for the energy
to be visibly close to E_∞. Use N ≥ 500 for Lemma 3 to hold reliably.

---

## 8. Summary of what you write vs. what's already there

| File | Status | What you write |
|---|---|---|
| `algorithm/p2.py` | ✅ Done (partner) | Do not touch |
| `algorithm/p3.py` | ✅ Done (partner) | Do not touch |
| `algorithm/protocol.py` | ✅ Infrastructure | Do not touch |
| `algorithm/snapshot.py` | ✅ Infrastructure | Do not touch |
| `algorithm/registry.py` | ✅ Infrastructure | Add one line per new model |
| `algorithm/subag.py` | ✅ Infrastructure | Do not touch |
| `algorithm/pure_p2.py` | ⬜ TODO | Disorder tensor + 3 callables + call p2 |
| `algorithm/pure_p3.py` | ⬜ TODO | Disorder tensor + 3 callables + call p3 |
| `algorithm/mixed_pspin.py` | ⬜ TODO | Disorder tensors + ξ'' helpers + 3 callables + descent call |
| `algorithm/mixed_descent.py` | ⬜ TODO (new file) | The descent loop above (Steps A–D) |
