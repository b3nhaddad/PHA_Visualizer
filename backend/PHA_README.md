# Converting the Hessian-Descent Code to a Mixed FRSB Spherical Spin Glass

A reference for transforming the current pure-p notebook into the genuine setting of Subag's *"Following the ground state of full-RSB spherical spin glasses"* (2021).

---

## 0. Background: the three axes of spin glass models

Three independent choices define a spin glass:

**Geometry.** σ ∈ {±1}^N (Ising/cube) or σ ∈ √N · S^{N-1} (sphere). The notebook is on the sphere, despite the markdown's references to SK.

**Interaction order p.** Each "p-body" Hamiltonian is

$$H_p(\sigma) = -\frac{1}{N^{(p-1)/2}} \sum_{i_1, \ldots, i_p} J_{i_1 \cdots i_p}\, \sigma_{i_1} \cdots \sigma_{i_p}, \quad J \sim \mathcal{N}(0,1)$$

with covariance E[H_p(σ)H_p(σ′)] = N · (σ·σ′/N)^p.

**Pure vs mixed.** Pure means a single p. Mixed means a superposition

$$H(\sigma) = \sum_{p \geq 2} \gamma_p\, H_p(\sigma), \qquad \nu(t) := \sum_{p \geq 2} \gamma_p^2\, t^p$$

The **mixture function** ν is the central object. Every RSB property of the model — replica symmetry, 1RSB, FRSB — is a property of ν.

| Model | Geometry | p | Pure/Mixed | Phase |
|---|---|---|---|---|
| SK | cube | 2 | pure | FRSB on cube |
| Pure p-spin Ising | cube | one p | pure | varies |
| Pure spherical p=2 | sphere | 2 | pure | RS (just GOE) |
| Pure spherical p≥3 | sphere | one p≥3 | pure | 1RSB |
| **Mixed spherical (target)** | **sphere** | **multiple** | **mixed** | **FRSB if ν chosen right** |

Subag's algorithm gives the ground state in the bottom row only.

---

## 1. The FRSB criterion

For a spherical mixed model, a sufficient condition for full RSB (Auffinger–Chen 2017) is:

> **The function $t \mapsto 1/\sqrt{\nu''(t)}$ is concave on $[0,1]$.**

When this holds, the Parisi measure has full support on [0,1], and the ground state energy density is

$$E_{\text{GS}} = -\int_0^1 \sqrt{\nu''(t)}\, dt$$

This integral is the target your descent code is *already* trying to hit. The whole point of converting to a mixed FRSB model is to make this prediction the actual ground state, so the algorithm has something to be tight against.

**Recommended test mixtures:**

| ν(t) | γ's | FRSB? | Predicted GS density |
|---|---|---|---|
| t² + t³ + t⁴ (after normalization) | γ₂=γ₃=γ₄=1 | yes | ≈ -1.30 |
| ½t² + ⅓t³ | γ₂=√½, γ₃=√⅓ | yes | ≈ -0.97 |
| ½t² + t⁴ | γ₂=√½, γ₄=1 | yes | ≈ -1.21 |
| t² alone | pure p=2 | RS, not FRSB | ≈ -1 (just λ_max) |
| t³ alone | pure p=3 | 1RSB, not FRSB | threshold ≠ GS |

Always check the concavity of 1/√ν'' numerically before trusting the predicted GS:

```python
def is_frsb(nu_double_prime, n=1000):
    ts = torch.linspace(1e-6, 1, n)
    f = 1.0 / torch.sqrt(nu_double_prime(ts))
    # discrete second derivative; should be ≤ 0 everywhere
    f_pp = f[2:] - 2*f[1:-1] + f[:-2]
    return (f_pp <= 1e-6).all().item()
```

---

## 2. Code changes, in order

### Change 1 — Mixed disorder generation

**Current:** a single tensor `J` (rank 2 or rank 3), used in isolation.

**New:** independent Gaussian tensors for each p in the mixture, properly symmetrized and scaled.

```python
import itertools

def symmetrize(J):
    """Average over all p! permutations of indices."""
    p = J.ndim
    N = J.shape[0]
    out = torch.zeros_like(J)
    for perm in itertools.permutations(range(p)):
        out = out + J.permute(*perm)
    return out / math.factorial(p)

def make_mixed_disorder(N, gammas, device='cpu', seed=None):
    """
    gammas: dict {p: gamma_p} for p in {2, 3, 4, ...}
    Returns dict {p: J_p} of symmetrized, normalized tensors.
    """
    if seed is not None:
        torch.manual_seed(seed)
    Js = {}
    for p, gamma in gammas.items():
        if gamma == 0:
            continue
        J = torch.randn(*([N] * p), device=device)
        J = symmetrize(J) * gamma / N**((p - 1) / 2)
        Js[p] = J
    return Js
```

**Tip:** The variance scaling 1/N^((p-1)/2) is what makes E[H(σ)H(σ′)] = N · ν(σ·σ′/N) come out clean. Get this wrong and the predicted GS won't match.

### Change 2 — Mixed energy, gradient, Hessian as derived objects

**Current:** hand-coded `energy(σ) = -σᵀJσ/(2N)` and `energy_p3(σ) = -⟨J,σ⊗³⟩/3`, with separate hand-coded gradients.

**New:** one `energy_mixed`, with gradient and Hessian generated automatically via autograd. This way switching mixtures is one edit.

```python
def tensor_contract(Jp, sigma):
    """Contract a symmetric p-tensor with p copies of sigma."""
    p = Jp.ndim
    indices = ''.join(chr(ord('i') + k) for k in range(p))
    eq = indices + ',' + ','.join(indices)
    return torch.einsum(eq, Jp, *([sigma] * p))

def make_mixed_energy(Js):
    def energy(sigma):
        total = sigma.new_zeros(())
        for p, Jp in Js.items():
            total = total + (-1.0 / p) * tensor_contract(Jp, sigma)
        return total
    return energy

def make_gradient_fn(energy_fn):
    def grad_fn(sigma):
        sigma = sigma.detach().flatten().requires_grad_(True)
        return torch.autograd.grad(energy_fn(sigma), sigma)[0]
    return grad_fn

def make_hessian_fn(energy_fn):
    def hess_fn(sigma):
        sigma = sigma.detach().flatten().requires_grad_(True)
        return torch.autograd.functional.hessian(energy_fn, sigma)
    return hess_fn
```

**Tip:** For larger N you can write the Hessian in closed form to skip autograd:

$$\nabla^2 H(\sigma) = -\sum_p (p-1)\, J_p[\sigma^{\otimes(p-2)}]$$

where the bracket means contracting (p−2) of the p indices with σ, leaving a rank-2 tensor. This is ~10× faster than `autograd.functional.hessian` because autograd builds a graph of size O(p · N²) per call. Closed form for the standard 2+3+4 mixture:

```python
def hessian_explicit(sigma, Js):
    H = torch.zeros(N, N, device=sigma.device)
    if 2 in Js: H = H - 1.0 * Js[2]                                       # ∇²H_2 = -J_2
    if 3 in Js: H = H - 2.0 * torch.einsum('ijk,k->ij', Js[3], sigma)
    if 4 in Js: H = H - 3.0 * torch.einsum('ijkl,k,l->ij', Js[4], sigma, sigma)
    return H
```

### Change 3 — Drop the constant-Hessian shortcut

**Current:** `spatial_hessian_descent_p2` computes eigendecomposition once outside the loop, exploiting that the pure-p=2 Hessian is just −J/N (constant in σ).

**New:** delete `spatial_hessian_descent_p2` from the run path. For mixed (or any model with p≥3 contributions), the Hessian's σ-dependence is essential — it's what makes the descent direction rotate as σ moves outward, which is the whole geometric content of the algorithm. Use only the general `spatial_hessian_descent`, recomputing eigendecomposition every step.

If you keep `_p2` around for benchmarking pure p=2, rename it `spatial_hessian_descent_constant_hessian` so it's clearly a special-case shortcut, not the default.

### Change 4 — Fix the hardcoded p=3 gradient inside the descent

**Current** (inside `spatial_hessian_descent`):

```python
if norm_sq > 1e-12 and torch.dot(v, torch.einsum('ijk,j,k', J, position, position)) < 0:
    v = -v
```

This references a global `J` and assumes it's a rank-3 tensor. Silent wrong-tensor bug as soon as you switch to mixed.

**New:** use the gradient function passed in, with the correct sign convention. v is a *descent* direction, so the energy should decrease when stepping along v:

```python
if gradient_fn is not None:
    grad = gradient_fn(position)
    if torch.dot(v, grad) > 0:
        v = -v
```

**Sign convention sanity check:** along a tangent direction v, the energy change to first order is dE = ⟨∇H, v⟩ · ds. We want dE ≤ 0, so we want ⟨∇H, v⟩ ≤ 0. If the dot product is positive, flip. The current code flips on negative, which is the opposite sign convention — this would have been silently wrong for p=3 too, but masked by the radial rescaling and small step size.

### Change 5 — Tangent projection (no change, but verify)

The general descent has

```python
M = I - σσᵀ/||σ||²
H_proj = M @ H @ M
```

This is correct geometrically and is independent of which p's contribute to H. No code change. Just note: if you use the closed-form Hessian (the tip in Change 2), you still apply M @ H @ M afterward — the projection is on the matrix, not on how you computed it.

### Change 6 — Cleaner radial-plus-tangent step

**Current:** `position = position + sqrt(N) * step_size * v` followed by hard rescale to radius √(Nq).

**Equivalent but cleaner** (matches Subag's writeup more directly):

$$\sigma_{q+\Delta q} \;=\; \sigma_q\,\sqrt{\tfrac{q+\Delta q}{q}} \;+\; v\,\sqrt{N\,\Delta q\,\bigl(1 - \tfrac{\Delta q}{q+\Delta q}\bigr)}$$

This places σ_{q+Δq} exactly on the radius-√(N(q+Δq)) sphere with v tangent at σ_q, no rescaling needed. Verify the norm:

$$\|\sigma_{q+\Delta q}\|^2 = Nq \cdot \tfrac{q+\Delta q}{q} + N\Delta q \cdot \tfrac{q}{q+\Delta q} = N(q + \Delta q)$$

(the cross-term vanishes because v ⟂ σ_q after projection). For small Δq this matches the current update to leading order, but the exact form lets you check the energy increment

$$\Delta E \approx \sqrt{N\,\nu''(q)} \cdot \sqrt{N\,\Delta q} \cdot \langle v, \text{(unit eigvec)}\rangle = -\sqrt{\nu''(q)} \cdot N \cdot \Delta q$$

per step, which integrates to E(1) = -∫₀¹ √ν''. The hard-rescale form works numerically but obscures this verification.

### Change 7 — Initialization at exactly σ = 0

**Current:** `start[0] = 1e-8` to avoid divide-by-zero in `M = I - σσᵀ/||σ||²`.

**New:** handle σ = 0 explicitly. At the origin there's no tangent constraint, so the first step uses the bare Hessian and steps out to radius √(Nτ):

```python
position = torch.zeros(N, device=device)

for i in range(n_steps):
    norm_sq = torch.dot(position, position)

    if norm_sq < 1e-12:
        # First step: no tangent projection, use bare Hessian
        H = hessian_fn(position)
        eigvals, eigvecs = torch.linalg.eigh(H)
        v = eigvecs[:, 0]              # most negative direction
    else:
        # Tangent-projected step as before
        ...

    target_q = (i + 1) / n_steps
    position = position * something + v * something_else  # see Change 6
```

The 1e-8 seed worked but biased the first eigenvector slightly. Starting at exactly zero is one fewer arbitrary choice.

### Change 8 — FRSB verification harness

This is what turns the run from "produces a number" into "tests the theorem." Add three things:

```python
def predicted_gs(nu_dd, n_int=10000):
    ts = torch.linspace(1e-6, 1, n_int)
    return -torch.trapz(torch.sqrt(nu_dd(ts)), ts).item()

def predicted_energy_at_q(nu_dd, q, n_int=10000):
    ts = torch.linspace(1e-6, q, n_int)
    return -torch.trapz(torch.sqrt(nu_dd(ts)), ts).item()
```

Along the trajectory, log:

1. **Energy density vs q.** `energies = [energy_mixed(s).item() / N for s in trajectory]`. Plot against q = i/k. Overlay `predicted_energy_at_q(ν'', q)`. They should agree to within finite-N noise (O(1/√N) ≈ 16% at N=40, ≈ 5% at N=400).

2. **Hessian bulk edge along the path.** For FRSB, the smallest eigenvalue of the projected Hessian should approach 0 — TAP states are marginally stable. For pure 1RSB it stays strictly negative until the threshold q* and jumps. Plotting this is the cleanest way to *see* you're in the FRSB regime:

```python
edges = []
for s in trajectory[::10]:
    H = hessian_fn(s)
    M = torch.eye(N, device=device) - torch.outer(s, s) / torch.dot(s, s)
    eig = torch.linalg.eigvalsh(M @ H @ M)
    edges.append(eig[0].item())
```

3. **Multi-seed averaging.** Disorder fluctuations at finite N are sizeable. Run with 10+ seeds and average the energy curves. The variance across seeds is itself a useful diagnostic: small variance ⇒ you're in the self-averaging regime.

### Change 9 — ν as the single source of truth

Lift ν, ν′, ν′′ out of being implicit in the energy:

```python
def make_nu_functions(gammas):
    def nu(t):    return sum(g**2 * t**p           for p, g in gammas.items())
    def nu_p(t):  return sum(g**2 * p * t**(p-1)   for p, g in gammas.items())
    def nu_pp(t): return sum(g**2 * p*(p-1) * t**(p-2) for p, g in gammas.items())
    return nu, nu_p, nu_pp
```

Then `gammas`, the disorder, the energy, the gradient, the Hessian, the FRSB check, the predicted GS, and the verification all derive from `gammas` alone. Switching mixtures = one dict change.

---

## 3. Additional tips

### Memory and tractability

The disorder tensor at order p has N^p entries. At N = 200:

| p | Tensor size | Memory (float32) |
|---|---|---|
| 2 | 4 × 10⁴ | 160 KB |
| 3 | 8 × 10⁶ | 32 MB |
| 4 | 1.6 × 10⁹ | 6.4 GB |
| 5 | 3.2 × 10¹¹ | 1.3 TB |

p=4 is already the practical ceiling for dense storage on a single GPU. p=5+ requires either much smaller N or a sparse symmetric representation (which torch doesn't support natively — you'd need to roll your own contraction). For most FRSB experiments, ν = c₂t² + c₃t³ + c₄t⁴ is more than enough to demonstrate the regime.

### Symmetric tensor storage

The fully symmetric p-tensor has only N^p / p! distinct entries (e.g. for p=4, N=200: ~1.66 × 10⁸ entries vs 1.6 × 10⁹). If you hit memory limits, generate only the upper triangle (sorted index tuples i₁ ≤ i₂ ≤ … ≤ iₚ) and write a custom contraction. Not worth doing until N > 200 with p=4.

### Step size and number of steps

With τ = 1/k, the discretization error is O(τ) per step, O(1) total. Keep k ≥ 100 for the integral ∫√ν'' to be accurately approximated by the trajectory's energy. k = 1000 is overkill at small N but cheap. The cost per step is dominated by `eigh` (O(N³)) and Hessian construction (O(N^p) for the p-th term). Total: O(k · (Σ_p N^p + N³)).

### Finite-N effects

At N = 40, 1/√N ≈ 0.16, so expect 10–20% deviation from the predicted GS. Sweep N ∈ {40, 80, 160, 320} and check that |E_observed − E_predicted| decays like c/√N. If it doesn't decay, something is wrong with the implementation (sign, normalization, projection). A clean log-log plot of error vs N is the single best correctness test.

### Common pitfalls

- **Normalization conventions vary across papers.** Subag uses E[H(σ)H(σ′)] = N · ξ(σ·σ′/N) with ξ(t) = Σ a_p t^p; Auffinger–Chen use ν; Crisanti–Sommers use a different prefactor. When comparing your numbers to a paper's predicted GS, double-check whose convention they use. The factor sometimes differs by a sqrt(2) or a 1/p!.

- **Symmetrization is not optional.** A non-symmetric J_p still gives a well-defined energy (because contracting with σ⊗p symmetrizes implicitly), but the gradient and Hessian formulas use J_p directly and will be wrong by combinatorial factors. Always store the fully symmetric tensor.

- **`torch.linalg.eigh` vs `eig`.** Use `eigh` — your projected Hessian is guaranteed real symmetric, and `eigh` is faster and numerically stabler. Using `eig` returns complex eigenvalues with tiny imaginary parts that will break sorting.

- **Watch for the sign convention in `eigvecs[:, 0]`.** `eigh` returns eigenvalues in ascending order, so `[:, 0]` is the most negative — which is what you want for descent. Verify by printing `eigvals[0]` once and confirming it's negative for typical σ.

- **Double precision matters at small N.** Float32 noise floor is ~10⁻⁷, and Hessian eigenvalue separations can be smaller than that for clustered eigenvalues. If you see weird trajectory jumps, switch to `torch.float64`. Cost is 2× memory and ~2× compute.

### TAP-tree connection (for intuition while debugging)

The trajectory σ_q is approximately the centroid of a TAP state at overlap level q. The most-negative-eigenvector step is the move from a parent TAP state to a child along the FRSB ultrametric tree. The Hessian's bulk edge tracking 0 in FRSB is the statement that all TAP children are marginally stable — there's a continuum of branches at every level, and Subag's algorithm picks the steepest one.

If you instrument the projected Hessian's full spectrum every ~10 steps and plot it as a heatmap (eigenvalue index × q), the FRSB signature is a bulk that creeps up to touch zero from below. The 1RSB pure case shows a bulk that stays strictly below zero with a gap that closes only at the threshold q*.

### Recommended reading order

1. **Subag (2018)** — *"Free energy landscapes in spherical spin glasses"*, arXiv:1804.10576. Establishes TAP-state structure rigorously for mixed spherical models. Read for the geometric picture.
2. **Subag (2021)** — *"Following the ground state of full-RSB spherical spin glasses"*, Comm. Pure Appl. Math. The actual paper your code is implementing. The construction in Section 2 and the optimality theorem are what matter; the proofs are heavy.
3. **Auffinger–Chen (2017)** — *"Parisi formula for the ground state energy in the mixed p-spin model"*. Contains the FRSB criterion and the GS = -∫√ν'' formula in clean form.
4. **Chen–Panchenko–Subag (2023)** — *"The generalized TAP free energy"*. The TAP free energy formula at general overlap, useful if you want to compute F_TAP along the trajectory.
5. **Sellke (2024)** — *"Optimizing mean field spin glasses with external field"*. Generalizations and computational versions of Subag's algorithm; the writing is friendlier than Subag's.

Skip the original 1980s TAP and Parisi papers unless you want context — they cover the cube case and use replica-trick heuristics that have since been replaced by the rigorous spherical formulations above.

### Visualizing the trajectory

A simple but informative plot: project {σ_q} onto its first two principal components (across q) and trace the path on a 2D disk. For FRSB the path is a smooth spiral outward; for pure p=2 it's a straight radial line in the top-eigenvector direction; for pure p≥3 it sometimes shows kinks or near-discontinuities at the threshold q*.

```python
S = torch.stack(trajectory).cpu().numpy()  # (k+1, N)
U, _, _ = np.linalg.svd(S - S.mean(0), full_matrices=False)
plt.plot(U[:, 0], U[:, 1])
plt.axis('equal')
```

### Performance baseline

For N = 200, k = 100, mixture ν = ½t² + ⅓t³ + ¼t⁴:

- Disorder generation: < 1 s on GPU
- One descent run: ~5–15 s on a T4 / A100
- Full sweep over 10 seeds: ~2 min

If yours is much slower, the suspect is autograd's Hessian — switch to the closed-form Hessian from Change 2.

---

## 4. Order of operations (checklist)

- [ ] Write `make_nu_functions`, `is_frsb`, `predicted_gs`, `predicted_energy_at_q` (Change 9 + verification)
- [ ] Pick a test mixture: γ₂ = γ₃ = γ₄ = 1; verify FRSB; compute predicted GS
- [ ] Write `symmetrize` and `make_mixed_disorder` (Change 1)
- [ ] Write `make_mixed_energy`, `make_gradient_fn`, `make_hessian_fn` (Change 2); optionally the closed-form Hessian
- [ ] Delete `spatial_hessian_descent_p2` from the run path (Change 3)
- [ ] Fix the hardcoded p=3 gradient inside `spatial_hessian_descent` (Change 4)
- [ ] Add σ = 0 special case at the top of the descent loop (Change 7)
- [ ] (Optional) Switch to the explicit radial+tangent step form (Change 6)
- [ ] Add energy / Hessian-edge logging (Change 8)
- [ ] Run at N = 40 to make sure nothing crashes
- [ ] Run at N ∈ {80, 160, 320} with 10 seeds; plot mean energy density vs q against the theoretical curve
- [ ] Plot Hessian-edge vs q — should approach 0 in FRSB
- [ ] Compare to a pure p=3 baseline at the same N — mixed FRSB endpoint should be more negative

If the predicted-vs-observed energy curves overlay (within a few percent at N ≥ 200) and the Hessian edge approaches zero, you've successfully reproduced Subag's FRSB algorithm.

---

## 5. Two final sanity checks worth running

**Check A — pure p=2 reduction.** Set γ₃ = γ₄ = 0, γ₂ = 1. The mixed code should give the same answer as the original `spatial_hessian_descent_p2` on the same J₂. If it doesn't, there's a normalization or sign bug. This is the cheapest unit test.

**Check B — invariance under disorder rotation.** The model is rotationally invariant, so for any orthogonal Q, running the algorithm on J_p ↦ J_p[Q,Q,…,Q] (rotated tensor) should give the same final *energy* (the trajectory rotates with Q). This catches subtle bugs where some part of the code accidentally privileges a coordinate axis.

Both are < 20 lines of additional test code and will save hours of confused debugging if anything is off.
