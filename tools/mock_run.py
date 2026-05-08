"""
Mock snapshot generator — Stage A substitute for testing the viewer.
Produces schema v0.2.0 JSON files compatible with the API contract
(src/api/types.ts).

Usage:
    python tools/mock_run.py [output.json] [--model pure|mixed] [--p 3] [--N 40] [--k 100] [--seed 42]

Examples:
    python tools/mock_run.py public/mock_run.json
    python tools/mock_run.py public/mock_mixed.json --model mixed
    python tools/mock_run.py public/mock_p4.json --model pure --p 4 --N 60 --k 150
"""
import argparse
import json
import math
import random
from datetime import datetime, timezone


# ── Model helpers ──────────────────────────────────────────────────

def pure_spin_funcs(p):
    E_inf = 2.0 * math.sqrt((p - 1) / p)
    def xi_pp(q):
        return p * (p - 1) * max(q, 1e-9) ** (p - 2)
    def edge(q):
        return -2.0 * math.sqrt(xi_pp(q))
    e_formula = f"2*sqrt((p-1)/p) = {E_inf:.6f}"
    edge_formula = "-2*sqrt(p*(p-1))*q^((p-2)/2)"
    return E_inf, edge, e_formula, edge_formula


def mixed_spin_funcs(components):
    def xi_pp(q):
        return sum(c["beta"] ** 2 * c["p"] * (c["p"] - 1) * max(q, 1e-9) ** (c["p"] - 2)
                   for c in components)
    def edge(q):
        return -2.0 * math.sqrt(xi_pp(q))
    # Rough E_inf for demo — real value from Parisi formula (backend)
    E_inf = sum(abs(c["beta"]) * 2 * math.sqrt((c["p"] - 1) / c["p"]) for c in components)
    comp_str = " + ".join(f"{c['beta']:.2f}*H_{c['p']}" for c in components)
    e_formula = f"Parisi E_inf for {comp_str} ~= {E_inf:.4f} (demo)"
    edge_formula = "-2*sqrt(xi''(q))  xi''(q)=sum(beta_p^2*p*(p-1)*q^(p-2))"
    return E_inf, edge, e_formula, edge_formula


# ── Trajectory ────────────────────────────────────────────────────

def make_trajectory(N, k, rng):
    cur = [0.0] * N
    sigmas = []
    step_size = math.sqrt(N / k)
    for j in range(k):
        target_r = math.sqrt(N * (j + 1) / k)
        v = [rng.gauss(0, 1) for _ in range(N)]
        v[0] += 0.5
        for i in range(1, min(4, N)):
            v[i] += 0.15 * math.sin(math.pi * j / k)
        vn = math.sqrt(sum(x*x for x in v))
        cur = [c + step_size * (vi / vn) for c, vi in zip(cur, v)]
        cn = math.sqrt(sum(x*x for x in cur))
        if cn > 0:
            cur = [x * target_r / cn for x in cur]
        sigmas.append(list(cur))
    return sigmas


# ── Proof annotation ──────────────────────────────────────────────

def proof_annotation(j, k, is_mixed):
    if j == 0:
        return "boundary_origin", r"\sigma_0 = 0 \implies M = I", \
            "At step 0 the iterate is at the origin; M equals the identity."
    if j == k - 1:
        return "terminal_energy", r"\frac{H_N(\sigma_k)}{N} \to -E_\infty", \
            "Final step: energy density converges to -E_inf as N -> inf (Theorem 4 / Parisi)."
    if is_mixed:
        return ("lemma_3_spectral_edge",
                r"\#\{i:\lambda_i\le-2\sqrt{\xi''(q)}+\varepsilon\}\ge N\delta",
                "Projected Hessian has >= N*delta eigenvalues below -2*sqrt(xi''(q)).")
    return ("lemma_3_spectral_edge",
            r"\#\{i:\lambda_i\le-2\sqrt{p(p-1)}\,q^{(p-2)/2}+\varepsilon\}\ge N\delta",
            "Projected Hessian has >= N*delta eigenvalues below predicted spectral edge.")


# ── Main generator ────────────────────────────────────────────────

def make_run(model_type="pure", p=3, N=40, k=100, seed=42,
             components=None):
    rng = random.Random(seed)

    if model_type == "mixed":
        if components is None:
            components = [{"p": 2, "beta": 0.6}, {"p": 3, "beta": 0.8}]
        model_spec = {"type": "mixed_p_spin", "N": N, "components": components, "seed": seed}
        E_inf, edge, e_formula, edge_formula = mixed_spin_funcs(components)
        is_mixed = True
    else:
        model_spec = {"type": "pure_p_spin", "p": p, "N": N, "seed": seed}
        E_inf, edge, e_formula, edge_formula = pure_spin_funcs(p)
        is_mixed = False

    sigmas = make_trajectory(N, k, rng)
    final_sigma = sigmas[-1]

    snapshots = []
    for j, sigma in enumerate(sigmas):
        q = (j + 1) / k
        edge_val = edge(q)
        energy = -E_inf * math.sqrt(q) + rng.gauss(0, 0.04)
        lam_min = edge_val - 0.04 - abs(rng.gauss(0, 0.025))
        top_k = sorted(lam_min + i*0.035 + rng.gauss(0, 0.008) for i in range(20))
        full_spec = None
        if j % 5 == 0:
            full_spec = sorted(rng.gauss(0, 1.0)*1.8 + lam_min + 2.5 for _ in range(N-1))
        tag, katex, human = proof_annotation(j, k, is_mixed)
        overlap_final = sum(a*b for a,b in zip(sigma, final_sigma)) / N

        snapshots.append({
            "index": j, "j": j,
            "q": round(q, 6),
            "radius": round(math.sqrt(N * q), 6),
            "sigma": [round(x, 6) for x in sigma],
            "v_chosen": [round(rng.gauss(0, 1), 6) for _ in range(N)],
            "energy_density": round(energy, 6),
            "lambda_min_tangent": round(lam_min, 6),
            "spectral_edge_target": round(edge_val, 6),
            "edge_target_satisfied": lam_min <= edge_val + 0.01,
            "grad_norm_tangent": round(abs(rng.gauss(0, 0.005)), 8),
            "step_orthogonality_residual": round(abs(rng.gauss(0, 1e-15)), 20),
            "eigvals_topK": [round(x, 6) for x in top_k],
            "full_spectrum": ([round(x, 6) for x in full_spec] if full_spec else None),
            "overlap_with_start": round(sigma[0] / math.sqrt(N), 6),
            "overlap_with_final": round(overlap_final, 6),
            "proof_annotation": {
                "active_lemma": tag.split("_")[0],
                "tag": tag,
                "human_readable": human,
                "katex": katex,
            },
        })

    # Adaptive batch size: max(1, floor(200/sqrt(N)))
    batch_size = max(1, int(200 / math.sqrt(N)))
    batches = []
    run_id = f"mock_{model_type}_N{N}_k{k}_seed{seed}"
    predictions = {
        "E_infinity": E_inf,
        "E_infinity_formula": e_formula,
        "spectral_edge_formula": edge_formula,
    }

    i = 0
    while i < k:
        end = min(i + batch_size - 1, k - 1)
        batches.append({
            "schema_version": "0.2.0",
            "run_id": run_id,
            "batch_index": len(batches),
            "step_range": {"start": i, "end": end},
            "total_steps": k,
            "is_final": end == k - 1,
            "model": model_spec,
            "predictions": predictions,
            "snapshots": snapshots[i:end + 1],
        })
        i = end + 1

    # For static files, wrap all batches into a single document
    return {
        "schema_version": "0.2.0",
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "batches": batches,
        # Flat snapshots at top level for convenience (backwards compat)
        "model": model_spec,
        "predictions": predictions,
        "snapshots": snapshots,
        "results": {
            "final_energy_density": round(snapshots[-1]["energy_density"], 6),
            "final_norm_sq_over_N": round(sum(x*x for x in final_sigma) / N, 6),
            "energy_gap_from_E_infinity": round(abs(snapshots[-1]["energy_density"] + E_inf), 6),
            "validation_passed": True,
        },
    }


# ── CLI ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate mock spin-glass run (schema v0.2.0)")
    ap.add_argument("output", nargs="?", default="public/mock_run.json")
    ap.add_argument("--model", choices=["pure", "mixed"], default="pure")
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--N", type=int, default=40)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data = make_run(model_type=args.model, p=args.p, N=args.N, k=args.k, seed=args.seed)
    with open(args.output, "w") as f:
        json.dump(data, f)
    model_desc = f"p={args.p}" if args.model == "pure" else "2+3 mixture"
    print(f"Wrote {len(data['snapshots'])} snapshots ({args.model}, {model_desc}) to {args.output}")
    print(f"  run_id : {data['run_id']}")
    print(f"  N={args.N}, k={args.k}, batch_size~={max(1, int(200/args.N**0.5))}")
    print(f"  E_inf  : {data['predictions']['E_infinity']:.6f}")
