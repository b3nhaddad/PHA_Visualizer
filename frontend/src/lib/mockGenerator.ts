/**
 * Browser-side mock data generator.
 * Produces plausible-looking (not real physics) snapshot batches for UI testing.
 * Uses the same adaptive batch_size formula as the recommended backend default.
 */
import type { SnapshotBatch, Snapshot, ModelSpec, RunPredictions, ProofTag } from '../api/types';

export interface MockConfig {
  model: ModelSpec;
  n_steps: number;
  seed?: number;
}

// ── Seeded RNG (LCG + Box-Muller) ─────────────────────────────────

class RNG {
  private s: number;
  constructor(seed: number) {
    this.s = (seed >>> 0) || 1;
  }
  next(): number {
    this.s = Math.imul(1664525, this.s) + 1013904223;
    return (this.s >>> 0) / 0x100000000;
  }
  gauss(mean = 0, std = 1): number {
    const u1 = Math.max(1e-12, this.next());
    const u2 = this.next();
    return mean + std * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }
  arr(n: number, std = 1): number[] {
    return Array.from({ length: n }, () => this.gauss(0, std));
  }
}

// ── Model functions ───────────────────────────────────────────────

function getModelFunctions(model: ModelSpec) {
  if (model.type === 'pure_p_spin') {
    const { p } = model;
    const E_inf = 2 * Math.sqrt((p - 1) / p);
    const xiPP = (q: number) => p * (p - 1) * Math.max(q, 1e-9) ** (p - 2);
    const edge = (q: number) => -2 * Math.sqrt(xiPP(q));
    const eFormula = `2*sqrt((p-1)/p) = ${E_inf.toFixed(6)}`;
    const edgeFormula = `-2*sqrt(p*(p-1))*q^((p-2)/2)`;
    return { E_inf, xiPP, edge, eFormula, edgeFormula };
  } else {
    const { components } = model;
    const xiPP = (q: number) =>
      components.reduce(
        (s, c) => s + c.beta ** 2 * c.p * (c.p - 1) * Math.max(q, 1e-9) ** (c.p - 2),
        0
      );
    const edge = (q: number) => -2 * Math.sqrt(xiPP(q));
    // Rough E_inf heuristic for demo (real value comes from Parisi formula on backend)
    const E_inf = components.reduce(
      (s, c) => s + Math.abs(c.beta) * 2 * Math.sqrt((c.p - 1) / c.p),
      0
    );
    const compStr = components.map(c => `${c.beta.toFixed(2)}·H_${c.p}`).join(' + ');
    const eFormula = `Parisi E_∞ for ${compStr} ≈ ${E_inf.toFixed(4)} (demo)`;
    const edgeFormula = `-2*sqrt(xi''(q))  xi''(q) = sum beta_p^2*p(p-1)*q^(p-2)`;
    return { E_inf, xiPP, edge, eFormula, edgeFormula };
  }
}

// ── Trajectory generation ─────────────────────────────────────────

function generateTrajectory(N: number, k: number, rng: RNG): number[][] {
  const cur = new Array(N).fill(0);
  const sigmas: number[][] = [];
  const stepSize = Math.sqrt(N / k);

  for (let j = 0; j < k; j++) {
    const targetR = Math.sqrt(N * ((j + 1) / k));
    const v = rng.arr(N);
    // Mild curvature bias
    v[0] += 0.5;
    for (let i = 1; i < Math.min(4, N); i++) v[i] += 0.15 * Math.sin(Math.PI * j / k);
    const vn = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
    for (let i = 0; i < N; i++) cur[i] += stepSize * (v[i] / vn);
    const cn = Math.sqrt(cur.reduce((s, x) => s + x * x, 0));
    const scale = cn > 0 ? targetR / cn : 1;
    sigmas.push(cur.map(x => x * scale));
  }
  return sigmas;
}

// ── Proof annotation ──────────────────────────────────────────────

function proofAnnotation(
  j: number,
  k: number,
  model: ModelSpec
): { tag: ProofTag; katex: string; human: string } {
  const isMixed = model.type === 'mixed_p_spin';

  if (j === 0) {
    return {
      tag: 'boundary_origin',
      katex: '\\sigma_0 = 0 \\implies M = I',
      human:
        'At step 0 the iterate is at the origin; the tangent projection M equals the identity, so the full Hessian is used.',
    };
  }
  if (j === k - 1) {
    return {
      tag: 'terminal_energy',
      katex: '\\frac{H_N(\\sigma_k)}{N} \\to -E_\\infty',
      human:
        'Final step: the energy density converges to −E∞ as N → ∞ (Subag Theorem 4 / Parisi formula).',
    };
  }
  if (isMixed) {
    return {
      tag: 'lemma_3_spectral_edge',
      katex:
        '\\#\\{i : \\lambda_i(\\sigma) \\le -2\\sqrt{\\xi\'\'(q)} + \\varepsilon\\} \\ge N\\delta',
      human:
        'Projected Hessian has ≥ Nδ eigenvalues below −2√ξ″(q). ' +
        'For mixed p-spin: ξ″(q) = Σ β²_p p(p−1) q^{p−2}.',
    };
  }
  return {
    tag: 'lemma_3_spectral_edge',
    katex:
      '\\#\\{i : \\lambda_i(\\sigma) \\le -2\\sqrt{p(p{-}1)}\\,q^{(p-2)/2} + \\varepsilon\\} \\ge N\\delta',
    human:
      'Projected Hessian has ≥ Nδ eigenvalues below the predicted spectral edge; a greedy descent direction exists.',
  };
}

// ── Main generator ────────────────────────────────────────────────

export function generateMockBatches(config: MockConfig): SnapshotBatch[] {
  const { model, n_steps: k, seed = 42 } = config;
  const N = model.N;
  const rng = new RNG(seed);
  const { E_inf, edge, eFormula, edgeFormula } = getModelFunctions(model);

  const sigmas = generateTrajectory(N, k, rng);
  const finalSigma = sigmas[k - 1];

  const snapshots: Snapshot[] = sigmas.map((sigma, j) => {
    const q = (j + 1) / k;
    const edgeVal = edge(q);
    const energy = -E_inf * Math.sqrt(q) + rng.gauss(0, 0.04);
    const lamMin = edgeVal - 0.04 - Math.abs(rng.gauss(0, 0.025));
    const topK = Array.from({ length: 20 }, (_, i) => lamMin + i * 0.035 + rng.gauss(0, 0.008)).sort(
      (a, b) => a - b
    );
    const fullSpec =
      j % 5 === 0
        ? Array.from({ length: N - 1 }, () => rng.gauss(0, 1.0) * 1.8 + lamMin + 2.5).sort(
            (a, b) => a - b
          )
        : null;
    const { tag, katex, human } = proofAnnotation(j, k, model);
    const overlapFinal = sigma.reduce((s, x, i) => s + x * finalSigma[i], 0) / N;

    return {
      index: j,
      j,
      q: +q.toFixed(6),
      radius: +Math.sqrt(N * q).toFixed(6),
      sigma: sigma.map(x => +x.toFixed(6)),
      v_chosen: rng.arr(N).map(x => +x.toFixed(6)),
      energy_density: +energy.toFixed(6),
      lambda_min_tangent: +lamMin.toFixed(6),
      spectral_edge_target: +edgeVal.toFixed(6),
      edge_target_satisfied: lamMin <= edgeVal + 0.01,
      grad_norm_tangent: +Math.abs(rng.gauss(0, 0.005)).toFixed(8),
      step_orthogonality_residual: +Math.abs(rng.gauss(0, 1e-15)).toFixed(20),
      eigvals_topK: topK.map(x => +x.toFixed(6)),
      full_spectrum: fullSpec ? fullSpec.map(x => +x.toFixed(6)) : null,
      overlap_with_start: +(sigma[0] / Math.sqrt(N)).toFixed(6),
      overlap_with_final: +overlapFinal.toFixed(6),
      proof_annotation: {
        active_lemma: tag.split('_')[0],
        tag,
        human_readable: human,
        katex,
      },
    };
  });

  const predictions: RunPredictions = {
    E_infinity: E_inf,
    E_infinity_formula: eFormula,
    spectral_edge_formula: edgeFormula,
  };

  const run_id = `mock_${model.type}_N${N}_k${k}_seed${seed}`;

  // Adaptive batch size: max(1, floor(200/sqrt(N)))
  const batchSize = Math.max(1, Math.floor(200 / Math.sqrt(N)));
  const batches: SnapshotBatch[] = [];

  for (let start = 0; start < k; start += batchSize) {
    const end = Math.min(start + batchSize - 1, k - 1);
    batches.push({
      schema_version: '0.2.0',
      run_id,
      batch_index: batches.length,
      step_range: { start, end },
      total_steps: k,
      is_final: end === k - 1,
      model,
      predictions,
      snapshots: snapshots.slice(start, end + 1),
    });
  }

  return batches;
}
