/**
 * ================================================================
 * Spherical Hessian Descent Visualizer — Backend API Contract
 * schema_version "0.2.0"
 * ================================================================
 *
 * Implement three endpoints (base prefix: /api):
 *
 *   POST  /api/runs
 *         Body:    StartRunRequest
 *         Returns: StartRunResponse
 *
 *   GET   /api/runs/:run_id/status
 *         Returns: RunStatusResponse
 *
 *   GET   /api/runs/:run_id/batches?from_batch=<n>
 *         Returns: BatchesResponse
 *         (from_batch is 0-indexed; return all batches with batch_index >= from_batch)
 *
 * Polling flow:
 *   1. POST /runs → receive run_id, batch_size
 *   2. Poll GET /runs/:id/batches?from_batch=0 every ~1 s
 *   3. Accumulate batches; stop when is_final=true in any batch
 *      OR status.status === 'complete' | 'failed'
 *
 * Adaptive batch size (recommended):
 *   batch_size = max(1, floor(200 / sqrt(N)))
 *   e.g. N=40 → 31  |  N=100 → 20  |  N=300 → 11
 * ================================================================
 */

// ── Model Specifications ──────────────────────────────────────────

/**
 * Pure p-spin spherical glass.
 * H(σ) = N^{-(p-1)/2} Σ_{i_1…i_p} J_{i_1…i_p} σ_{i_1}…σ_{i_p}
 * Covariance: ξ(q) = q^p
 * Ground-state energy: E_∞ = 2√((p-1)/p)
 */
export interface PureSpinSpec {
  type: 'pure_p_spin';
  p: number;     // spin order ≥ 2 (integer)
  N: number;     // ambient dimension
  seed?: number; // RNG seed for the disorder J
}

/**
 * Mixed p-spin spherical glass (Subag 2018 — full-RSB regime).
 * H(σ) = Σ_p β_p H_p(σ)
 * Covariance: ξ(q) = Σ_p β_p² q^p
 * E_∞: provided by backend (Parisi variational formula — non-trivial)
 */
export interface MixedSpinSpec {
  type: 'mixed_p_spin';
  N: number;
  components: Array<{
    p: number;    // integer ≥ 2
    beta: number; // β_p coefficient (positive)
  }>;
  seed?: number;
}

export type ModelSpec = PureSpinSpec | MixedSpinSpec;

// ── POST /api/runs ────────────────────────────────────────────────

export interface StartRunRequest {
  model: ModelSpec;
  algorithm: {
    /** Number of discrete algorithm steps k (Subag Theorem 4). */
    n_steps: number;
    /** ε for spectral-edge verification. Default: 0.01 */
    epsilon_spectral?: number;
    /**
     * Hint for batch size. Backend may override.
     * If omitted, backend uses: max(1, floor(200/sqrt(N))).
     */
    batch_size?: number;
  };
}

export interface StartRunResponse {
  run_id: string;
  status: 'queued' | 'running';
  total_steps: number;
  /** Actual batch_size the backend will use. */
  batch_size: number;
}

// ── GET /api/runs/:run_id/status ──────────────────────────────────

export interface RunStatusResponse {
  run_id: string;
  status: 'running' | 'complete' | 'failed';
  steps_complete: number;
  total_steps: number;
  /** Number of fully computed batches ready to fetch. */
  batches_available: number;
  /** Present only if status === 'failed'. */
  error?: string;
}

// ── GET /api/runs/:run_id/batches?from_batch=<n> ──────────────────

export interface BatchesResponse {
  run_id: string;
  /** All batches with batch_index >= from_batch that are ready. May be []. */
  batches: SnapshotBatch[];
}

// ── Snapshot Batch ────────────────────────────────────────────────

export interface SnapshotBatch {
  schema_version: '0.2.0';
  run_id: string;
  /** 0-indexed batch sequence number. */
  batch_index: number;
  /** Inclusive range of algorithm step indices covered by this batch. */
  step_range: { start: number; end: number };
  /** Total algorithm steps in the whole run. */
  total_steps: number;
  /** true for the last batch — signals the frontend to stop polling. */
  is_final: boolean;
  /** Repeated in every batch so each batch is self-contained. */
  model: ModelSpec;
  predictions: RunPredictions;
  snapshots: Snapshot[];
}

export interface RunPredictions {
  /** Ground-state energy density magnitude: H_N(σ_k)/N → −E_∞ */
  E_infinity: number;
  /** Human-readable formula, e.g. "2*sqrt((p-1)/p)" */
  E_infinity_formula: string;
  /**
   * Human-readable formula for the Lemma 3 spectral edge.
   * Pure p-spin:  "-2*sqrt(p*(p-1))*q^((p-2)/2)"
   * Mixed p-spin: "-2*sqrt(xi''(q))"  where xi''(q) = Σ β_p² p(p-1) q^(p-2)
   */
  spectral_edge_formula: string;
}

// ── Per-Step Snapshot ─────────────────────────────────────────────

export type ProofTag =
  | 'boundary_origin'
  | 'tangent_projection'
  | 'lemma_3_spectral_edge'
  | 'theorem_4_descent'
  | 'eq_1_10_step'
  | 'terminal_energy';

export interface ProofAnnotation {
  active_lemma: string;
  tag: ProofTag;
  human_readable: string;
  /** LaTeX source fed directly to KaTeX. Single backslashes (JSON-escaped as \\). */
  katex: string;
}

export interface Snapshot {
  /** 0-indexed position in the full run (NOT local to this batch). */
  index: number;
  j: number;
  /** q = ‖σ‖²/N ∈ (0, 1]. */
  q: number;
  /** √(N·q) — convenience for 3D rendering. */
  radius: number;

  /** Full iterate in ℝ^N. Required for client-side PCA. Length = N. */
  sigma: number[];
  /** Greedy descent direction in ℝ^N. Used for arrow rendering. Length = N. */
  v_chosen: number[];

  energy_density: number;
  lambda_min_tangent: number;
  /**
   * Lemma 3 spectral edge prediction.
   * Pure:  −2√(p(p−1)) q^((p−2)/2)
   * Mixed: −2√(ξ''(q)) where ξ''(q) = Σ β_p² p(p−1) q^(p−2)
   */
  spectral_edge_target: number;
  edge_target_satisfied: boolean;

  /** ‖M∇H‖ — tangent gradient norm. Should approach 0 near sphere. */
  grad_norm_tangent: number;
  /** |σ_j · v_j| / ‖σ_j‖ — should be ~1e-14 (orthogonality check). */
  step_orthogonality_residual: number;

  /** K most-negative tangent-Hessian eigenvalues (K ≈ 20). Always present. */
  eigvals_topK: number[];
  /** Full N−1 tangent eigenvalues. Null when skipped (e.g. every 5th step only). */
  full_spectrum: number[] | null;

  overlap_with_start: number;
  overlap_with_final: number;

  proof_annotation: ProofAnnotation;
}

// ── Aggregate type assembled by the frontend ──────────────────────

/** RunData is the frontend's assembled view — built from one or more SnapshotBatches. */
export interface RunData {
  run_id: string;
  model: ModelSpec;
  predictions: RunPredictions;
  snapshots: Snapshot[];
  totalSteps: number;
  isComplete: boolean;
}
