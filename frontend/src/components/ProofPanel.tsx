import { useMemo } from 'react';
import katex from 'katex';
import { useStore } from '../store';
import type { ProofTag } from '../api/types';

const TAG_TITLE: Record<ProofTag, string> = {
  boundary_origin: 'Initial Condition — σ₀ = 0',
  tangent_projection: 'Tangent Projection at σ',
  lemma_3_spectral_edge: 'Lemma 3 — Spectral Edge Control',
  theorem_4_descent: 'Theorem 4 — Greedy Descent',
  eq_1_10_step: 'Eq. (1.10) — Iterate Update',
  terminal_energy: 'Terminal Energy Convergence',
};

const TAG_SECTION: Record<ProofTag, string> = {
  boundary_origin: 'Subag (2018) §3',
  tangent_projection: 'Subag (2018) §1.2',
  lemma_3_spectral_edge: 'Subag (2018) §1.2, Lemma 3',
  theorem_4_descent: 'Subag (2018) §1.2, Theorem 4',
  eq_1_10_step: 'Subag (2018) §1.2, Eq. (1.10)',
  terminal_energy: 'Subag (2018) §1.2',
};

const TAG_ACCENT: Record<ProofTag, string> = {
  boundary_origin: 'border-l-violet-500 bg-violet-50',
  tangent_projection: 'border-l-blue-500 bg-blue-50',
  lemma_3_spectral_edge: 'border-l-cyan-500 bg-cyan-50',
  theorem_4_descent: 'border-l-green-500 bg-green-50',
  eq_1_10_step: 'border-l-amber-500 bg-amber-50',
  terminal_energy: 'border-l-orange-500 bg-orange-50',
};

export default function ProofPanel() {
  const { runData, currentStep } = useStore();
  if (!runData) return null;

  const snap = runData.snapshots[currentStep];
  if (!snap) return null;
  const ann = snap.proof_annotation;
  const tag = ann.tag as ProofTag;

  const html = useMemo(() => {
    try {
      return katex.renderToString(ann.katex, { throwOnError: false, displayMode: true });
    } catch {
      return `<span style="color:#dc2626">KaTeX error</span>`;
    }
  }, [ann.katex]);

  const accent = TAG_ACCENT[tag] ?? 'border-l-zinc-400 bg-zinc-50';

  return (
    <div className="h-full flex flex-col p-3 gap-3 overflow-y-auto panel-scroll bg-white">
      {/* Annotation card */}
      <div className={`border-l-4 pl-3 pr-2 py-2 rounded-r ${accent}`}>
        <div className="text-xs font-mono text-zinc-500 mb-0.5">{tag}</div>
        <div className="text-sm font-semibold text-zinc-800 leading-snug">
          {TAG_TITLE[tag] ?? tag}
        </div>
      </div>

      {/* KaTeX */}
      <div
        className="bg-zinc-50 rounded border border-zinc-200 p-3 overflow-x-auto shrink-0"
        dangerouslySetInnerHTML={{ __html: html }}
      />

      {/* Human explanation */}
      <p className="text-xs text-zinc-600 leading-relaxed">{ann.human_readable}</p>

      {/* Sanity metrics */}
      <div className="text-xs font-mono text-zinc-400 space-y-0.5 shrink-0">
        <div>
          ‖M∇H‖ = <span className="text-zinc-600">{snap.grad_norm_tangent.toExponential(2)}</span>
        </div>
        <div>
          orth = <span className="text-zinc-600">{snap.step_orthogonality_residual.toExponential(2)}</span>
        </div>
        <div>
          overlap(σ_k) = <span className="text-zinc-600">{snap.overlap_with_final.toFixed(4)}</span>
        </div>
      </div>

      {/* Footer */}
      <div className="mt-auto pt-2 border-t border-zinc-100 shrink-0">
        <a
          href="https://arxiv.org/abs/1812.04588"
          target="_blank"
          rel="noopener noreferrer"
          className="text-xs text-blue-600 hover:text-blue-700 underline"
        >
          {TAG_SECTION[tag] ?? 'Subag (2018)'} — arXiv:1812.04588
        </a>
      </div>
    </div>
  );
}
