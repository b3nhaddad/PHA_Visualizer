import { useMemo } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, ReferenceLine, ResponsiveContainer, Tooltip,
  ScatterChart, Scatter,
} from 'recharts';
import { useStore } from '../store';

function histBins(values: number[], nBins = 36) {
  const min = Math.min(...values);
  const max = Math.max(...values);
  if (max === min) return [{ x: min, count: values.length }];
  const w = (max - min) / nBins;
  const bins = Array.from({ length: nBins }, (_, i) => ({
    x: +(min + (i + 0.5) * w).toFixed(3),
    count: 0,
  }));
  for (const v of values) {
    const idx = Math.min(Math.floor((v - min) / w), nBins - 1);
    if (idx >= 0) bins[idx].count++;
  }
  return bins;
}

const TIP_STYLE = {
  contentStyle: { background: '#fff', border: '1px solid #d4d4d8', fontSize: 11, borderRadius: 6 },
  labelStyle: { color: '#3f3f46' },
};
const TICK = { fill: '#52525b', fontSize: 10 };
const tickFmt = (v: unknown) => (v as number).toFixed(1);

export default function SpectrumPanel() {
  const { runData, currentStep } = useStore();
  if (!runData) return null;

  const snap = runData.snapshots[currentStep];
  if (!snap) return null;
  const { eigvals_topK, spectral_edge_target, lambda_min_tangent, q } = snap;

  // Walk back to find the most recent full_spectrum so the histogram
  // doesn't disappear on non-5 steps during fast playback.
  const latestFullSpectrum = useMemo(() => {
    for (let i = currentStep; i >= 0; i--) {
      if (runData.snapshots[i]?.full_spectrum) return runData.snapshots[i].full_spectrum!;
    }
    return null;
  }, [runData.snapshots, currentStep]);

  const histData = useMemo(
    () => (latestFullSpectrum ? histBins(latestFullSpectrum) : null),
    [latestFullSpectrum]
  );

  // Two-row jitter: even indices top row, odd indices bottom row.
  const topKData = useMemo(
    () => eigvals_topK.map((v, i) => ({ x: v, y: i % 2 === 0 ? 0.3 : 0.7 })),
    [eigvals_topK]
  );

  // Each chart scales independently to its own data.
  const pad = 0.15;
  const topKMin = Math.min(...eigvals_topK) - pad;
  const topKMax = Math.max(...eigvals_topK) + pad;

  const histVals = latestFullSpectrum ?? eigvals_topK;
  const histMin = Math.min(...histVals, spectral_edge_target) - pad;
  const histMax = Math.max(...histVals) + pad;

  return (
    <div className="h-full flex flex-col p-3 gap-2 bg-white overflow-hidden">
      {/* Title row */}
      <div className="flex items-center justify-between shrink-0">
        <span className="text-xs font-semibold text-zinc-700">
          Hessian Spectrum — q = {q.toFixed(3)}
        </span>
        <span
          className={`text-xs px-2 py-0.5 rounded-full font-semibold border ${
            snap.edge_target_satisfied
              ? 'bg-green-50 text-green-700 border-green-300'
              : 'bg-red-50 text-red-700 border-red-300'
          }`}
        >
          {snap.edge_target_satisfied ? 'edge ✓' : 'edge ✗'}
        </span>
      </div>

      {/* Legend */}
      <div className="flex gap-3 text-xs shrink-0">
        <span className="flex items-center gap-1">
          <span className="inline-block w-4 h-0.5 bg-amber-500" />
          <span className="text-zinc-500">predicted edge</span>
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block w-4 h-0.5 bg-red-500" />
          <span className="text-zinc-500">λ_min</span>
        </span>
      </div>

      {/* Top-K scatter — always visible */}
      <div className="shrink-0" style={{ height: 90 }}>
        <p className="text-xs text-zinc-400 mb-0.5">Top-{eigvals_topK.length} eigenvalues</p>
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 4, right: 8, bottom: 14, left: 0 }}>
            <XAxis type="number" dataKey="x" domain={[topKMin, topKMax]} tickCount={5}
              tickFormatter={tickFmt} tick={TICK} />
            <YAxis type="number" dataKey="y" hide domain={[0, 1]} padding={{ top: 10, bottom: 10 }} />
            <Tooltip {...TIP_STYLE}
              formatter={(v, name) => [name === 'x' ? (v as number).toFixed(4) : v, name === 'x' ? 'λ' : name]} />
            <Scatter data={topKData} fill="#3b82f6" isAnimationActive={false} r={3} />
            <ReferenceLine x={spectral_edge_target} stroke="#d97706" strokeWidth={1.5} strokeDasharray="4 3" />
            <ReferenceLine x={lambda_min_tangent} stroke="#dc2626" strokeWidth={1.5} />
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      <div className="w-full h-px bg-zinc-100 shrink-0" />

      {/* Distribution histogram — holds last stored full_spectrum */}
      <div className="flex-1 min-h-0">
        <p className="text-xs text-zinc-400 mb-0.5">
          {histData
            ? `Full distribution (${latestFullSpectrum!.length} eigvals)`
            : 'Full distribution — waiting for first spectrum…'}
        </p>
        {histData ? (
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={histData} margin={{ top: 2, right: 8, bottom: 14, left: 0 }}>
              <XAxis dataKey="x" type="number" domain={[histMin, histMax]} tickCount={5}
                tickFormatter={tickFmt} tick={TICK} />
              <YAxis hide />
              <Tooltip {...TIP_STYLE} labelFormatter={v => `λ ≈ ${(v as number).toFixed(3)}`}
                formatter={v => [v, 'count']} />
              <Bar dataKey="count" fill="#93c5fd" isAnimationActive={false} />
              <ReferenceLine x={spectral_edge_target} stroke="#d97706" strokeWidth={1.5} strokeDasharray="4 3" />
              <ReferenceLine x={lambda_min_tangent} stroke="#dc2626" strokeWidth={1.5} />
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <div className="flex items-center justify-center h-full text-zinc-300 text-xs">
            stored every 5 steps
          </div>
        )}
      </div>

      {/* Footer stats */}
      <div className="text-xs text-zinc-500 font-mono shrink-0 flex gap-4 border-t border-zinc-100 pt-1">
        <span>λ_min = <span className="text-red-600">{lambda_min_tangent.toFixed(4)}</span></span>
        <span>edge = <span className="text-amber-600">{spectral_edge_target.toFixed(4)}</span></span>
      </div>
    </div>
  );
}
