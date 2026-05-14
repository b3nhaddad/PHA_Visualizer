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

export default function SpectrumPanel() {
  const { runData, currentStep } = useStore();
  if (!runData) return null;

  const snap = runData.snapshots[currentStep];
  if (!snap) return null;
  const { full_spectrum, eigvals_topK, spectral_edge_target, lambda_min_tangent, q } = snap;

  const histData = useMemo(
    () => (full_spectrum ? histBins(full_spectrum) : null),
    [full_spectrum]
  );

  const topKData = useMemo(
    () => eigvals_topK.map(v => ({ x: v, y: 0.5 })),
    [eigvals_topK]
  );

  const allVals = full_spectrum ?? eigvals_topK;
  const xMin = Math.min(...allVals, spectral_edge_target) - 0.3;
  const xMax = Math.max(...allVals) + 0.3;

  const tickFmt = (v: unknown) => (v as number).toFixed(1);
  const tickStyle = { fill: '#52525b', fontSize: 10 };

  return (
    <div className="h-full flex flex-col p-3 gap-2 bg-white">
      {/* Title */}
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

      {/* Chart */}
      <div className="flex-1 min-h-0">
        {histData ? (
          <>
            <p className="text-xs text-zinc-400 mb-1">
              Full spectrum ({full_spectrum!.length} tangent eigvals)
            </p>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={histData} margin={{ top: 2, right: 8, bottom: 16, left: 0 }}>
                <XAxis dataKey="x" type="number" domain={[xMin, xMax]} tickCount={5}
                  tickFormatter={tickFmt} tick={tickStyle} />
                <YAxis hide />
                <Tooltip {...TIP_STYLE} labelFormatter={v => `λ ≈ ${(v as number).toFixed(3)}`}
                  formatter={v => [v, 'count']} />
                <Bar dataKey="count" fill="#93c5fd" isAnimationActive={false} />
                <ReferenceLine x={spectral_edge_target} stroke="#d97706" strokeWidth={1.5} strokeDasharray="4 3" />
                <ReferenceLine x={lambda_min_tangent} stroke="#dc2626" strokeWidth={1.5} />
              </BarChart>
            </ResponsiveContainer>
          </>
        ) : (
          <>
            <p className="text-xs text-zinc-400 mb-1">Top-20 eigenvalues (full spectrum not stored)</p>
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 16, right: 8, bottom: 16, left: 0 }}>
                <XAxis type="number" dataKey="x" domain={[xMin, xMax]} tickCount={5}
                  tickFormatter={tickFmt} tick={tickStyle} />
                <YAxis type="number" dataKey="y" hide domain={[0, 1]} />
                <Tooltip {...TIP_STYLE}
                  formatter={(v, name) => [name === 'x' ? (v as number).toFixed(4) : v, name === 'x' ? 'λ' : name]} />
                <Scatter data={topKData} fill="#3b82f6" isAnimationActive={false} />
                <ReferenceLine x={spectral_edge_target} stroke="#d97706" strokeWidth={1.5} strokeDasharray="4 3" />
                <ReferenceLine x={lambda_min_tangent} stroke="#dc2626" strokeWidth={1.5} />
              </ScatterChart>
            </ResponsiveContainer>
          </>
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
