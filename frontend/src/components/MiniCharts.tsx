import { useMemo } from 'react';
import {
  LineChart, Line, XAxis, YAxis, ReferenceLine, ResponsiveContainer, Tooltip,
} from 'recharts';
import { useStore } from '../store';

const TICK = { fill: '#52525b', fontSize: 10 };
const TIP = {
  contentStyle: { background: '#fff', border: '1px solid #d4d4d8', fontSize: 11, borderRadius: 6 },
};

export default function MiniCharts() {
  const { runData, currentStep } = useStore();
  if (!runData) return null;

  const { snapshots, predictions } = runData;
  const currentQ = snapshots[currentStep]?.q ?? 0;

  const energyData = useMemo(
    () => snapshots.map(s => ({ q: s.q, energy: s.energy_density })),
    [snapshots]
  );
  const lambdaData = useMemo(
    () => snapshots.map(s => ({ q: s.q, lam: s.lambda_min_tangent, edge: s.spectral_edge_target })),
    [snapshots]
  );

  const qFmt = (v: unknown) => (v as number).toFixed(1);

  return (
    <div
      className="shrink-0 border-t border-zinc-300 bg-white flex gap-0"
      style={{ height: 120 }}
    >
      {/* Energy */}
      <div className="flex-1 min-w-0 px-3 pt-2 pb-1">
        <div className="text-xs text-zinc-400 font-mono mb-0.5">H_N(σ)/N vs q</div>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={energyData} margin={{ top: 2, right: 8, bottom: 14, left: 0 }}>
            <XAxis dataKey="q" type="number" domain={[0, 1]} tickCount={5} tickFormatter={qFmt} tick={TICK} />
            <YAxis tickCount={4} tickFormatter={v => (v as number).toFixed(2)} tick={TICK} width={38} />
            <Tooltip {...TIP} labelFormatter={v => `q = ${(v as number).toFixed(3)}`}
              formatter={v => [(v as number).toFixed(4), 'H/N']} />
            {/* −E∞ target */}
            <ReferenceLine y={-predictions.E_infinity} stroke="#16a34a" strokeDasharray="5 3" strokeWidth={1} />
            {/* Current step */}
            <ReferenceLine x={currentQ} stroke="#7c3aed" strokeWidth={1} opacity={0.7} />
            <Line type="monotone" dataKey="energy" stroke="#2563eb" dot={false} strokeWidth={1.5} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="w-px bg-zinc-200 self-stretch shrink-0" />

      {/* λ_min + edge */}
      <div className="flex-1 min-w-0 px-3 pt-2 pb-1">
        <div className="text-xs text-zinc-400 font-mono mb-0.5">λ_min & edge vs q</div>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={lambdaData} margin={{ top: 2, right: 8, bottom: 14, left: 0 }}>
            <XAxis dataKey="q" type="number" domain={[0, 1]} tickCount={5} tickFormatter={qFmt} tick={TICK} />
            <YAxis tickCount={4} tickFormatter={v => (v as number).toFixed(1)} tick={TICK} width={38} />
            <Tooltip {...TIP} labelFormatter={v => `q = ${(v as number).toFixed(3)}`}
              formatter={(v, name) => [(v as number).toFixed(4), name === 'lam' ? 'λ_min' : 'edge']} />
            <ReferenceLine x={currentQ} stroke="#7c3aed" strokeWidth={1} opacity={0.7} />
            <Line type="monotone" dataKey="lam" stroke="#dc2626" dot={false} strokeWidth={1.5} isAnimationActive={false} />
            <Line type="monotone" dataKey="edge" stroke="#d97706" dot={false} strokeWidth={1.5} strokeDasharray="4 3" isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
