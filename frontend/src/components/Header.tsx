import { useStore } from '../store';

function Pill({ label, value, ok }: { label: string; value: string | number; ok?: boolean }) {
  return (
    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-zinc-100 border border-zinc-300 text-xs font-mono">
      <span className="text-zinc-400">{label}</span>
      <span
        className={
          ok === undefined
            ? 'text-zinc-700'
            : ok
            ? 'text-green-700'
            : 'text-red-600'
        }
      >
        {value}
      </span>
    </span>
  );
}

export default function Header() {
  const { runData, setShowParamPanel, showParamPanel } = useStore();

  const model = runData?.model;
  const modelLabel =
    model?.type === 'pure_p_spin'
      ? `Pure p=${model.p}`
      : model?.type === 'mixed_p_spin'
      ? `Mixed (${model.components.map(c => `p${c.p}`).join('+')})`
      : '—';

  return (
    <div className="h-11 flex items-center gap-3 px-4 border-b border-zinc-300 bg-white shrink-0 overflow-x-auto">
      {/* Brand */}
      <span className="font-semibold text-sm text-zinc-800 whitespace-nowrap tracking-tight">
        Hessian Descent Visualizer
      </span>
      <span className="text-zinc-300 text-sm">|</span>
      <span className="text-xs text-zinc-500 font-mono whitespace-nowrap">
        Subag — Full-RSB Spherical Spin Glasses
      </span>

      {/* Run metadata */}
      {runData && (
        <div className="flex items-center gap-2 ml-2">
          <Pill label="model" value={modelLabel} />
          <Pill label="N" value={runData.model.N} />
          <Pill label="k" value={runData.totalSteps} />
          <Pill label="E∞" value={runData.predictions.E_infinity.toFixed(4)} />
          {runData.isComplete && (
            <>
              <Pill
                label="H_final"
                value={runData.snapshots[runData.snapshots.length - 1].energy_density.toFixed(4)}
              />
              <Pill label="valid" value="✓" ok={true} />
            </>
          )}
        </div>
      )}

      {/* Buttons */}
      <div className="flex items-center gap-2 ml-auto shrink-0">
        <button
          onClick={() => setShowParamPanel(!showParamPanel)}
          className="px-3 py-1 text-xs rounded border border-zinc-300 bg-zinc-50 hover:bg-zinc-100 text-zinc-700 transition-colors font-medium"
        >
          Configure
        </button>
      </div>
    </div>
  );
}
