import { useState } from 'react';
import { useStore, type RunConfig, DEFAULT_CONFIG, configToModelSpec } from '../store';
import { pingBackend } from '../api/client';

function Slider({
  label, value, min, max, step = 1, fmt, onChange,
}: {
  label: string; value: number; min: number; max: number; step?: number;
  fmt?: (v: number) => string; onChange: (v: number) => void;
}) {
  return (
    <div className="flex items-center gap-3">
      <span className="w-16 text-xs text-zinc-600 shrink-0">{label}</span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={e => onChange(+e.target.value)}
        className="flex-1 accent-blue-600 h-1 cursor-pointer"
      />
      <span className="w-14 text-xs font-mono text-zinc-700 text-right">
        {fmt ? fmt(value) : value}
      </span>
    </div>
  );
}

export default function ParameterPanel() {
  const { setShowParamPanel, loadDemo, connectLive, backendUrl, setBackendUrl, connectionStatus } =
    useStore();

  const [cfg, setCfg] = useState<RunConfig>({ ...DEFAULT_CONFIG });
  const [pingState, setPingState] = useState<'idle' | 'ok' | 'fail'>('idle');
  const [urlInput, setUrlInput] = useState(backendUrl);

  const update = (patch: Partial<RunConfig>) => setCfg(c => ({ ...c, ...patch }));

  const addComponent = () =>
    update({ components: [...cfg.components, { p: 4, beta: 0.5 }] });

  const removeComponent = (i: number) =>
    update({ components: cfg.components.filter((_, idx) => idx !== i) });

  const updateComponent = (i: number, field: 'p' | 'beta', value: number) =>
    update({
      components: cfg.components.map((c, idx) =>
        idx === i ? { ...c, [field]: value } : c
      ),
    });

  const handlePing = async () => {
    setPingState('idle');
    const ok = await pingBackend(urlInput);
    setPingState(ok ? 'ok' : 'fail');
    if (ok) setBackendUrl(urlInput);
  };

  const isLoading = connectionStatus === 'loading' || connectionStatus === 'streaming';

  return (
    // Backdrop
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-zinc-900/30 backdrop-blur-sm"
      onClick={e => { if (e.target === e.currentTarget) setShowParamPanel(false); }}
    >
      {/* Modal card */}
      <div className="bg-white rounded-xl shadow-xl border border-zinc-200 w-[480px] max-h-[90vh] overflow-y-auto panel-scroll">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-zinc-100">
          <h2 className="font-semibold text-zinc-800 text-sm">Run Configuration</h2>
          <button
            onClick={() => setShowParamPanel(false)}
            className="text-zinc-400 hover:text-zinc-600 text-lg leading-none"
          >
            ×
          </button>
        </div>

        <div className="px-5 py-4 space-y-5">
          {/* ── Model type ──────────────────────────────────────── */}
          <section>
            <label className="block text-xs font-semibold text-zinc-500 uppercase tracking-wide mb-2">
              Model Type
            </label>
            <div className="flex gap-2">
              {(['pure_p_spin', 'mixed_p_spin'] as const).map(t => (
                <button
                  key={t}
                  onClick={() => update({ modelType: t })}
                  className={`flex-1 py-2 rounded-lg text-xs font-semibold border transition-colors ${
                    cfg.modelType === t
                      ? 'bg-blue-600 text-white border-blue-600'
                      : 'bg-zinc-50 text-zinc-600 border-zinc-300 hover:bg-zinc-100'
                  }`}
                >
                  {t === 'pure_p_spin' ? 'Pure p-Spin' : 'Mixed p-Spin'}
                </button>
              ))}
            </div>
            <p className="text-xs text-zinc-400 mt-2 leading-snug">
              {cfg.modelType === 'pure_p_spin'
                ? 'Pure p-spin: H(σ) = N^{-(p-1)/2} Σ J·σ^p. Ground state from Theorem 4.'
                : 'Mixed: H(σ) = Σ β_p H_p(σ). Full-RSB following Subag (2018). E∞ from Parisi formula (backend).'}
            </p>
          </section>

          {/* ── Common sliders ──────────────────────────────────── */}
          <section className="space-y-3">
            <label className="block text-xs font-semibold text-zinc-500 uppercase tracking-wide mb-2">
              Dimensions
            </label>
            <Slider label="N" value={cfg.N} min={10} max={300} onChange={v => update({ N: v })} />
            <Slider label="Steps k" value={cfg.n_steps} min={50} max={500} step={10}
              onChange={v => update({ n_steps: v })} />
            <div className="flex items-center gap-3">
              <span className="w-16 text-xs text-zinc-600">Seed</span>
              <input
                type="number"
                value={cfg.seed}
                onChange={e => update({ seed: +e.target.value })}
                className="w-24 px-2 py-1 text-xs font-mono border border-zinc-300 rounded bg-zinc-50 text-zinc-800"
              />
            </div>
          </section>

          {/* ── Model-specific ──────────────────────────────────── */}
          {cfg.modelType === 'pure_p_spin' ? (
            <section className="space-y-3">
              <label className="block text-xs font-semibold text-zinc-500 uppercase tracking-wide mb-2">
                Pure p-Spin Parameters
              </label>
              <Slider label="p (order)" value={cfg.p} min={2} max={7}
                fmt={v => `p = ${v}`} onChange={v => update({ p: v })} />
              <p className="text-xs text-zinc-400">
                E∞ = 2√((p−1)/p) ={' '}
                <span className="font-mono">{(2 * Math.sqrt((cfg.p - 1) / cfg.p)).toFixed(5)}</span>
              </p>
            </section>
          ) : (
            <section className="space-y-3">
              <label className="block text-xs font-semibold text-zinc-500 uppercase tracking-wide mb-2">
                Mixture Components
              </label>
              {cfg.components.map((c, i) => (
                <div key={i} className="flex items-center gap-2 bg-zinc-50 border border-zinc-200 rounded-lg px-3 py-2">
                  <span className="text-xs text-zinc-500 font-mono w-4">{i + 1}.</span>
                  <div className="flex items-center gap-1.5">
                    <span className="text-xs text-zinc-500">p =</span>
                    <select
                      value={c.p}
                      onChange={e => updateComponent(i, 'p', +e.target.value)}
                      className="text-xs font-mono border border-zinc-300 rounded px-1 py-0.5 bg-white"
                    >
                      {[2, 3, 4, 5, 6, 7].map(p => (
                        <option key={p} value={p}>{p}</option>
                      ))}
                    </select>
                  </div>
                  <div className="flex items-center gap-1.5 flex-1">
                    <span className="text-xs text-zinc-500">β =</span>
                    <input
                      type="number"
                      value={c.beta}
                      step={0.1}
                      min={0.01}
                      onChange={e => updateComponent(i, 'beta', +e.target.value)}
                      className="w-20 text-xs font-mono border border-zinc-300 rounded px-1.5 py-0.5 bg-white"
                    />
                  </div>
                  <button
                    onClick={() => removeComponent(i)}
                    disabled={cfg.components.length <= 1}
                    className="text-zinc-400 hover:text-red-500 disabled:opacity-30 text-sm leading-none ml-auto"
                  >
                    ×
                  </button>
                </div>
              ))}
              <button
                onClick={addComponent}
                className="text-xs text-blue-600 hover:text-blue-700 font-medium"
              >
                + Add component
              </button>
              <p className="text-xs text-zinc-400">
                ξ″(q) = Σ β²p(p−1)q^(p−2) &nbsp;|&nbsp; E∞ from Parisi (backend provides)
              </p>
            </section>
          )}

          {/* ── Backend URL ──────────────────────────────────────── */}
          <section>
            <label className="block text-xs font-semibold text-zinc-500 uppercase tracking-wide mb-2">
              Backend (for live runs)
            </label>
            <div className="flex gap-2">
              <input
                type="text"
                value={urlInput}
                onChange={e => setUrlInput(e.target.value)}
                placeholder="http://localhost:8000"
                className="flex-1 text-xs font-mono px-2 py-1.5 border border-zinc-300 rounded bg-zinc-50 text-zinc-800"
              />
              <button
                onClick={handlePing}
                className="px-3 py-1 text-xs rounded border border-zinc-300 bg-zinc-50 hover:bg-zinc-100 text-zinc-700 transition-colors"
              >
                Ping
              </button>
            </div>
            {pingState === 'ok' && (
              <p className="text-xs text-green-600 mt-1">✓ Backend reachable</p>
            )}
            {pingState === 'fail' && (
              <p className="text-xs text-red-600 mt-1">
                ✗ Could not reach {urlInput} — implement GET /api/health → {'{ status: "ok" }'}
              </p>
            )}
          </section>
        </div>

        {/* Actions */}
        <div className="px-5 py-4 border-t border-zinc-100 flex gap-3">
          <button
            onClick={() => loadDemo(cfg)}
            disabled={isLoading}
            className="flex-1 py-2 text-xs font-semibold rounded-lg bg-zinc-800 text-white hover:bg-zinc-700 disabled:opacity-50 transition-colors"
          >
            Load Demo
          </button>
          <button
            onClick={() => {
              setBackendUrl(urlInput);
              connectLive(cfg);
            }}
            disabled={isLoading}
            className="flex-1 py-2 text-xs font-semibold rounded-lg bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50 transition-colors"
          >
            Run on Backend
          </button>
        </div>
      </div>
    </div>
  );
}
