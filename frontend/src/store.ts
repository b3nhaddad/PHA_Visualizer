import { create } from 'zustand';
import type { RunData, SnapshotBatch, ModelSpec } from './api/types';
import type { PCAResult } from './lib/pca';
import { computePCA } from './lib/pca';
import { generateMockBatches } from './lib/mockGenerator';
import { startRun, getRunStatus, getNewBatches, DEFAULT_BACKEND_URL } from './api/client';
import type { StartRunRequest } from './api/types';

// ── Run configuration (UI form state) ─────────────────────────────

export interface RunConfig {
  modelType: 'pure_p_spin' | 'mixed_p_spin';
  N: number;
  n_steps: number;
  seed: number;
  // pure p-spin
  p: number;
  // mixed p-spin
  components: Array<{ p: number; beta: number }>;
}

export const DEFAULT_CONFIG: RunConfig = {
  modelType: 'pure_p_spin',
  N: 40,
  n_steps: 100,
  seed: 42,
  p: 3,
  components: [
    { p: 2, beta: 0.6 },
    { p: 3, beta: 0.8 },
  ],
};

export function configToModelSpec(cfg: RunConfig): ModelSpec {
  if (cfg.modelType === 'pure_p_spin') {
    return { type: 'pure_p_spin', p: cfg.p, N: cfg.N, seed: cfg.seed };
  }
  return { type: 'mixed_p_spin', N: cfg.N, components: cfg.components, seed: cfg.seed };
}

// ── Store ─────────────────────────────────────────────────────────

type ConnectionStatus = 'idle' | 'loading' | 'streaming' | 'complete' | 'error';

interface AppState {
  // Assembled run data
  runData: RunData | null;
  pcaResult: PCAResult | null;
  batches: SnapshotBatch[];
  batchesReceived: number;

  // Playback
  currentStep: number;
  isPlaying: boolean;
  speed: number;

  // Run config (parameter panel state)
  runConfig: RunConfig;
  showParamPanel: boolean;

  // Connection
  connectionMode: 'demo' | 'live';
  connectionStatus: ConnectionStatus;
  connectionError: string | null;
  backendUrl: string;
  activeRunId: string | null;

  // Actions
  loadDemo: (config: RunConfig) => void;
  connectLive: (config: RunConfig) => void;
  appendBatch: (batch: SnapshotBatch) => void;
  setStep: (step: number | ((prev: number) => number)) => void;
  setIsPlaying: (v: boolean) => void;
  setSpeed: (v: number) => void;
  setRunConfig: (cfg: RunConfig) => void;
  setShowParamPanel: (v: boolean) => void;
  setBackendUrl: (url: string) => void;
}

export const useStore = create<AppState>((set, get) => ({
  runData: null,
  pcaResult: null,
  batches: [],
  batchesReceived: 0,

  currentStep: 0,
  isPlaying: false,
  speed: 5,

  runConfig: DEFAULT_CONFIG,
  showParamPanel: false,

  connectionMode: 'demo',
  connectionStatus: 'idle',
  connectionError: null,
  backendUrl: DEFAULT_BACKEND_URL,
  activeRunId: null,

  // ── loadDemo ─────────────────────────────────────────────────────
  loadDemo: (config) => {
    set({ connectionStatus: 'loading', connectionError: null, connectionMode: 'demo' });

    // Yield to let React re-render the loading state before heavy computation
    setTimeout(() => {
      try {
        const batches = generateMockBatches({
          model: configToModelSpec(config),
          n_steps: config.n_steps,
          seed: config.seed,
        });

        const allSnapshots = batches
          .flatMap(b => b.snapshots)
          .sort((a, b) => a.index - b.index);

        const last = batches[batches.length - 1];
        const runData: RunData = {
          run_id: last.run_id,
          model: last.model,
          predictions: last.predictions,
          snapshots: allSnapshots,
          totalSteps: allSnapshots.length,
          isComplete: true,
        };

        const pcaResult = computePCA(allSnapshots.map(s => s.sigma));

        set({
          runData,
          pcaResult,
          batches,
          batchesReceived: batches.length,
          currentStep: 0,
          isPlaying: false,
          connectionStatus: 'complete',
          showParamPanel: false,
          activeRunId: last.run_id,
        });
      } catch (e) {
        set({ connectionStatus: 'error', connectionError: (e as Error).message });
      }
    }, 0);
  },

  // ── connectLive ───────────────────────────────────────────────────
  connectLive: async (config) => {
    const { backendUrl } = get();
    set({
      connectionStatus: 'loading',
      connectionError: null,
      connectionMode: 'live',
      batches: [],
      batchesReceived: 0,
      runData: null,
      pcaResult: null,
      showParamPanel: false,
    });

    try {
      const model = configToModelSpec(config);
      const req: StartRunRequest = {
        model,
        algorithm: { n_steps: config.n_steps, epsilon_spectral: 0.01 },
      };
      const { run_id } = await startRun(backendUrl, req);
      set({ activeRunId: run_id, connectionStatus: 'streaming' });

      // Polling loop
      let nextBatch = 0;
      let done = false;
      while (!done) {
        await new Promise(r => setTimeout(r, 1000));

        const [status, batchResp] = await Promise.all([
          getRunStatus(backendUrl, run_id),
          getNewBatches(backendUrl, run_id, nextBatch),
        ]);

        for (const batch of batchResp.batches) {
          get().appendBatch(batch);
          nextBatch++;
        }

        if (status.status === 'complete' || status.status === 'failed') {
          done = true;
          if (status.status === 'failed') {
            set({ connectionStatus: 'error', connectionError: status.error ?? 'Run failed' });
          } else {
            set({ connectionStatus: 'complete' });
          }
        }
      }
    } catch (e) {
      set({ connectionStatus: 'error', connectionError: (e as Error).message });
    }
  },

  // ── appendBatch ───────────────────────────────────────────────────
  appendBatch: (batch) => {
    const prev = get();
    const batches = [...prev.batches, batch].sort((a, b) => a.batch_index - b.batch_index);

    const allSnapshots = batches
      .flatMap(b => b.snapshots)
      .sort((a, b) => a.index - b.index);

    const last = batches[batches.length - 1];
    const runData: RunData = {
      run_id: last.run_id,
      model: last.model,
      predictions: last.predictions,
      snapshots: allSnapshots,
      totalSteps: last.total_steps,
      isComplete: batch.is_final,
    };

    // Recompute PCA whenever we have new data (≥ 10 snapshots)
    const pcaResult =
      allSnapshots.length >= 10
        ? computePCA(allSnapshots.map(s => s.sigma))
        : prev.pcaResult;

    set({
      batches,
      batchesReceived: batches.length,
      runData,
      pcaResult,
      currentStep: Math.min(prev.currentStep, allSnapshots.length - 1),
    });
  },

  // ── Playback ──────────────────────────────────────────────────────
  setStep: (step) =>
    set(state => {
      const maxStep = (state.runData?.snapshots.length ?? 1) - 1;
      const next = typeof step === 'function' ? step(state.currentStep) : step;
      return { currentStep: Math.max(0, Math.min(maxStep, next)) };
    }),

  setIsPlaying: (v) => set({ isPlaying: v }),
  setSpeed: (v) => set({ speed: v }),

  // ── Config / UI ───────────────────────────────────────────────────
  setRunConfig: (cfg) => set({ runConfig: cfg }),
  setShowParamPanel: (v) => set({ showParamPanel: v }),
  setBackendUrl: (url) => set({ backendUrl: url }),
}));
