import { useStore } from '../store';

const STATUS_LABEL: Record<string, string> = {
  idle: 'Idle',
  loading: 'Generating…',
  streaming: 'Streaming',
  complete: 'Complete',
  error: 'Error',
};

const STATUS_COLOR: Record<string, string> = {
  idle: 'bg-zinc-200 text-zinc-500',
  loading: 'bg-blue-100 text-blue-700',
  streaming: 'bg-indigo-100 text-indigo-700',
  complete: 'bg-green-100 text-green-700',
  error: 'bg-red-100 text-red-700',
};

export default function RunStatusBar() {
  const {
    connectionMode,
    connectionStatus,
    connectionError,
    runData,
    batchesReceived,
    batches,
  } = useStore();

  const totalBatches = batches.length > 0 ? batches[batches.length - 1].batch_index + 1 : '?';
  const stepsComplete = runData?.snapshots.length ?? 0;
  const totalSteps = runData?.totalSteps ?? '?';

  const color = STATUS_COLOR[connectionStatus] ?? STATUS_COLOR.idle;

  return (
    <div className="h-7 flex items-center gap-3 px-4 border-b border-zinc-200 bg-zinc-50 shrink-0 text-xs overflow-hidden">
      {/* Mode badge */}
      <span className="font-mono text-zinc-400">
        {connectionMode === 'demo' ? 'Demo' : 'Live'}
      </span>

      {/* Status badge */}
      <span className={`px-2 py-0.5 rounded-full font-semibold ${color}`}>
        {STATUS_LABEL[connectionStatus] ?? connectionStatus}
      </span>

      {/* Progress */}
      {(connectionStatus === 'streaming' || connectionStatus === 'complete') && (
        <>
          <span className="text-zinc-500">
            Steps{' '}
            <span className="font-mono text-zinc-700">
              {stepsComplete}/{totalSteps}
            </span>
          </span>
          <span className="text-zinc-400">·</span>
          <span className="text-zinc-500">
            Batches{' '}
            <span className="font-mono text-zinc-700">
              {batchesReceived}/{totalBatches}
            </span>
          </span>
        </>
      )}

      {/* Error message */}
      {connectionStatus === 'error' && connectionError && (
        <span className="text-red-600 truncate">{connectionError}</span>
      )}
    </div>
  );
}
