import { useEffect, useRef } from 'react';
import { useStore } from '../store';

export default function Timeline() {
  const { runData, currentStep, isPlaying, speed, setStep, setIsPlaying, setSpeed } = useStore();
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const maxStep = (runData?.snapshots.length ?? 1) - 1;
  const snap = runData?.snapshots[currentStep];

  // Autoplay
  useEffect(() => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    if (!isPlaying) return;
    intervalRef.current = setInterval(() => {
      const { currentStep: cs, runData: rd } = useStore.getState();
      const max = (rd?.snapshots.length ?? 1) - 1;
      if (cs >= max) {
        useStore.getState().setIsPlaying(false);
      } else {
        useStore.getState().setStep(cs + 1);
      }
    }, 1000 / speed);
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [isPlaying, speed]);

  if (!runData) return null;

  return (
    <div className="shrink-0 border-t border-zinc-300 bg-white px-4 py-2 flex items-center gap-4">
      {/* Play/Pause */}
      <button
        onClick={() => setIsPlaying(!isPlaying)}
        className="w-8 h-8 flex items-center justify-center rounded border border-zinc-300 bg-zinc-50 hover:bg-zinc-100 text-zinc-700 transition-colors shrink-0"
        title={isPlaying ? 'Pause (Space)' : 'Play (Space)'}
      >
        {isPlaying ? '⏸' : '▶'}
      </button>

      {/* Counter */}
      <div className="text-xs font-mono text-zinc-500 whitespace-nowrap shrink-0">
        Step{' '}
        <span className="text-zinc-800 font-semibold">{currentStep}</span>
        {' / '}
        <span>{maxStep}</span>
        {snap && <span className="text-zinc-400 ml-2">q = {snap.q.toFixed(3)}</span>}
      </div>

      {/* Scrubber */}
      <input
        type="range"
        min={0}
        max={maxStep}
        value={currentStep}
        onChange={e => setStep(Number(e.target.value))}
        className="flex-1 accent-blue-600 cursor-pointer h-1"
      />

      {/* Speed */}
      <div className="flex items-center gap-2 shrink-0 text-xs text-zinc-500">
        <span>Speed</span>
        <input
          type="range"
          min={1}
          max={30}
          step={1}
          value={speed}
          onChange={e => setSpeed(Number(e.target.value))}
          className="w-20 accent-blue-600 cursor-pointer h-1"
        />
        <span className="w-10 text-zinc-700 font-mono">{speed}/s</span>
      </div>

      <span className="text-xs text-zinc-300 font-mono whitespace-nowrap shrink-0 hidden xl:block">
        Space · ← → · Home End
      </span>
    </div>
  );
}
