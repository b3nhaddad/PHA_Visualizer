import { useEffect } from 'react';
import { useStore } from './store';
import Header from './components/Header';
import Scene3D from './components/Scene3D';
import SpectrumPanel from './components/SpectrumPanel';
import ProofPanel from './components/ProofPanel';
import Timeline from './components/Timeline';
import MiniCharts from './components/MiniCharts';
import ParameterPanel from './components/ParameterPanel';
import RunStatusBar from './components/RunStatusBar';

export default function App() {
  const { runData, showParamPanel, setStep, setIsPlaying, currentStep, loadDemo, runConfig } =
    useStore();

  // Load default demo on first mount
  useEffect(() => {
    loadDemo(runConfig);
  }, []);

  // Global keyboard shortcuts
  useEffect(() => {
    const maxStep = (runData?.snapshots.length ?? 1) - 1;
    const handle = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement && e.target.type !== 'range') return;
      switch (e.key) {
        case ' ':
          e.preventDefault();
          setIsPlaying(!useStore.getState().isPlaying);
          break;
        case 'ArrowLeft':
          e.preventDefault();
          setStep(s => s - 1);
          break;
        case 'ArrowRight':
          e.preventDefault();
          setStep(s => s + 1);
          break;
        case 'Home':
          e.preventDefault();
          setStep(0);
          break;
        case 'End':
          e.preventDefault();
          setStep(maxStep);
          break;
      }
    };
    window.addEventListener('keydown', handle);
    return () => window.removeEventListener('keydown', handle);
  }, [runData, currentStep]);

  return (
    <div className="h-screen flex flex-col bg-zinc-200 text-zinc-900 select-none overflow-hidden">
      <Header />
      <RunStatusBar />

      {!runData ? (
        <div className="flex-1 flex items-center justify-center text-zinc-500 text-sm font-mono">
          Generating demo…
        </div>
      ) : (
        <>
          {/* Main area */}
          <div className="flex flex-1 min-h-0 overflow-hidden">
            {/* 3D scene */}
            <div className="flex-1 min-w-0 relative">
              <Scene3D />
            </div>

            {/* Sidebar */}
            <div
              className="flex flex-col border-l border-zinc-300 bg-white overflow-hidden shrink-0"
              style={{ width: 420 }}
            >
              <div className="flex-1 min-h-0 border-b border-zinc-200 overflow-hidden">
                <SpectrumPanel />
              </div>
              <div className="flex-1 min-h-0 overflow-hidden">
                <ProofPanel />
              </div>
            </div>
          </div>

          <Timeline />
          <MiniCharts />
        </>
      )}

      {/* Parameter modal */}
      {showParamPanel && <ParameterPanel />}
    </div>
  );
}
