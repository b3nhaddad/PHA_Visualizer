import { Suspense } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { useStore } from '../store';
import SphereWireframe from './SphereWireframe';
import TrajectoryLine from './TrajectoryLine';
import CurrentIterate from './CurrentIterate';

function SceneContents() {
  const { runData, pcaResult, currentStep } = useStore();
  if (!runData || !pcaResult || pcaResult.projectedPoints.length === 0) return null;

  const snap = runData.snapshots[currentStep];
  if (!snap) return null;

  return (
    <>
      <ambientLight intensity={1.2} />
      <directionalLight position={[3, 4, 3]} intensity={0.6} />

      <SphereWireframe currentQ={snap.q} />

      <TrajectoryLine
        projectedPoints={pcaResult.projectedPoints}
        currentStep={Math.min(currentStep, pcaResult.projectedPoints.length - 1)}
      />

      <CurrentIterate
        position={pcaResult.projectedPoints[Math.min(currentStep, pcaResult.projectedPoints.length - 1)]}
        vChosen={snap.v_chosen}
        pcaResult={pcaResult}
      />
    </>
  );
}

export default function Scene3D() {
  const { runData } = useStore();
  const N = runData?.model.N ?? '?';

  return (
    // Absolute-fill container inside a relative parent — prevents canvas resize glitches
    <div className="absolute inset-0">
      <Canvas
        camera={{ position: [1.8, 1.2, 1.8], fov: 48, near: 0.01, far: 100 }}
        gl={{ antialias: true }}
        resize={{ debounce: 0 }}
        style={{ width: '100%', height: '100%' }}
      >
        <color attach="background" args={['#eef0f6']} />
        <Suspense fallback={null}>
          <SceneContents />
        </Suspense>
        <OrbitControls
          makeDefault
          enablePan
          enableZoom
          enableRotate
          dampingFactor={0.1}
          enableDamping
        />
      </Canvas>

      {/* PCA label */}
      <div className="absolute bottom-3 left-3 text-xs text-zinc-400 font-mono pointer-events-none leading-snug bg-white/60 px-2 py-1 rounded backdrop-blur-sm">
        PCA projection S<sup>N−1</sup> ⊂ ℝ<sup>N</sup> (N={N}) → ℝ³
      </div>
    </div>
  );
}
