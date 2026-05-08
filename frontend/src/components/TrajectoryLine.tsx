import { useMemo } from 'react';
import { Line } from '@react-three/drei';
import * as THREE from 'three';

interface Props {
  projectedPoints: [number, number, number][];
  currentStep: number;
}

export default function TrajectoryLine({ projectedPoints, currentStep }: Props) {
  const visibleCount = currentStep + 1;

  const { points, colors } = useMemo(() => {
    const pts = projectedPoints.slice(0, visibleCount);
    if (pts.length < 2) return { points: pts, colors: [] };

    const cols = pts.map((_, i) => {
      const t = pts.length > 1 ? i / (pts.length - 1) : 0;
      // blue (210°) → violet (270°) on light background
      const h = 0.583 - t * 0.18; // 0.583 = blue, 0.4 = indigo/violet
      const c = new THREE.Color().setHSL(h, 0.85, 0.45);
      return [c.r, c.g, c.b] as [number, number, number];
    });

    return { points: pts, colors: cols };
  }, [projectedPoints, visibleCount]);

  if (points.length < 2) return null;

  return <Line points={points} vertexColors={colors} lineWidth={2.5} />;
}
