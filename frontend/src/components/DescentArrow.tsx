import { Line } from '@react-three/drei';
import * as THREE from 'three';
import { projectVector } from '../lib/projection';
import type { PCAResult } from '../lib/pca';

interface Props {
  position: [number, number, number];
  vChosen: number[];
  pcaResult: PCAResult;
}

const ARROW_LEN = 0.2;
const HEAD_BACK = 0.35;
const HEAD_SPREAD = 0.18;

export default function DescentArrow({ position, vChosen, pcaResult }: Props) {
  const proj = projectVector(vChosen, pcaResult);
  const len = Math.sqrt(proj[0] ** 2 + proj[1] ** 2 + proj[2] ** 2);
  if (len < 1e-10) return null;

  const dir = proj.map(x => (x / len) * ARROW_LEN) as [number, number, number];
  const tip: [number, number, number] = [
    position[0] + dir[0],
    position[1] + dir[1],
    position[2] + dir[2],
  ];

  const tipV = new THREE.Vector3(...tip);
  const dirV = new THREE.Vector3(...dir).normalize();
  const perp = new THREE.Vector3(dirV.y, -dirV.x, dirV.z).normalize();

  const base = tipV.clone().sub(dirV.clone().multiplyScalar(ARROW_LEN * HEAD_BACK));
  const L = base.clone().add(perp.clone().multiplyScalar(ARROW_LEN * HEAD_SPREAD));
  const R = base.clone().sub(perp.clone().multiplyScalar(ARROW_LEN * HEAD_SPREAD));

  return (
    <>
      <Line points={[position, tip]} color="#ea580c" lineWidth={2.5} />
      <Line
        points={[[L.x, L.y, L.z], tip, [R.x, R.y, R.z]]}
        color="#ea580c"
        lineWidth={2}
      />
    </>
  );
}
