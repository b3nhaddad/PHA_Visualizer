import DescentArrow from './DescentArrow';
import type { PCAResult } from '../lib/pca';

interface Props {
  position: [number, number, number];
  vChosen: number[];
  pcaResult: PCAResult;
}

export default function CurrentIterate({ position, vChosen, pcaResult }: Props) {
  return (
    <>
      <mesh position={position}>
        <sphereGeometry args={[0.042, 14, 10]} />
        <meshBasicMaterial color="#dc2626" />
      </mesh>
      {/* Soft halo */}
      <mesh position={position}>
        <sphereGeometry args={[0.065, 14, 10]} />
        <meshBasicMaterial color="#dc2626" opacity={0.18} transparent />
      </mesh>
      <DescentArrow position={position} vChosen={vChosen} pcaResult={pcaResult} />
    </>
  );
}
