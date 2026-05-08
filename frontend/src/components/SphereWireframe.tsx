interface Props {
  currentQ: number;
}

export default function SphereWireframe({ currentQ }: Props) {
  const currentR = Math.sqrt(Math.max(currentQ, 0));

  return (
    <>
      {/* Unit sphere boundary */}
      <mesh>
        <sphereGeometry args={[1, 28, 20]} />
        <meshBasicMaterial color="#1e3a8a" wireframe opacity={0.12} transparent />
      </mesh>

      {/* Current-radius semi-transparent shell */}
      {currentR > 0.03 && (
        <mesh>
          <sphereGeometry args={[currentR, 22, 16]} />
          <meshBasicMaterial color="#3b82f6" opacity={0.07} transparent side={2} />
        </mesh>
      )}

      {/* Coordinate axes */}
      <axesHelper args={[1.15]} />
    </>
  );
}
