import type { PCAResult } from './pca';

export function projectVector(
  v: number[],
  pca: PCAResult
): [number, number, number] {
  const { basis, scaleFactor } = pca;
  return [
    dot(v, basis[0]) * scaleFactor,
    dot(v, basis[1]) * scaleFactor,
    dot(v, basis[2]) * scaleFactor,
  ];
}

function dot(a: number[], b: number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}
