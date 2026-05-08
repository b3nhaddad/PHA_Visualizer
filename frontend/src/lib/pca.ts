import { Matrix } from 'ml-matrix';

export interface PCAResult {
  basis: [number[], number[], number[]];
  scaleFactor: number;
  projectedPoints: [number, number, number][];
}

export function computePCA(sigmas: number[][]): PCAResult {
  const k = sigmas.length;
  const N = sigmas[0].length;

  // Center the data
  const mean = Array.from({ length: N }, (_, i) =>
    sigmas.reduce((s, row) => s + row[i], 0) / k
  );
  const centered = sigmas.map(row => row.map((v, i) => v - mean[i]));

  // Build covariance X^T X (N × N), then find top-3 eigenvectors via
  // power iteration + deflation. This avoids SVD API shape uncertainty.
  const X = new Matrix(centered); // k × N
  const XTX = X.transpose().mmul(X); // N × N  (symmetric, PSD)

  const basis = powerIteration3(XTX, N);

  const projected: [number, number, number][] = sigmas.map(sigma => [
    dot(sigma, basis[0]),
    dot(sigma, basis[1]),
    dot(sigma, basis[2]),
  ]);

  const last = projected[projected.length - 1];
  const lastR = Math.sqrt(last[0] ** 2 + last[1] ** 2 + last[2] ** 2);
  const scaleFactor = lastR > 1e-10 ? 1 / lastR : 1;

  return {
    basis,
    scaleFactor,
    projectedPoints: projected.map(([x, y, z]) => [
      x * scaleFactor,
      y * scaleFactor,
      z * scaleFactor,
    ]),
  };
}

function dot(a: number[], b: number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function matVec(M: Matrix, v: number[]): number[] {
  const n = v.length;
  return Array.from({ length: n }, (_, i) => {
    let s = 0;
    for (let j = 0; j < n; j++) s += M.get(i, j) * v[j];
    return s;
  });
}

// Seeded LCG so the projection is deterministic across renders
function seededVec(n: number, seed: number): number[] {
  let s = seed;
  return Array.from({ length: n }, () => {
    s = (1664525 * s + 1013904223) & 0xffffffff;
    return ((s >>> 0) / 0xffffffff) * 2 - 1;
  });
}

function normalise(v: number[]): number[] {
  const n = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
  return n > 1e-12 ? v.map(x => x / n) : v;
}

function deflate(M: Matrix, v: number[], lambda: number): Matrix {
  const n = v.length;
  const vvT = new Matrix(n, n);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++)
      vvT.set(i, j, v[i] * v[j] * lambda);
  return M.sub(vvT);
}

function powerIteration3(M: Matrix, N: number): [number[], number[], number[]] {
  const basis: number[][] = [];
  let current = M;

  for (let comp = 0; comp < 3; comp++) {
    let v = normalise(seededVec(N, comp + 1));

    // Orthogonalise initial guess against already-found directions
    for (const b of basis) {
      const d = dot(v, b);
      v = normalise(v.map((x, i) => x - d * b[i]));
    }

    // Power iteration
    for (let iter = 0; iter < 200; iter++) {
      let Av = matVec(current, v);
      // Re-orthogonalise
      for (const b of basis) {
        const d = dot(Av, b);
        Av = Av.map((x, i) => x - d * b[i]);
      }
      const prev = v;
      v = normalise(Av);
      // Convergence check
      const diff = v.reduce((s, x, i) => s + (x - prev[i]) ** 2, 0);
      if (diff < 1e-20) break;
    }

    const lambda = dot(v, matVec(current, v));
    basis.push(v);
    current = deflate(current, v, lambda);
  }

  return basis as [number[], number[], number[]];
}
