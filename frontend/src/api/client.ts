/**
 * ================================================================
 * API Client — Spherical Hessian Descent Visualizer
 * ================================================================
 * Backend integration checklist:
 *
 *  1. Start your server at DEFAULT_BACKEND_URL (or configure in the UI)
 *  2. Implement POST  /api/runs          → StartRunResponse
 *  3. Implement GET   /api/runs/:id/status → RunStatusResponse
 *  4. Implement GET   /api/runs/:id/batches?from_batch=<n> → BatchesResponse
 *  5. Click "Connect to Backend" in the visualizer
 *
 * All types are in src/api/types.ts — copy that file to your backend
 * as the source-of-truth schema.
 * ================================================================
 */

import type {
  StartRunRequest,
  StartRunResponse,
  RunStatusResponse,
  BatchesResponse,
} from './types';

/**
 * Empty string → requests go to the same origin (Vite proxy routes /api → backend).
 * Override via the UI's backend URL field for direct connections.
 */
export const DEFAULT_BACKEND_URL = '';

// ── Internal helper ───────────────────────────────────────────────

async function apiFetch<T>(
  baseUrl: string,
  path: string,
  init?: RequestInit
): Promise<T> {
  // Empty baseUrl → relative path, routed through Vite proxy (dev) or nginx (prod)
  const prefix = baseUrl ? baseUrl.replace(/\/$/, '') : '';
  const url = `${prefix}/api${path}`;
  let res: Response;
  try {
    res = await fetch(url, {
      headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
      ...init,
    });
  } catch (e) {
    throw new Error(`Network error reaching ${url}: ${(e as Error).message}`);
  }
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new Error(`${res.status} ${res.statusText}${body ? ` — ${body}` : ''}`);
  }
  return res.json() as Promise<T>;
}

// ── Public API ────────────────────────────────────────────────────

/**
 * POST /api/runs
 * Start a new algorithm run. Returns run_id immediately; the run proceeds
 * asynchronously. Poll status + batches to receive results.
 */
export async function startRun(
  baseUrl: string,
  request: StartRunRequest
): Promise<StartRunResponse> {
  return apiFetch<StartRunResponse>(baseUrl, '/runs', {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

/**
 * GET /api/runs/:run_id/status
 * Poll until status === 'complete' | 'failed'.
 */
export async function getRunStatus(
  baseUrl: string,
  runId: string
): Promise<RunStatusResponse> {
  return apiFetch<RunStatusResponse>(baseUrl, `/runs/${runId}/status`);
}

/**
 * GET /api/runs/:run_id/batches?from_batch=<n>
 * Returns all SnapshotBatch objects with batch_index >= fromBatch.
 * May return [] if no new batches are ready yet — keep polling.
 */
export async function getNewBatches(
  baseUrl: string,
  runId: string,
  fromBatch = 0
): Promise<BatchesResponse> {
  return apiFetch<BatchesResponse>(
    baseUrl,
    `/runs/${runId}/batches?from_batch=${fromBatch}`
  );
}

/**
 * Convenience: ping the backend to verify it's reachable.
 * Expects GET /api/health → { status: 'ok' } (implement this optionally).
 */
export async function pingBackend(baseUrl: string): Promise<boolean> {
  try {
    const prefix = baseUrl ? baseUrl.replace(/\/$/, '') : '';
    const res = await fetch(`${prefix}/api/health`, {
      signal: AbortSignal.timeout(3000),
    });
    return res.ok;
  } catch {
    return false;
  }
}
