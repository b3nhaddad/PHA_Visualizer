/**
 * Loads a static snapshot file produced by Stage A (the Python algorithm).
 * Supports both v0.1.0 (legacy flat format) and v0.2.0 (batch format).
 *
 * When the teammate's algorithm is ready, they can drop the JSON file into
 * public/ and use loadStaticFile('/their_run.json') to view it.
 */
import type { RunData, SnapshotBatch } from '../api/types';

export class SchemaVersionError extends Error {}
export class LoadError extends Error {}

/** Load a complete run from a static JSON file (Stage A output). */
export async function loadStaticFile(url: string): Promise<RunData> {
  let raw: unknown;
  try {
    const res = await fetch(url);
    if (!res.ok) throw new LoadError(`HTTP ${res.status} fetching ${url}`);
    raw = await res.json();
  } catch (e) {
    if (e instanceof LoadError) throw e;
    throw new LoadError(`Failed to fetch ${url}: ${(e as Error).message}`);
  }

  // Duck-type: v0.2.0 has batch_index, v0.1.0 has snapshots at top level
  const data = raw as Record<string, unknown>;
  const version = (data.schema_version as string) ?? '';

  if (version.startsWith('0.2.')) {
    // Could be a single batch or an array of batches wrapped in an object
    if ('batch_index' in data) {
      // Single batch file
      const batch = data as unknown as SnapshotBatch;
      return batchToRunData([batch]);
    }
    if (Array.isArray(data.batches)) {
      return batchToRunData(data.batches as SnapshotBatch[]);
    }
    throw new LoadError('v0.2.0 file must have batch_index or batches[]');
  }

  if (version.startsWith('0.1.')) {
    return normalise_v1(data);
  }

  if (!version) {
    throw new SchemaVersionError('Missing schema_version field');
  }
  throw new SchemaVersionError(`Unsupported schema version "${version}"`);
}

function batchToRunData(batches: SnapshotBatch[]): RunData {
  const sorted = [...batches].sort((a, b) => a.batch_index - b.batch_index);
  const last = sorted[sorted.length - 1];
  return {
    run_id: last.run_id,
    model: last.model,
    predictions: last.predictions,
    snapshots: sorted.flatMap(b => b.snapshots).sort((a, b) => a.index - b.index),
    totalSteps: last.total_steps,
    isComplete: last.is_final,
  };
}

// Normalise the old v0.1.0 pure-p-spin flat format into a RunData
function normalise_v1(data: Record<string, unknown>): RunData {
  const model = data.model as Record<string, unknown>;
  const N = model.N as number;
  const p = model.p as number;
  const preds = data.predictions as Record<string, unknown>;

  if (typeof N !== 'number' || typeof p !== 'number') {
    throw new LoadError('v0.1.0: model.N / model.p missing or non-numeric');
  }

  return {
    run_id: (data.run_id as string) ?? 'unknown',
    model: { type: 'pure_p_spin', p, N },
    predictions: {
      E_infinity: preds.E_infinity as number,
      E_infinity_formula: (preds.E_infinity_formula as string) ?? '',
      spectral_edge_formula: (preds.spectral_edge_formula as string) ?? '',
    },
    snapshots: (data.snapshots as unknown[]) as RunData['snapshots'],
    totalSteps: (data.snapshots as unknown[]).length,
    isComplete: true,
  };
}
