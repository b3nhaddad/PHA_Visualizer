// Re-export all types from the canonical API contract.
// Import from here OR directly from src/api/types — they are the same types.
export type {
  PureSpinSpec,
  MixedSpinSpec,
  ModelSpec,
  RunPredictions,
  ProofTag,
  ProofAnnotation,
  Snapshot,
  SnapshotBatch,
  BatchesResponse,
  RunData,
} from '../api/types';
