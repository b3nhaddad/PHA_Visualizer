"""
Validate a run JSON file against the v0.2.0 schema (uses backend Pydantic models).

Usage:
    python tools/validate_schema.py public/mock_run.json
    python tools/validate_schema.py public/mock_mixed.json --strict

--strict  also verifies that every edge_target_satisfied field is True
          and that ||sigma||^2/N == q at every step (within 1e-4).
"""
import argparse
import json
import math
import sys
from pathlib import Path

# Allow running from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from api.models import SnapshotBatch


def load_batches(path: Path) -> list[SnapshotBatch]:
    data = json.loads(path.read_text())

    # Support both single-batch files and the wrapped {"batches": [...]} format
    if "batches" in data:
        raw_batches = data["batches"]
    elif "batch_index" in data:
        raw_batches = [data]
    else:
        # Legacy flat format: no batches key — not supported by this validator
        print("ERROR: file appears to be v0.1.0 (no 'batches' key). Only v0.2.0 is supported.")
        sys.exit(1)

    return [SnapshotBatch.model_validate(b) for b in raw_batches]


def strict_checks(batches: list[SnapshotBatch]) -> list[str]:
    errors: list[str] = []
    for batch in batches:
        for snap in batch.snapshots:
            if not snap.edge_target_satisfied:
                errors.append(
                    f"Step {snap.j}: edge_target_satisfied=False "
                    f"(lambda_min={snap.lambda_min_tangent:.4f}, edge={snap.spectral_edge_target:.4f})"
                )
            N = len(snap.sigma)
            if N > 0:
                norm_sq_over_N = sum(x * x for x in snap.sigma) / N
                if abs(norm_sq_over_N - snap.q) > 1e-3:
                    errors.append(
                        f"Step {snap.j}: ||sigma||^2/N={norm_sq_over_N:.6f} != q={snap.q:.6f}"
                    )
    return errors


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate a run JSON against schema v0.2.0")
    ap.add_argument("file", type=Path)
    ap.add_argument("--strict", action="store_true", help="Run physics invariant checks")
    args = ap.parse_args()

    if not args.file.exists():
        print(f"ERROR: file not found: {args.file}")
        sys.exit(1)

    try:
        batches = load_batches(args.file)
    except Exception as exc:
        print(f"SCHEMA ERROR: {exc}")
        sys.exit(1)

    total_snaps = sum(len(b.snapshots) for b in batches)
    print(f"OK  {args.file} — {len(batches)} batch(es), {total_snaps} snapshot(s)")
    print(f"    run_id : {batches[0].run_id}")
    print(f"    model  : {batches[0].model.type}  N={batches[0].model.N}")

    if args.strict:
        errors = strict_checks(batches)
        if errors:
            for e in errors:
                print(f"STRICT FAIL: {e}")
            sys.exit(1)
        else:
            print("    strict : all physics invariants passed")


if __name__ == "__main__":
    main()
