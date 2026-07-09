# src/analysis/latency_benchmark.py
# python -m src.analysis.latency_benchmark --n 30
# python -m src.analysis.latency_benchmark --n 30 --device cuda
"""
List A item: Benchmark latency. Seconds per study, on CPU and on one GPU.
One number each — the commercial decision depends on it.

Usage:
    python -m src.analysis.latency_benchmark --input-dir data/raw/sample_dicoms --n 30
    python -m src.analysis.latency_benchmark --input-dir data/raw/sample_dicoms --n 30 --device cuda

Notes:
- Runs run_pipeline() end-to-end (full ingest + all scorers + fusion),
  since that's what actually determines real-world per-study latency,
  not just model forward-pass time.
- Reports both wall-clock mean/median/p95 and a breakdown that flags
  whether rejected (fail-safe) studies were excluded from timing, since
  a RejectedResult returns early and would understate true scoring time
  if mixed in with valid studies.
- GPU timing requires the relevant scorers' models to actually support
  device placement (see src/ml/model_registry.py — load_*(device=...)).
  This script does not modify model loading; it only sets the CUDA
  device env var / passes through to the pipeline, so confirm your
  ModelRegistry honours device selection before trusting GPU numbers.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import mean, median
from typing import List, Dict, Any

import numpy as np


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return float("nan")
    arr = np.array(sorted(values))
    idx = (pct / 100.0) * (len(arr) - 1)
    lo, hi = int(np.floor(idx)), int(np.ceil(idx))
    if lo == hi:
        return float(arr[lo])
    frac = idx - lo
    return float(arr[lo] * (1 - frac) + arr[hi] * frac)


def benchmark(input_dir: Path, n: int, config_path: str) -> Dict[str, Any]:
    from src.pipeline import run_pipeline
    from schemas.rejected_result import RejectedResult

    candidates = sorted(
        [p for p in input_dir.iterdir() if p.suffix.lower() in (".dcm", ".png", ".jpg", ".jpeg")]
    )
    if not candidates:
        raise FileNotFoundError(f"No .dcm/.png/.jpg files found in {input_dir}")

    paths = candidates[:n] if n else candidates
    print(f"Benchmarking {len(paths)} studies from {input_dir}")

    timings_valid: List[float] = []
    timings_rejected: List[float] = []
    errors: List[Dict[str, str]] = []

    # Warm-up run — excluded from timing. First call pays model-loading
    # cost (ModelRegistry loads weights from disk); including it would
    # make latency numbers reflect cold-start, not steady-state per-study cost.
    print("Warm-up run (excluded from timing)...")
    try:
        run_pipeline(str(paths[0]), config_path=config_path)
    except Exception as e:
        print(f"  Warm-up run raised: {e} (continuing — this may be expected for a bad sample file)")

    for i, path in enumerate(paths):
        t0 = time.perf_counter()
        try:
            result = run_pipeline(str(path), config_path=config_path)
            elapsed = time.perf_counter() - t0

            if isinstance(result, RejectedResult):
                timings_rejected.append(elapsed)
            else:
                timings_valid.append(elapsed)

        except Exception as e:
            elapsed = time.perf_counter() - t0
            errors.append({"file": str(path), "error": str(e), "elapsed_sec": round(elapsed, 4)})

        print(f"  [{i+1}/{len(paths)}] {path.name}: {elapsed:.3f}s")

    def _stats(timings: List[float]) -> Dict[str, Any]:
        if not timings:
            return {"n": 0, "mean_sec": None, "median_sec": None, "p95_sec": None, "min_sec": None, "max_sec": None}
        return {
            "n": len(timings),
            "mean_sec": round(mean(timings), 4),
            "median_sec": round(median(timings), 4),
            "p95_sec": round(_percentile(timings, 95), 4),
            "min_sec": round(min(timings), 4),
            "max_sec": round(max(timings), 4),
        }

    return {
        "n_total_attempted": len(paths),
        "n_valid_scored": len(timings_valid),
        "n_rejected_by_failsafe": len(timings_rejected),
        "n_errors": len(errors),
        "valid_scoring_latency": _stats(timings_valid),
        "rejected_failsafe_latency": _stats(timings_rejected),
        "errors": errors,
        "note": (
            "valid_scoring_latency is the number that matters for the "
            "commercial decision — it reflects full pipeline cost "
            "(ingest + all 7 scorers + fusion + explanation) for studies "
            "that were actually scored. rejected_failsafe_latency is "
            "reported separately since the fail-safe gate returns early "
            "and would understate true per-study cost if averaged "
            "together with valid_scoring_latency."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Latency benchmark for List A item.")
    parser.add_argument("--input-dir", type=str, default="data/raw/sample_dicoms",
                         help="Directory of .dcm/.png/.jpg files to benchmark against.")
    parser.add_argument("--n", type=int, default=30, help="Number of studies to time.")
    parser.add_argument("--config", type=str, default="configs/v1.yaml")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                         help="Label for this run's output filename only — confirm "
                              "src/ml/model_registry.py actually places models on "
                              "this device before trusting GPU numbers.")
    parser.add_argument("--out", type=str, default=None,
                         help="Output JSON path. Default: reports/latency_<device>.json")
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else Path(f"reports/latency_{args.device}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = benchmark(Path(args.input_dir), args.n, args.config)
    results["device_label"] = args.device

    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {out_path}")

    vs = results["valid_scoring_latency"]
    if vs["mean_sec"] is not None:
        print(
            f"\n*** {args.device.upper()}: mean {vs['mean_sec']}s/study, "
            f"median {vs['median_sec']}s, p95 {vs['p95_sec']}s "
            f"(n={vs['n']}) ***"
        )
    else:
        print(f"\n*** {args.device.upper()}: no successfully scored studies — check errors in {out_path} ***")


if __name__ == "__main__":
    main()