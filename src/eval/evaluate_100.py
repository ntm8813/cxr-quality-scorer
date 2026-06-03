import time
import json
from pathlib import Path
import numpy as np
from src.pipeline import run_pipeline


class RealCXREvaluator:

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Dataset not found: {data_dir}")

        self.images = sorted([
            str(p) for p in self.data_dir.rglob("*")
            if p.suffix.lower() in [".png", ".jpg", ".jpeg"]
        ])

        if len(self.images) < 100:
            raise ValueError(f"Need at least 100 images, found {len(self.images)}")

        print(f"[DATASET] Found {len(self.images)} images")

    def get_100_unseen(self, seed=42):
        np.random.seed(seed)
        selected = np.random.choice(self.images, 100, replace=False)
        return selected.tolist()


def evaluate_real(data_dir: str, limit=100):

    evaluator = RealCXREvaluator(data_dir)
    study_paths = evaluator.get_100_unseen()

    results = []
    runtimes = []

    for i, path in enumerate(study_paths):

        start = time.perf_counter()

        try:
            study_result = run_pipeline(path)
        except Exception as e:
            print(f"[{i}] ERROR:", path, e)
            continue

        end = time.perf_counter()
        runtime = end - start

        runtimes.append(runtime)

        # ================================
        # FIXED: CLEAN, CONSISTENT SCHEMA
        # ================================
        results.append({
            "study_uid": study_result.study_uid,
            "composite_score": study_result.composite_score,
            "overall_flag": study_result.overall_flag,

            "runtime": runtime,

            "axis_results": [
                {
                    "axis": str(a.axis),
                    "score": float(a.score)
                }
                for a in study_result.axis_results
            ],

            "path": path
        })

        print(f"[{i+1}/{limit}] runtime={runtime:.4f}s score={study_result.composite_score}")

    if len(results) == 0:
        raise RuntimeError("NO VALID RUNS — pipeline failed completely")

    avg = float(np.mean(runtimes))
    p95 = float(np.percentile(runtimes, 95))

    print("\n=== FINAL RUNTIME REPORT ===")
    print(f"Average runtime : {avg:.4f} sec")
    print(f"P95 runtime     : {p95:.4f} sec")

    output = {
        "dataset_path": str(data_dir),
        "num_samples": len(results),
        "avg_runtime_sec": avg,
        "p95_runtime_sec": p95,
        "runs": results
    }

    output_path = Path("evaluation_results.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Saved: {output_path}")


if __name__ == "__main__":

    DATA_DIR = r"C:\Users\nirma\Documents\cxr-quality-scorer\data\raw\nih_subset"

    evaluate_real(DATA_DIR, 100)