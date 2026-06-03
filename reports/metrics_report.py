import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error

JSON_PATH = Path("evaluation_results.json")
OUT_PATH = Path("reports/metrics_report.txt")


def extract_arrays(data):
    runs = data.get("runs", [])

    composite = []
    exposure = []
    sharpness = []
    rotation = []
    coverage = []
    inspiration = []

    for r in runs:
        composite.append(r.get("composite_score", 0.0))

        axes = {a["axis"].upper(): a["score"] for a in r.get("axis_results", [])}

        exposure.append(axes.get("EXPOSURE", 0.0))
        sharpness.append(axes.get("SHARPNESS", 0.0))
        rotation.append(axes.get("ROTATION", 0.0))
        coverage.append(axes.get("COVERAGE", 0.0))
        inspiration.append(axes.get("INSPIRATION", 0.0))

    return {
        "composite": np.array(composite, dtype=float),
        "exposure": np.array(exposure, dtype=float),
        "sharpness": np.array(sharpness, dtype=float),
        "rotation": np.array(rotation, dtype=float),
        "coverage": np.array(coverage, dtype=float),
        "inspiration": np.array(inspiration, dtype=float),
    }


def safe_spearman(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return spearmanr(x, y).correlation


def compute_metrics():
    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    arr = extract_arrays(data)

    lines = []
    lines.append("=== MAE REPORT (vs composite_score) ===")

    for name, values in arr.items():
        if name == "composite":
            continue

        mae = mean_absolute_error(arr["composite"], values)
        rho = safe_spearman(arr["composite"], values)

        line = f"{name:12s} | MAE={mae:.4f} | Spearman ρ={rho}"
        print(line)
        lines.append(line)

    lines.append("\n=== GLOBAL COMPOSITE DISTRIBUTION ===")
    lines.append(f"Mean : {float(np.mean(arr['composite']))}")
    lines.append(f"Std  : {float(np.std(arr['composite']))}")
    lines.append(f"Min  : {float(np.min(arr['composite']))}")
    lines.append(f"Max  : {float(np.max(arr['composite']))}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    compute_metrics()