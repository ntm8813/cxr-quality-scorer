import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error


JSON_PATH = Path("evaluation_results.json")
OUTPUT_PATH = Path("reports/metrics_report.txt")


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

        axes = {}
        for a in r.get("axis_results", []):
            axes[str(a.get("axis"))] = float(a.get("score", 0.0))

        exposure.append(axes.get("AxisName.EXPOSURE", 0.0))
        sharpness.append(axes.get("AxisName.SHARPNESS", 0.0))
        rotation.append(axes.get("AxisName.ROTATION", 0.0))
        coverage.append(axes.get("AxisName.COVERAGE", 0.0))
        inspiration.append(axes.get("AxisName.INSPIRATION", 0.0))

    return {
        "composite": np.array(composite),
        "exposure": np.array(exposure),
        "sharpness": np.array(sharpness),
        "rotation": np.array(rotation),
        "coverage": np.array(coverage),
        "inspiration": np.array(inspiration),
    }


def compute_metrics():

    if not JSON_PATH.exists():
        raise FileNotFoundError("evaluation_results.json not found")

    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    arr = extract_arrays(data)

    lines = []
    lines.append("=== MAE REPORT (vs composite_score) ===")

    for name, values in arr.items():
        if name == "composite":
            continue

        mae = mean_absolute_error(arr["composite"], values)
        rho, _ = spearmanr(arr["composite"], values)

        line = f"{name:12s} | MAE={mae:.4f} | Spearman ρ={rho:.4f}"
        print(line)
        lines.append(line)

    lines.append("\n=== GLOBAL COMPOSITE DISTRIBUTION ===")
    lines.append(f"Mean : {float(np.mean(arr['composite']))}")
    lines.append(f"Std  : {float(np.std(arr['composite']))}")
    lines.append(f"Min  : {float(np.min(arr['composite']))}")
    lines.append(f"Max  : {float(np.max(arr['composite']))}")

    output_text = "\n".join(lines)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(output_text, encoding="utf-8")

    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    compute_metrics()