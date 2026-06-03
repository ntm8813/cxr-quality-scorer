import json
from pathlib import Path
import numpy as np

JSON_PATH = Path("evaluation_results.json")
OUT_PATH = Path("reports/runtime_report.txt")


def load_runtimes(data):
    runs = data.get("runs", [])
    runtimes = []

    for r in runs:
        if not isinstance(r, dict):
            continue

        # primary key used in your dataset
        rt = r.get("runtime", None)

        # backward compatibility (older versions)
        if rt is None:
            rt = r.get("runtime_sec", None)

        if isinstance(rt, (int, float)):
            runtimes.append(float(rt))

    return runtimes


def generate_runtime_report():

    if not JSON_PATH.exists():
        raise FileNotFoundError("evaluation_results.json not found")

    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))

    runtimes = load_runtimes(data)

    if len(runtimes) == 0:
        raise ValueError("No runtime values found in runs[]")

    avg = float(np.mean(runtimes))
    p95 = float(np.percentile(runtimes, 95))
    mn = float(np.min(runtimes))
    mx = float(np.max(runtimes))

    report = (
        "CXR PIPELINE RUNTIME REPORT\n\n"
        f"Samples evaluated : {len(runtimes)}\n"
        f"Average runtime   : {avg:.4f} sec\n"
        f"P95 runtime       : {p95:.4f} sec\n"
        f"Min runtime       : {mn:.4f} sec\n"
        f"Max runtime       : {mx:.4f} sec\n\n"
        f"Requirement (<2s) : {'PASS' if avg < 2.0 else 'FAIL'}"
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    OUT_PATH.write_text(report, encoding="utf-8")

    print(report)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    generate_runtime_report()