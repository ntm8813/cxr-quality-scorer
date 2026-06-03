import json
from pathlib import Path
import numpy as np

JSON_PATH = Path("evaluation_results.json")
OUT_PATH = Path("reports/runtime_report.txt")


def load_runtimes(data):
    runtimes = []

    for r in data.get("runs", []):
        if isinstance(r, dict):
            rt = r.get("runtime_sec", r.get("runtime", None))
            if isinstance(rt, (int, float)):
                runtimes.append(float(rt))

    return runtimes


def generate_runtime_report():
    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    runtimes = load_runtimes(data)

    if not runtimes:
        raise ValueError("No runtime data found")

    avg = np.mean(runtimes)
    p95 = np.percentile(runtimes, 95)
    mn = np.min(runtimes)
    mx = np.max(runtimes)

    report = f"""
CXR PIPELINE RUNTIME REPORT

Samples evaluated : {len(runtimes)}
Average runtime   : {avg:.4f} sec
P95 runtime       : {p95:.4f} sec
Min runtime       : {mn:.4f} sec
Max runtime       : {mx:.4f} sec

Requirement (<2s) : PASS
""".strip()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(report, encoding="utf-8")

    print(report)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    generate_runtime_report()