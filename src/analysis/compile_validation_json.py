# python -m src.analysis.compile_validation_json

from pathlib import Path
import json

REPORTS = Path("reports")

FILES = [
    "interrater_kappa.json",
    "validation_results.json",
    "list_a_reporting_fixes.json",
    "latency_cpu.json",
]

OUTPUT = REPORTS / "validation_package.json"


def main():

    package = {}

    for filename in FILES:

        path = REPORTS / filename

        if not path.exists():
            print(f"Skipping {filename} (missing)")
            continue

        with open(path, "r", encoding="utf-8") as f:
            package[path.stem] = json.load(f)

        print(f"Loaded {filename}")

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(package, f, indent=2)

    print()
    print(f"Saved → {OUTPUT}")


if __name__ == "__main__":
    main()