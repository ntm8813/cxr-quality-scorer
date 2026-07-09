# python -m src.analysis.compile_final_report

from __future__ import annotations

import shutil
from pathlib import Path

REPORTS = Path("reports")
OUTPUT = REPORTS / "final_validation_package"

FILES = [
    "interrater_kappa.json",
    "validation_results.json",
    "list_a_reporting_fixes.json",
    "list_a_reporting_fixes.md",
    "latency_cpu.json",
    "failure_catalogue.md",
    "failure_catalogue_images_snippet.md",
]

DIRECTORIES = [
    "failure_catalogue_images",
    "figures",
]


def copy_file(name: str):
    src = REPORTS / name
    if not src.exists():
        print(f"[Missing] {src}")
        return

    shutil.copy2(src, OUTPUT / src.name)
    print(f"[Copied] {src}")


def copy_directory(name: str):
    src = REPORTS / name

    if not src.exists():
        print(f"[Missing] {src}")
        return

    dst = OUTPUT / name

    if dst.exists():
        shutil.rmtree(dst)

    shutil.copytree(src, dst)

    print(f"[Copied] {src}/")


def main():

    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)

    OUTPUT.mkdir(parents=True)

    print("Copying report files...\n")

    for file in FILES:
        copy_file(file)

    print()

    for directory in DIRECTORIES:
        copy_directory(directory)

    print()
    print("=" * 60)
    print("Validation package created successfully.")
    print(f"Location: {OUTPUT.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()