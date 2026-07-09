# src/analysis/build_failure_catalogue_images.py
# python -m src.analysis.build_failure_catalogue_images
"""
List A item: Put real example images in the failure catalogue — one per
defect class. So it is obvious what each axis is actually targeting.

reports/failure_catalogue.md already exists with failure-mode tables
(FN_REPEAT, FP_REPEAT, FN_BORDER, FP_BORDER, etc. per axis) generated
from reports/disagreements.csv — but it's tables only, no images. This
script does NOT regenerate that catalogue's tables; it reads
disagreements.csv, picks one representative real example per
(axis, failure_mode) combination — preferring the highest-severity case
so the example is illustrative rather than a borderline edge case — and
copies the source image alongside a small markdown snippet you can
splice into failure_catalogue.md.

Requires: reports/disagreements.csv (already exists in the repo) and
the original image files reachable on disk (path resolved via
data/predictions/model_v1.csv's source path column, or pass
--image-dir to point at wherever the raw files actually live).
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Optional

import pandas as pd


DISAGREEMENTS_PATH = Path("reports/disagreements.csv")
PREDICTIONS_PATH = Path("data/predictions/model_v1.csv")
OUTPUT_DIR = Path("reports/failure_catalogue_images")
SNIPPET_PATH = Path("reports/failure_catalogue_images_snippet.md")
DEFAULT_IMAGE_DIR = Path(
    r"C:\Users\nirma\Documents\cxr-quality-scorer\data\raw\nih_subset"
)


def _find_source_path(study_uid: str, predictions: pd.DataFrame, image_dir: Optional[Path]) -> Optional[Path]:
    """
    Resolve a study_uid back to an actual image file on disk.

    Tries, in order:
      1. A 'source_path' or 'file_path' or 'path' column in model_v1.csv,
         if one of those exists and has a value for this study_uid.
      2. image_dir / f"{study_uid}.png" (and .dcm, .jpg) if --image-dir given.
    """
    candidate_cols = ["source_path", "file_path", "path"]
    row = predictions[predictions["study_uid"] == study_uid]
    if not row.empty:
        for col in candidate_cols:
            if col in predictions.columns:
                val = row.iloc[0].get(col)
                if isinstance(val, str) and val.strip():
                    p = Path(val)
                    if p.exists():
                        return p

    if image_dir is not None:
        for ext in (".png", ".jpg", ".jpeg", ".dcm"):
            candidate = image_dir / f"{study_uid}{ext}"
            if candidate.exists():
                return candidate

    return None


def build_catalogue(image_dir: Optional[Path]) -> None:
    if not DISAGREEMENTS_PATH.exists():
        raise FileNotFoundError(
            f"Missing: {DISAGREEMENTS_PATH}. This should already exist from "
            "the original failure-mode analysis — see "
            "src/analysis/error_analysis.py."
        )
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Missing: {PREDICTIONS_PATH}. Run "
            "src/scripts/run_on_gold_standard.py first."
        )

    disagreements = pd.read_csv(DISAGREEMENTS_PATH)
    predictions = pd.read_csv(PREDICTIONS_PATH)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # One representative example per (axis, failure_mode), preferring
    # highest severity — that's the clearest illustration of the defect.
    representative = (
        disagreements
        .sort_values("severity", ascending=False)
        .drop_duplicates(subset=["axis", "failure_mode"], keep="first")
    )

    snippet_lines = ["## Real Example Images Per Defect Class\n"]
    copied = 0
    missing = []

    for _, row in representative.iterrows():
        study_uid = row["study_uid"]
        axis = row["axis"]
        mode = row["failure_mode"]

        src = _find_source_path(study_uid, predictions, image_dir)
        if src is None:
            missing.append(f"{axis}/{mode}: study_uid={study_uid} — source file not found")
            continue

        dest_name = f"{axis}_{mode}_{study_uid}{src.suffix}"
        dest = OUTPUT_DIR / dest_name
        shutil.copy2(src, dest)
        copied += 1

        snippet_lines.append(f"### {axis} — {mode}\n")
        snippet_lines.append(f"Study `{study_uid}` — model said `{row['model_flag']}`, "
                              f"reviewer consensus `{row['consensus']}`, severity `{row['severity']}`.\n")
        snippet_lines.append(f"![{axis} {mode} example]({OUTPUT_DIR}/{dest_name})\n")

    SNIPPET_PATH.write_text("\n".join(snippet_lines))

    print(f"Copied {copied} representative example images → {OUTPUT_DIR}")
    print(f"Markdown snippet → {SNIPPET_PATH}")
    print("Splice this snippet into reports/failure_catalogue.md under each axis's table.")

    if missing:
        print(f"\n{len(missing)} examples could not be resolved to a source file:")
        for m in missing:
            print(f"  - {m}")
        print(
            "\nIf model_v1.csv doesn't carry a source path column, pass "
            "--image-dir pointing at the directory holding the raw "
            "validation images (filenames matching <study_uid>.<ext>)."
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image-dir",
        type=str,
        default=str(DEFAULT_IMAGE_DIR),
        help="Directory containing the raw source images."
    )
    args = parser.parse_args()
    image_dir = Path(args.image_dir)
    build_catalogue(image_dir)


if __name__ == "__main__":
    main()