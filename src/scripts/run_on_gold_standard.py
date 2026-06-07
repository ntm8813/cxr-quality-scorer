# scripts/run_on_gold_standard.py
from __future__ import annotations

import pandas as pd
from pathlib import Path
from src.pipeline import run_pipeline

RATINGS_FILE = Path("data/ratings/reviewer_1.csv")
IMAGE_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/predictions")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not RATINGS_FILE.exists():
    raise FileNotFoundError(
        f"Reviewer 1 ratings not found at {RATINGS_FILE}. "
        "Complete rating with the rating tool first."
    )

ratings = pd.read_csv(RATINGS_FILE)
study_ids = ratings["study_uid"].tolist()

print(f"Running pipeline on {len(study_ids)} rated studies...")

records = []
failed = []

for i, uid in enumerate(study_ids):

    # deterministic search order
    candidates = (
        list(IMAGE_DIR.rglob(f"{uid}.png")) +
        list(IMAGE_DIR.rglob(f"{uid}.jpg")) +
        list(IMAGE_DIR.rglob(f"{uid}.dcm"))
    )

    if not candidates:
        print(f"[SKIP] No file found for uid: {uid}")
        continue

    path = candidates[0]

    try:
        # IMPORTANT FIX: removed invalid explain=True
        result = run_pipeline(str(path))

        row = {
            "study_uid": uid,
            "composite_score": result.composite_score,
            "overall_flag": result.overall_flag,
        }

        for ar in result.axis_results:
            ax = ar.axis if isinstance(ar.axis, str) else ar.axis.value
            row[f"{ax}_score"] = round(ar.score, 4)
            row[f"{ax}_flag"] = ar.flag if isinstance(ar.flag, str) else ar.flag.value

        records.append(row)

    except Exception as e:
        print(f"[ERROR] {uid}: {e}")
        failed.append(uid)

    if (i + 1) % 50 == 0:
        print(f"Progress: {i + 1}/{len(study_ids)}")

df = pd.DataFrame(records)

out_path = OUTPUT_DIR / "model_v1.csv"
df.to_csv(out_path, index=False)

print("\n" + "=" * 50)
print(f"Saved {len(df)} predictions → {out_path}")
print(f"Failed: {len(failed)}")

if len(df) > 0:
    print("\nFlag distribution:")
    print(df["overall_flag"].value_counts().to_string())

    print("\nComposite score stats:")
    print(df["composite_score"].describe().to_string())