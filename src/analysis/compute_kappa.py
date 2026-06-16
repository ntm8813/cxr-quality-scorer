# src/analysis/compute_kappa.py
# python -m src.analysis.compute_kappa
"""
Day 22 — Inter-rater Cohen's kappa between Reviewer 1 and Reviewer 2.
Computes per-axis kappa and overall (global_rating) kappa.
Saves results to reports/interrater_kappa.json and prints a table.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

REVIEWER_1 = Path("data/ratings/reviewer_1.csv")
REVIEWER_2 = Path("data/ratings/reviewer_2.csv")
OUTPUT_DIR = Path("reports")
OUTPUT_DIR.mkdir(exist_ok=True)

AXES = ["sharpness", "exposure", "rotation", "coverage",
        "inspiration", "artifact", "metadata", "global_rating"]


def load_and_align(r1_path: Path, r2_path: Path) -> pd.DataFrame:
    r1 = pd.read_csv(r1_path)
    r2 = pd.read_csv(r2_path)

    # Merge on study_uid — keep only studies rated by BOTH reviewers
    merged = r1.merge(r2, on="study_uid", suffixes=("_r1", "_r2"))
    print(f"Reviewer 1 rated : {len(r1)} studies")
    print(f"Reviewer 2 rated : {len(r2)} studies")
    print(f"Overlap (both)   : {len(merged)} studies")
    return merged


def compute_kappa_table(merged: pd.DataFrame) -> dict:
    results = {}
    for axis in AXES:
        col_r1 = f"{axis}_r1"
        col_r2 = f"{axis}_r2"
        if col_r1 not in merged.columns or col_r2 not in merged.columns:
            print(f"  [SKIP] {axis} — columns not found")
            continue

        y1 = merged[col_r1].dropna().astype(int)
        y2 = merged[col_r2].dropna().astype(int)

        # Align index after dropna
        common = y1.index.intersection(y2.index)
        y1 = y1.loc[common]
        y2 = y2.loc[common]

        if len(y1) < 2:
            results[axis] = {"kappa": None, "n": 0, "note": "insufficient data"}
            continue

        try:
            kappa = float(cohen_kappa_score(y1, y2, weights="quadratic"))
            agreement_pct = float((y1 == y2).mean() * 100)
            results[axis] = {
                "kappa"          : round(kappa, 4),
                "n"              : int(len(y1)),
                "agreement_pct"  : round(agreement_pct, 1),
                "interpretation" : _interpret_kappa(kappa),
            }
        except Exception as e:
            results[axis] = {"kappa": None, "n": int(len(y1)), "note": str(e)}

    return results


def _interpret_kappa(k: float) -> str:
    if k < 0:       return "poor"
    if k < 0.20:    return "slight"
    if k < 0.40:    return "fair"
    if k < 0.60:    return "moderate"
    if k < 0.80:    return "substantial"
    return "almost perfect"


def print_table(results: dict) -> None:
    print(f"\n{'Axis':<18} {'κ':>8} {'Agreement':>12} {'N':>6} {'Interpretation'}")
    print("-" * 65)
    for axis, row in results.items():
        if row.get("kappa") is None:
            print(f"  {axis:<16} {'N/A':>8} {'':>12} {row.get('n',0):>6}")
        else:
            print(
                f"  {axis:<16} {row['kappa']:>8.4f} "
                f"{row['agreement_pct']:>11.1f}% "
                f"{row['n']:>6}  {row['interpretation']}"
            )


def main() -> None:
    if not REVIEWER_1.exists():
        raise FileNotFoundError(f"Missing: {REVIEWER_1}")
    if not REVIEWER_2.exists():
        raise FileNotFoundError(f"Missing: {REVIEWER_2}")

    merged  = load_and_align(REVIEWER_1, REVIEWER_2)
    results = compute_kappa_table(merged)
    print_table(results)

    output = {
        "n_studies"  : len(merged),
        "per_axis"   : results,
        "ceiling_kappa": results.get("global_rating", {}).get("kappa"),
        "note"       : (
            "Quadratic-weighted Cohen's kappa. "
            "global_rating kappa is the ceiling for model evaluation — "
            "model should not be expected to exceed human inter-rater agreement."
        ),
    }

    out_path = OUTPUT_DIR / "interrater_kappa.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved → {out_path}")

    # Also save consensus ratings for Day 23
    r1 = pd.read_csv(REVIEWER_1)
    r2 = pd.read_csv(REVIEWER_2)
    merged_full = r1.merge(r2, on="study_uid", suffixes=("_r1", "_r2"))

    consensus_rows = []
    for _, row in merged_full.iterrows():
        cons = {"study_uid": row["study_uid"]}
        for axis in AXES:
            v1 = row.get(f"{axis}_r1")
            v2 = row.get(f"{axis}_r2")
            if pd.notna(v1) and pd.notna(v2):
                # Majority vote — with two raters, take the mean rounded
                cons[axis] = int(round((float(v1) + float(v2)) / 2))
            elif pd.notna(v1):
                cons[axis] = int(v1)
            elif pd.notna(v2):
                cons[axis] = int(v2)
            else:
                cons[axis] = None
        consensus_rows.append(cons)

    consensus_df = pd.DataFrame(consensus_rows)
    cons_path = Path("data/gold_standard_consensus.csv")
    consensus_df.to_csv(cons_path, index=False)
    print(f"Saved → {cons_path}")


if __name__ == "__main__":
    main()