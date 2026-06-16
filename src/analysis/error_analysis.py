# src/analysis/error_analysis.py
# python -m src.analysis.error_analysis
"""
Day 24 — Error analysis and failure catalogue.
Identifies all model-reviewer disagreements, categorises them
into failure modes, and writes reports/failure_catalogue.md
"""
from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

PREDICTIONS = Path("data/predictions/model_v1.csv")
CONSENSUS   = Path("data/gold_standard_consensus.csv")
OUTPUT_DIR  = Path("reports")

AXES = ["sharpness", "exposure", "rotation", "coverage",
        "inspiration", "artifact", "metadata"]

FLAG_TO_INT = {"acceptable": 1, "borderline": 2, "repeat": 3}

# Failure mode taxonomy
FAILURE_MODES = {
    "FN_REPEAT": "False negative — model missed a repeat-quality image",
    "FP_REPEAT": "False positive — model incorrectly flagged acceptable image as repeat",
    "FN_BORDER": "False negative — model missed a borderline image (called it acceptable)",
    "FP_BORDER": "False positive — model called borderline image acceptable",
    "OVER_PENALISE": "Model more strict than reviewer consensus",
    "UNDER_PENALISE": "Model more lenient than reviewer consensus",
}


def classify_disagreement(model_int: int, cons_int: int) -> str:
    diff = model_int - cons_int
    if cons_int == 3 and model_int < 3:
        return "FN_REPEAT"
    if model_int == 3 and cons_int < 3:
        return "FP_REPEAT"
    if cons_int == 2 and model_int == 1:
        return "FN_BORDER"
    if model_int == 2 and cons_int == 1:
        return "FP_BORDER"
    if diff > 0:
        return "OVER_PENALISE"
    if diff < 0:
        return "UNDER_PENALISE"
    return "AGREE"


def main() -> None:
    preds    = pd.read_csv(PREDICTIONS)
    consensus= pd.read_csv(CONSENSUS)
    merged   = preds.merge(consensus, on="study_uid", how="inner")

    print(f"Analysing {len(merged)} studies...")

    # Per-axis disagreement catalogue
    all_disagreements = []
    failure_counts    = defaultdict(lambda: defaultdict(int))

    for _, row in merged.iterrows():
        uid = row["study_uid"]
        for axis in AXES:
            model_flag = row.get(f"{axis}_flag")
            cons_int   = row.get(axis)

            if pd.isna(model_flag) or pd.isna(cons_int):
                continue

            model_int = FLAG_TO_INT.get(str(model_flag))
            if model_int is None:
                continue

            cons_int = int(cons_int)
            mode = classify_disagreement(model_int, cons_int)

            if mode != "AGREE":
                all_disagreements.append({
                    "study_uid"   : uid,
                    "axis"        : axis,
                    "model_flag"  : model_flag,
                    "model_int"   : model_int,
                    "consensus"   : cons_int,
                    "failure_mode": mode,
                    "severity"    : abs(model_int - cons_int),
                })
                failure_counts[axis][mode] += 1

    disagree_df = pd.DataFrame(all_disagreements)
    disagree_df.to_csv(OUTPUT_DIR / "disagreements.csv", index=False)
    print(f"Total disagreements: {len(all_disagreements)}")

    # Overall flag disagreements
    overall_disag = []
    if "overall_flag" in merged.columns and "global_rating" in merged.columns:
        for _, row in merged.iterrows():
            model_int = FLAG_TO_INT.get(str(row["overall_flag"]))
            cons_int  = int(row["global_rating"]) if pd.notna(row["global_rating"]) else None
            if model_int is None or cons_int is None:
                continue
            mode = classify_disagreement(model_int, cons_int)
            if mode != "AGREE":
                overall_disag.append({
                    "study_uid"   : row["study_uid"],
                    "model_flag"  : row["overall_flag"],
                    "consensus"   : cons_int,
                    "failure_mode": mode,
                })

    # Write failure catalogue markdown
    _write_catalogue(all_disagreements, overall_disag, failure_counts, merged)
    print(f"Saved → {OUTPUT_DIR / 'failure_catalogue.md'}")


def _write_catalogue(
    axis_disag: list,
    overall_disag: list,
    failure_counts: dict,
    merged: pd.DataFrame,
) -> None:
    n_total     = len(merged)
    n_disag     = len(axis_disag)
    n_ov_disag  = len(overall_disag)

    lines = [
        "# Failure Mode Catalogue — MTV-INT-RAD-003",
        "",
        "## Summary",
        f"- Studies analysed: **{n_total}**",
        f"- Per-axis disagreements: **{n_disag}** "
        f"({100*n_disag/(n_total*7):.1f}% of all axis evaluations)",
        f"- Overall flag disagreements: **{n_ov_disag}** "
        f"({100*n_ov_disag/max(n_total,1):.1f}% of studies)",
        "",
        "## Failure Mode Definitions",
        "",
    ]
    for code, desc in FAILURE_MODES.items():
        lines.append(f"- **{code}**: {desc}")
    lines.append("")

    # Per-axis breakdown
    lines += ["## Per-Axis Failure Counts", ""]
    lines += ["| Axis | FN_REPEAT | FP_REPEAT | FN_BORDER | FP_BORDER | OVER | UNDER |",
              "|------|-----------|-----------|-----------|-----------|------|-------|"]
    for axis in ["sharpness", "exposure", "rotation", "coverage",
                 "inspiration", "artifact", "metadata"]:
        counts = failure_counts.get(axis, {})
        lines.append(
            f"| {axis} "
            f"| {counts.get('FN_REPEAT',0)} "
            f"| {counts.get('FP_REPEAT',0)} "
            f"| {counts.get('FN_BORDER',0)} "
            f"| {counts.get('FP_BORDER',0)} "
            f"| {counts.get('OVER_PENALISE',0)} "
            f"| {counts.get('UNDER_PENALISE',0)} |"
        )
    lines.append("")

    # Top 20 worst disagreements
    disag_df = pd.DataFrame(axis_disag)
    if len(disag_df) > 0:
        worst = disag_df.sort_values("severity", ascending=False).head(20)
        lines += [
            "## Top 20 Worst Disagreements (by severity)",
            "",
            "| Study UID | Axis | Model | Consensus | Mode |",
            "|-----------|------|-------|-----------|------|",
        ]
        for _, row in worst.iterrows():
            uid_short = str(row["study_uid"])[:20]
            lines.append(
                f"| {uid_short} | {row['axis']} "
                f"| {row['model_flag']} | {row['consensus']} "
                f"| {row['failure_mode']} |"
            )
        lines.append("")

    # Categorised failure interpretation
    lines += [
        "## Interpretation",
        "",
        "### Critical failures (FN_REPEAT)",
        "Images the model called acceptable or borderline that reviewers "
        "flagged as repeat. These are the most clinically dangerous "
        "disagreements — a poor-quality image passed to a clinician.",
        "",
        "### False alarms (FP_REPEAT)",
        "Images the model flagged as repeat that reviewers considered "
        "acceptable. These cause unnecessary repeat exposures.",
        "",
        "### Borderline misses (FN_BORDER)",
        "Borderline images scored as acceptable by the model. Lower "
        "clinical risk but indicates the model's thresholds are too lenient.",
        "",
        "### Systematic bias",
        "Compare OVER_PENALISE vs UNDER_PENALISE totals per axis. "
        "A consistent direction indicates a threshold calibration issue "
        "rather than a model accuracy issue.",
        "",
        "## Recommended Threshold Adjustments",
        "",
        "_To be filled in after Day 24 review session._",
        "",
        "| Axis | Current repeat_max | Suggested adjustment | Rationale |",
        "|------|--------------------|----------------------|-----------|",
        "| sharpness | 40 | TBD | TBD |",
        "| exposure  | 40 | TBD | TBD |",
        "| rotation  | 40 | TBD | TBD |",
        "",
        "---",
        "_Generated automatically by src/analysis/error_analysis.py_",
    ]

    out = OUTPUT_DIR / "failure_catalogue.md"
    out.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()