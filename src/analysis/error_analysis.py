# src/analysis/error_analysis.py
# python -m src.analysis.error_analysis
"""
Day 24 — Error analysis and failure catalogue.
Identifies all model-reviewer disagreements, categorises them
into failure modes, and writes reports/failure_catalogue.md

Re-run after the RotationScorer v2 fix to confirm the rotation failure
mode documented in reports/WEEK4_STATUS_REPORT.md has been resolved.
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


def check_estimation_method_skew(predictions_csv: Path) -> dict:
    """
    Diagnostic added after the Week 4 rotation bug.

    Checks data/predictions/model_v1.csv for a rotation_estimation_method
    column (if the pipeline logs raw_metrics into predictions) and flags
    if one method dominates >95% of studies — a signal that the primary
    estimator is silently falling back almost always, which is exactly
    what happened with the gradient-PCA rotation bug.

    If the column isn't present in predictions (it may only live in
    full AxisResult.raw_metrics, not the flattened CSV), this returns
    a note rather than failing.
    """
    df = pd.read_csv(predictions_csv)
    method_col = "rotation_estimation_method"

    if method_col not in df.columns:
        return {
            "checked": False,
            "note": (
                f"'{method_col}' not found in {predictions_csv}. "
                "This check requires the pipeline to flatten "
                "raw_metrics['estimation_method'] into the predictions CSV. "
                "Skipping skew check."
            ),
        }

    counts = df[method_col].value_counts(normalize=True).to_dict()
    dominant_method = max(counts, key=counts.get)
    dominant_pct = counts[dominant_method] * 100

    return {
        "checked": True,
        "method_distribution_pct": {k: round(v * 100, 1) for k, v in counts.items()},
        "dominant_method": dominant_method,
        "dominant_pct": round(dominant_pct, 1),
        "warning": (
            f"WARNING: '{dominant_method}' used in {dominant_pct:.1f}% of studies. "
            "A single estimation method dominating almost all studies can mask "
            "a silent fallback failure (see Week 4 rotation bug)."
            if dominant_pct > 95.0
            else None
        ),
    }


def main() -> None:
    preds    = pd.read_csv(PREDICTIONS)
    consensus= pd.read_csv(CONSENSUS)
    merged   = preds.merge(consensus, on="study_uid", how="inner")

    print(f"Analysing {len(merged)} studies...")

    # ── New: estimation-method skew check ──────────────────────────────
    skew_check = check_estimation_method_skew(PREDICTIONS)
    if skew_check.get("checked"):
        print(f"\nRotation estimation method distribution: "
              f"{skew_check['method_distribution_pct']}")
        if skew_check.get("warning"):
            print(f"  {skew_check['warning']}")
    else:
        print(f"\n{skew_check['note']}")

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
    print(f"\nTotal disagreements: {len(all_disagreements)}")

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
    _write_catalogue(all_disagreements, overall_disag, failure_counts, merged, skew_check)
    print(f"Saved → {OUTPUT_DIR / 'failure_catalogue.md'}")


def _write_catalogue(
    axis_disag: list,
    overall_disag: list,
    failure_counts: dict,
    merged: pd.DataFrame,
    skew_check: dict,
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
    ]

    # ── New: skew check section ─────────────────────────────────────
    lines += ["## Rotation Estimation Method Distribution", ""]
    if skew_check.get("checked"):
        for method, pct in skew_check["method_distribution_pct"].items():
            lines.append(f"- `{method}`: {pct}%")
        if skew_check.get("warning"):
            lines.append("")
            lines.append(f"⚠️ **{skew_check['warning']}**")
    else:
        lines.append(f"_{skew_check.get('note', 'Not checked.')}_")
    lines.append("")

    lines += ["## Failure Mode Definitions", ""]
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
        "| rotation  | 40 | TBD | Re-evaluate after RotationScorer v2 (mask-anchored) |",
        "",
        "---",
        "_Generated automatically by src/analysis/error_analysis.py_",
    ]

    out = OUTPUT_DIR / "failure_catalogue.md"
    out.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()