# src/analysis/reporting_fixes.py
# python -m src.analysis.reporting_fixes
"""
List A reporting fixes (items 1-3 from the post-delivery review):

  1. Show the inter-rater kappa CEILING next to every model kappa.
     "0.22 against a human ceiling of X" is the only honest line.
  2. Add confusion matrices + precision/recall for borderline/repeat,
     per axis. Agreement-% and kappa alone hide that the model catches
     almost no bad images — show the actual catch rate on minority
     classes.
  3. Explain the overall-vs-per-axis kappa gap. Every axis is ~0 yet
     overall is 0.22 — state exactly how "overall" is computed and show
     the math, since as written it looks like a bug.

This script does NOT recompute kappa or confusion matrices from scratch —
those already exist correctly in:
  - reports/interrater_kappa.json     (from src/analysis/compute_kappa.py)
  - reports/validation_results.json   (from src/analysis/compute_validation.py)

It joins them, derives precision/recall from the existing confusion
matrices, and writes one combined, honest report.

Run AFTER compute_kappa.py and compute_validation.py (it will tell you
if either hasn't been run yet).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np

INTERRATER_PATH = Path("reports/interrater_kappa.json")
VALIDATION_PATH = Path("reports/validation_results.json")
OUTPUT_JSON = Path("reports/list_a_reporting_fixes.json")
OUTPUT_MD = Path("reports/list_a_reporting_fixes.md")

LABELS = ["acceptable", "borderline", "repeat"]  # matches confusion_matrix(labels=[1,2,3]) ordering


# -----------------------------------------------------------------------
# Item 2 — precision/recall from an existing confusion matrix
# -----------------------------------------------------------------------

def precision_recall_from_confusion_matrix(cm: List[List[int]]) -> Dict[str, Dict[str, float]]:
    """
    cm is a 3x3 confusion matrix as produced by sklearn's
    confusion_matrix(y_true=consensus, y_pred=model, labels=[1,2,3]),
    i.e. rows = reviewer consensus (ground truth), columns = model
    prediction, in order [acceptable, borderline, repeat].

    Precision for class k = TP_k / (predicted as k)   = column sum
    Recall    for class k = TP_k / (actually k)        = row sum

    Returns per-class precision/recall/f1/support, focused especially
    on borderline and repeat — the minority classes the review flagged
    as the ones actually hidden by agreement-% and overall kappa.
    """
    cm_arr = np.array(cm, dtype=float)
    result = {}

    for idx, label in enumerate(LABELS):
        tp = cm_arr[idx, idx]
        predicted_as_k = cm_arr[:, idx].sum()
        actually_k = cm_arr[idx, :].sum()

        precision = float(tp / predicted_as_k) if predicted_as_k > 0 else None
        recall = float(tp / actually_k) if actually_k > 0 else None

        if precision is not None and recall is not None and (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = None

        result[label] = {
            "precision": round(precision, 4) if precision is not None else None,
            "recall": round(recall, 4) if recall is not None else None,
            "f1": round(f1, 4) if f1 is not None else None,
            "support": int(actually_k),
            "note": (
                "support=0 — no real examples of this class in the gold-standard "
                "set, so precision/recall here are not meaningful regardless of "
                "the model's behaviour."
                if actually_k == 0 else None
            ),
        }

    return result


def minority_class_catch_rate(cm: List[List[int]]) -> Dict[str, Any]:
    """
    Item 2's core ask in plain terms: of all studies a human reviewer
    actually marked borderline or repeat (the ones that MATTER to catch),
    what fraction did the model also flag as borderline or repeat?

    This is the number agreement-% and kappa hide, because both are
    dominated by the "acceptable" majority class.
    """
    cm_arr = np.array(cm, dtype=float)
    # rows/cols 1,2 = borderline, repeat (index 1, 2)
    bad_rows = cm_arr[1:, :]  # reviewer said borderline or repeat
    total_bad = bad_rows.sum()
    caught = bad_rows[:, 1:].sum()  # model also said borderline or repeat

    if total_bad == 0:
        return {
            "catch_rate": None,
            "n_actual_bad": 0,
            "n_caught": 0,
            "note": "No borderline/repeat cases in this axis's gold-standard set.",
        }

    return {
        "catch_rate": round(float(caught / total_bad), 4),
        "n_actual_bad": int(total_bad),
        "n_caught": int(caught),
        "note": (
            f"Of {int(total_bad)} studies reviewers marked borderline/repeat, "
            f"the model also flagged {int(caught)} as borderline/repeat."
        ),
    }


# -----------------------------------------------------------------------
# Item 3 — explain the overall-vs-per-axis kappa gap with the actual math
# -----------------------------------------------------------------------

def explain_overall_vs_per_axis_gap(per_axis: Dict[str, Any], overall: Dict[str, Any]) -> Dict[str, Any]:
    """
    The review's concern: every per-axis kappa is ~0 yet overall kappa is
    0.22. As written that looks like a bug. It isn't — it's because
    "overall" is NOT an aggregate of the seven per-axis kappas. It is
    computed completely separately, from a different pair of columns:

        overall kappa = kappa(model.overall_flag, reviewer.global_rating)

    i.e. the model's single fused composite-score flag vs. the reviewer's
    single holistic global_rating — a totally different comparison from
    "does the model's sharpness flag agree with the reviewer's sharpness
    rating." The two numbers are not derived from each other and there is
    no mathematical requirement that one be an average of the other set.

    A plausible mechanism for why overall can be higher even when every
    per-axis kappa is near zero: composite_score is a weighted SUM across
    axes (src/fusion/score_fusion.py), so it can land in the right
    acceptable/borderline/repeat bucket via compensating errors across
    axes, or simply by tracking whichever single axis dominates the
    weighting, even if no individual axis is reliably correlated with its
    OWN corresponding reviewer column. This is exactly why the review
    says "model should not be expected to exceed human inter-rater
    agreement" but also why overall kappa alone is not sufficient
    evidence of axis-level competence — it can mask exactly the
    per-axis failure this report exists to surface.
    """
    overall_kappa = overall.get("kappa")
    per_axis_kappas = {
        axis: row.get("kappa")
        for axis, row in per_axis.items()
        if row.get("kappa") is not None
    }

    if per_axis_kappas:
        mean_per_axis = round(float(np.mean(list(per_axis_kappas.values()))), 4)
        max_per_axis = round(float(np.max(list(per_axis_kappas.values()))), 4)
    else:
        mean_per_axis = None
        max_per_axis = None

    return {
        "overall_kappa": overall_kappa,
        "mean_per_axis_kappa": mean_per_axis,
        "max_per_axis_kappa": max_per_axis,
        "gap": (
            round(overall_kappa - mean_per_axis, 4)
            if overall_kappa is not None and mean_per_axis is not None
            else None
        ),
        "explanation": (
            "Overall kappa is NOT an average or aggregate of the seven "
            "per-axis kappas. It is computed independently as "
            "kappa(model.overall_flag, reviewer.global_rating) — the "
            "model's fused composite-score flag against the reviewer's "
            "separate holistic global rating column. Per-axis kappa is "
            "computed as kappa(model.<axis>_flag, reviewer.<axis>) for "
            "each of the seven axes individually. These are two distinct "
            "computations on two distinct column pairs (see "
            "compute_overall_kappa() vs compute_per_axis_kappa() in "
            "src/analysis/compute_validation.py). There is no "
            "mathematical requirement that overall track the per-axis "
            "mean, and a higher overall than per-axis mean does not "
            "indicate a bug — it indicates the composite score's "
            "weighted-sum fusion (src/fusion/score_fusion.py) can land "
            "in approximately the right overall bucket via compensating "
            "errors across axes or by being dominated by whichever "
            "axis carries the most weight, even while no individual "
            "axis reliably tracks its own corresponding reviewer rating. "
            "This is precisely why per-axis kappa, not overall kappa "
            "alone, is the correct metric for axis-level model quality."
        ),
    }


# -----------------------------------------------------------------------
# Item 1 — join ceiling kappa next to every model kappa
# -----------------------------------------------------------------------

def build_ceiling_comparison(per_axis: Dict[str, Any], overall: Dict[str, Any],
                              interrater: Dict[str, Any]) -> Dict[str, Any]:
    ceiling_per_axis = interrater.get("per_axis", {})
    ceiling_overall = interrater.get("ceiling_kappa")

    comparison = {}
    for axis, row in per_axis.items():
        model_kappa = row.get("kappa")
        ceiling_row = ceiling_per_axis.get(axis, {})
        ceiling_kappa = ceiling_row.get("kappa")

        comparison[axis] = {
            "model_kappa": model_kappa,
            "human_ceiling_kappa": ceiling_kappa,
            "pct_of_ceiling_achieved": (
                round(100.0 * model_kappa / ceiling_kappa, 1)
                if model_kappa is not None and ceiling_kappa not in (None, 0)
                else None
            ),
            "honest_line": (
                f"{model_kappa:.4f} against a human ceiling of {ceiling_kappa:.4f}"
                if model_kappa is not None and ceiling_kappa is not None
                else "Not computable — missing kappa value(s)."
            ),
        }

    comparison["OVERALL"] = {
        "model_kappa": overall.get("kappa"),
        "human_ceiling_kappa": ceiling_overall,
        "pct_of_ceiling_achieved": (
            round(100.0 * overall.get("kappa") / ceiling_overall, 1)
            if overall.get("kappa") is not None and ceiling_overall not in (None, 0)
            else None
        ),
        "honest_line": (
            f"{overall.get('kappa'):.4f} against a human ceiling of {ceiling_overall:.4f}"
            if overall.get("kappa") is not None and ceiling_overall is not None
            else "Not computable — missing kappa value(s)."
        ),
    }

    return comparison


# -----------------------------------------------------------------------
# Markdown report
# -----------------------------------------------------------------------

def render_markdown(ceiling_comparison: Dict[str, Any],
                     precision_recall: Dict[str, Dict[str, Any]],
                     catch_rates: Dict[str, Any],
                     gap_explanation: Dict[str, Any]) -> str:
    lines = []
    lines.append("# List A Reporting Fixes — MTV-INT-RAD-003\n")
    lines.append(
        "This report joins existing kappa/confusion-matrix outputs from "
        "`compute_kappa.py` and `compute_validation.py`, and adds the "
        "honesty checks the post-delivery review asked for: kappa read "
        "against its human ceiling, minority-class catch rate, and an "
        "explicit explanation of the overall-vs-per-axis kappa gap.\n"
    )

    lines.append("## 1. Model kappa vs. human inter-rater ceiling\n")
    lines.append("| Axis | Model κ | Human Ceiling κ | % of Ceiling | Honest line |")
    lines.append("|---|---|---|---|---|")
    for axis, row in ceiling_comparison.items():
        mk = row["model_kappa"]
        ck = row["human_ceiling_kappa"]
        pct = row["pct_of_ceiling_achieved"]
        lines.append(
            f"| {axis} | {mk if mk is not None else 'N/A'} | "
            f"{ck if ck is not None else 'N/A'} | "
            f"{f'{pct}%' if pct is not None else 'N/A'} | {row['honest_line']} |"
        )
    lines.append("")

    lines.append("## 2. Minority-class catch rate (borderline + repeat)\n")
    lines.append(
        "Agreement-% and kappa are dominated by the acceptable majority "
        "class. This is the number that actually answers \"does the model "
        "catch bad images\":\n"
    )
    lines.append("| Axis | Catch Rate | Actual Bad Cases | Caught |")
    lines.append("|---|---|---|---|")
    for axis, row in catch_rates.items():
        cr = row.get("catch_rate")
        lines.append(
            f"| {axis} | {f'{cr*100:.1f}%' if cr is not None else 'N/A (no bad cases in set)'} | "
            f"{row.get('n_actual_bad', 0)} | {row.get('n_caught', 0)} |"
        )
    lines.append("")

    lines.append("## 3. Per-class precision / recall / F1\n")
    for axis, classes in precision_recall.items():
        lines.append(f"### {axis}\n")
        lines.append("| Class | Precision | Recall | F1 | Support |")
        lines.append("|---|---|---|---|---|")
        for cls, m in classes.items():
            note = f" ({m['note']})" if m.get("note") else ""
            lines.append(
                f"| {cls} | {m['precision']} | {m['recall']} | {m['f1']} | "
                f"{m['support']}{note} |"
            )
        lines.append("")

    lines.append("## 4. Why overall κ (0.22) doesn't match near-zero per-axis κ\n")
    lines.append(f"- Overall kappa: **{gap_explanation['overall_kappa']}**")
    lines.append(f"- Mean per-axis kappa: **{gap_explanation['mean_per_axis_kappa']}**")
    lines.append(f"- Max per-axis kappa: **{gap_explanation['max_per_axis_kappa']}**")
    lines.append(f"- Gap: **{gap_explanation['gap']}**\n")
    lines.append(gap_explanation["explanation"])
    lines.append("")

    return "\n".join(lines)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main() -> None:
    for path in [INTERRATER_PATH, VALIDATION_PATH]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing: {path}\n"
                "Run src/analysis/compute_kappa.py and "
                "src/analysis/compute_validation.py first — this script "
                "joins their outputs, it does not recompute them."
            )

    interrater = json.loads(
        INTERRATER_PATH.read_text(encoding="utf-8")
    )

    validation = json.loads(
        VALIDATION_PATH.read_text(encoding="utf-8")
    )

    per_axis = validation.get("per_axis_kappa", {})
    overall = validation.get("overall_kappa", {})

    ceiling_comparison = build_ceiling_comparison(per_axis, overall, interrater)

    precision_recall = {}
    catch_rates = {}
    for axis, row in per_axis.items():
        cm = row.get("confusion_matrix")
        if cm:
            precision_recall[axis] = precision_recall_from_confusion_matrix(cm)
            catch_rates[axis] = minority_class_catch_rate(cm)

    overall_cm = overall.get("confusion_matrix")
    if overall_cm:
        precision_recall["OVERALL"] = precision_recall_from_confusion_matrix(overall_cm)
        catch_rates["OVERALL"] = minority_class_catch_rate(overall_cm)

    gap_explanation = explain_overall_vs_per_axis_gap(per_axis, overall)

    output = {
        "ceiling_comparison": ceiling_comparison,
        "precision_recall": precision_recall,
        "minority_class_catch_rate": catch_rates,
        "overall_vs_per_axis_gap_explanation": gap_explanation,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text( json.dumps(output, indent=2), encoding="utf-8",)
    print(f"Saved → {OUTPUT_JSON}")

    md = render_markdown(
        ceiling_comparison,
        precision_recall,
        catch_rates,
        gap_explanation,
    )

    OUTPUT_MD.write_text(md, encoding="utf-8",)
    print(f"Saved → {OUTPUT_MD}")


if __name__ == "__main__":
    main()