# src/analysis/compute_validation.py
# python -m src.analysis.compute_validation
"""
Day 23 — Model validation against gold-standard reviewer consensus.
Computes:
  - Weighted Cohen's kappa (model flag vs consensus flag) per axis + overall
  - Per-axis confusion matrices
  - Spearman rho (composite score vs reviewer global rating)
  - Calibration plot
Saves to reports/validation_results.json and reports/figures/
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score, confusion_matrix

PREDICTIONS  = Path("data/predictions/model_v1.csv")
CONSENSUS    = Path("data/gold_standard_consensus.csv")
REVIEWER_1   = Path("data/ratings/reviewer_1.csv")
REVIEWER_2   = Path("data/ratings/reviewer_2.csv")
OUTPUT_DIR   = Path("reports")
FIGURES_DIR  = Path("reports/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

AXES = ["sharpness", "exposure", "rotation", "coverage",
        "inspiration", "artifact", "metadata"]

# Map model flag string → integer scale matching reviewer scale
FLAG_TO_INT = {"acceptable": 1, "borderline": 2, "repeat": 3}
# Map reviewer integer → flag string for display
INT_TO_FLAG = {1: "acceptable", 2: "borderline", 3: "repeat"}


def load_data():
    preds    = pd.read_csv(PREDICTIONS)
    consensus= pd.read_csv(CONSENSUS)
    r1       = pd.read_csv(REVIEWER_1) if REVIEWER_1.exists() else None
    r2       = pd.read_csv(REVIEWER_2) if REVIEWER_2.exists() else None
    return preds, consensus, r1, r2


def merge_predictions_and_consensus(preds: pd.DataFrame,
                                    consensus: pd.DataFrame) -> pd.DataFrame:
    merged = preds.merge(consensus, on="study_uid", how="inner")
    print(f"Predictions : {len(preds)} studies")
    print(f"Consensus   : {len(consensus)} studies")
    print(f"Overlap     : {len(merged)} studies")
    return merged


def compute_per_axis_kappa(merged: pd.DataFrame) -> dict:
    results = {}
    for axis in AXES:
        model_col = f"{axis}_flag"
        cons_col  = axis  # consensus CSV uses plain axis name

        if model_col not in merged.columns:
            results[axis] = {"kappa": None, "note": "model column missing"}
            continue
        if cons_col not in merged.columns:
            results[axis] = {"kappa": None, "note": "consensus column missing"}
            continue

        model_ints = merged[model_col].map(FLAG_TO_INT).dropna().astype(int)
        cons_ints  = merged[cons_col].dropna().astype(int)

        common = model_ints.index.intersection(cons_ints.index)
        y_model = model_ints.loc[common]
        y_cons  = cons_ints.loc[common]

        if len(y_model) < 2:
            results[axis] = {"kappa": None, "n": 0}
            continue

        try:
            kappa = float(cohen_kappa_score(y_model, y_cons,
                                            weights="quadratic"))
            agree = float((y_model == y_cons).mean() * 100)
            cm    = confusion_matrix(y_cons, y_model,
                                     labels=[1, 2, 3]).tolist()
            results[axis] = {
                "kappa"         : round(kappa, 4),
                "n"             : int(len(y_model)),
                "agreement_pct" : round(agree, 1),
                "confusion_matrix": cm,
                "interpretation": _interpret_kappa(kappa),
            }
        except Exception as e:
            results[axis] = {"kappa": None, "note": str(e)}

    return results


def compute_overall_kappa(merged: pd.DataFrame) -> dict:
    """Overall flag: map model overall_flag vs consensus global_rating."""
    if "overall_flag" not in merged.columns:
        return {"kappa": None, "note": "overall_flag column missing"}
    if "global_rating" not in merged.columns:
        return {"kappa": None, "note": "global_rating column missing"}

    y_model = merged["overall_flag"].map(FLAG_TO_INT).dropna().astype(int)
    y_cons  = merged["global_rating"].dropna().astype(int)
    common  = y_model.index.intersection(y_cons.index)
    y_model = y_model.loc[common]
    y_cons  = y_cons.loc[common]

    if len(y_model) < 2:
        return {"kappa": None, "n": 0}

    kappa = float(cohen_kappa_score(y_model, y_cons, weights="quadratic"))
    agree = float((y_model == y_cons).mean() * 100)
    cm    = confusion_matrix(y_cons, y_model, labels=[1, 2, 3]).tolist()
    return {
        "kappa"            : round(kappa, 4),
        "n"                : int(len(y_model)),
        "agreement_pct"    : round(agree, 1),
        "confusion_matrix" : cm,
        "interpretation"   : _interpret_kappa(kappa),
    }


def compute_spearman(merged: pd.DataFrame) -> dict:
    """Spearman rho between model composite score and reviewer global rating."""
    if "composite_score" not in merged.columns:
        return {"rho": None, "note": "composite_score missing"}
    if "global_rating" not in merged.columns:
        return {"rho": None, "note": "global_rating missing"}

    scores  = merged["composite_score"].dropna()
    ratings = merged["global_rating"].dropna()
    common  = scores.index.intersection(ratings.index)

    rho, p = spearmanr(scores.loc[common], ratings.loc[common])
    return {
        "rho"   : round(float(rho), 4),
        "p_value": round(float(p), 6),
        "n"     : int(len(common)),
        "note"  : (
            "Negative rho expected: higher composite score = better quality, "
            "lower reviewer rating (1=acceptable) = better quality."
        ),
    }


def plot_calibration(merged: pd.DataFrame) -> None:
    if "composite_score" not in merged.columns:
        return
    if "global_rating" not in merged.columns:
        return

    scores  = (merged["composite_score"] * 100
               if merged["composite_score"].max() <= 1.0
               else merged["composite_score"])
    ratings = merged["global_rating"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Scatter
    colors = {1: "#2ecc71", 2: "#f39c12", 3: "#e74c3c"}
    for rating_val, color in colors.items():
        mask = ratings == rating_val
        label = INT_TO_FLAG.get(rating_val, str(rating_val))
        ax1.scatter(
            scores[mask], ratings[mask],
            alpha=0.4, c=color, s=18, label=label
        )
    ax1.set_xlabel("Model Composite Score (0–100)")
    ax1.set_ylabel("Reviewer Global Rating (1=Accept, 3=Repeat)")
    rho_val = compute_spearman(merged)["rho"] or 0
    ax1.set_title(f"Calibration Plot  ρ = {rho_val:.3f}")
    ax1.legend()
    ax1.axvline(40, color="red",    linestyle="--", alpha=0.5, label="Repeat threshold")
    ax1.axvline(70, color="orange", linestyle="--", alpha=0.5, label="Borderline threshold")

    # Box plot
    data_by_rating = {
        v: scores[ratings == v].tolist()
        for v in [1, 2, 3]
        if (ratings == v).sum() > 0
    }
    if data_by_rating:
        ax2.boxplot(
            data_by_rating.values(),
            tick_labels=[f"{INT_TO_FLAG.get(k,k)}\n(n={len(v)})"
                         for k, v in data_by_rating.items()],
        )
        ax2.set_ylabel("Model Composite Score (0–100)")
        ax2.set_title("Score Distribution by Reviewer Rating")

    plt.tight_layout()
    out = FIGURES_DIR / "validation_calibration.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


def plot_confusion_matrix(cm_data: list, axis_name: str) -> None:
    cm = np.array(cm_data)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    labels = ["acceptable", "borderline", "repeat"]
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticks([0, 1, 2]); ax.set_yticklabels(labels)
    ax.set_xlabel("Model Prediction")
    ax.set_ylabel("Reviewer Consensus")
    ax.set_title(f"Confusion Matrix — {axis_name}")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    out = FIGURES_DIR / f"confusion_{axis_name}.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()


def _interpret_kappa(k: float) -> str:
    if k < 0:     return "poor"
    if k < 0.20:  return "slight"
    if k < 0.40:  return "fair"
    if k < 0.60:  return "moderate"
    if k < 0.80:  return "substantial"
    return "almost perfect"


def print_summary(per_axis: dict, overall: dict, spearman: dict) -> None:
    print(f"\n{'Axis':<18} {'Model κ':>10} {'Agreement':>12} {'N':>6}")
    print("-" * 55)
    for axis, row in per_axis.items():
        if row.get("kappa") is None:
            print(f"  {axis:<16} {'N/A':>10}")
        else:
            print(
                f"  {axis:<16} {row['kappa']:>10.4f} "
                f"{row['agreement_pct']:>11.1f}% {row['n']:>6}"
            )
    print("-" * 55)
    ov_k = overall.get("kappa", "N/A")
    ov_a = overall.get("agreement_pct", 0)
    print(f"  {'OVERALL':<16} {str(ov_k):>10} {ov_a:>11.1f}%")
    rho = spearman.get("rho", "N/A")
    print(f"\nSpearman ρ (composite vs global rating): {rho}")


def main() -> None:
    for path in [PREDICTIONS, CONSENSUS]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing: {path}\n"
                "Run src/scripts/run_on_gold_standard.py and "
                "src/analysis/compute_kappa.py first."
            )

    preds, consensus, r1, r2 = load_data()
    merged    = merge_predictions_and_consensus(preds, consensus)
    per_axis  = compute_per_axis_kappa(merged)
    overall   = compute_overall_kappa(merged)
    spearman  = compute_spearman(merged)

    print_summary(per_axis, overall, spearman)
    plot_calibration(merged)

    for axis, row in per_axis.items():
        if row.get("confusion_matrix"):
            plot_confusion_matrix(row["confusion_matrix"], axis)

    output = {
        "n_studies"    : len(merged),
        "per_axis_kappa": per_axis,
        "overall_kappa" : overall,
        "spearman"      : spearman,
    }
    out_path = OUTPUT_DIR / "validation_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()