# src/analysis/threshold_tuning.py
# python -m src.analysis.threshold_tuning
"""
Day 24 — Grid search over score_ranges thresholds.
Tests repeat_max and borderline_max combinations.
Reports which combination maximises weighted kappa vs reviewer consensus.
Updates configs/v1.yaml if improvement found.
"""
from __future__ import annotations

import json
import yaml
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

PREDICTIONS = Path("data/predictions/model_v1.csv")
CONSENSUS   = Path("data/gold_standard_consensus.csv")
CONFIG_PATH = Path("configs/v1.yaml")

FLAG_TO_INT = {"acceptable": 1, "borderline": 2, "repeat": 3}

def score_to_flag(score: float, repeat_max: float,
                  borderline_max: float) -> int:
    if score <= repeat_max / 100:
        return 3
    if score <= borderline_max / 100:
        return 2
    return 1


def main() -> None:
    preds    = pd.read_csv(PREDICTIONS)
    consensus= pd.read_csv(CONSENSUS)
    merged   = preds.merge(consensus, on="study_uid", how="inner")

    if "composite_score" not in merged.columns:
        print("composite_score column missing from predictions.")
        return
    if "global_rating" not in merged.columns:
        print("global_rating column missing from consensus.")
        return

    scores   = merged["composite_score"].values
    ratings  = merged["global_rating"].fillna(2).astype(int).values

    # Grid search
    repeat_candidates    = list(range(25, 55, 5))   # 25,30,35,40,45,50
    borderline_candidates= list(range(55, 80, 5))   # 55,60,65,70,75

    best_kappa  = -999
    best_repeat = 40
    best_border = 70
    results     = []

    print(f"Testing {len(repeat_candidates)*len(borderline_candidates)} threshold combinations...")

    for r_max in repeat_candidates:
        for b_max in borderline_candidates:
            if b_max <= r_max:
                continue
            model_ints = np.array([
                score_to_flag(s, r_max, b_max) for s in scores
            ])
            try:
                kappa = cohen_kappa_score(ratings, model_ints,
                                          weights="quadratic")
                results.append({
                    "repeat_max"   : r_max,
                    "borderline_max": b_max,
                    "kappa"        : round(float(kappa), 4),
                })
                if kappa > best_kappa:
                    best_kappa  = kappa
                    best_repeat = r_max
                    best_border = b_max
            except Exception:
                pass

    results_df = pd.DataFrame(results).sort_values("kappa", ascending=False)
    print(f"\nTop 5 configurations:")
    print(results_df.head(5).to_string(index=False))
    print(f"\nBest: repeat_max={best_repeat}, borderline_max={best_border}, kappa={best_kappa:.4f}")

    # Load current config and compare
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    current_repeat  = config["score_ranges"]["repeat_max"]
    current_border  = config["score_ranges"]["borderline_max"]
    current_ints    = np.array([
        score_to_flag(s, current_repeat, current_border) for s in scores
    ])
    current_kappa   = float(cohen_kappa_score(ratings, current_ints,
                                               weights="quadratic"))
    print(f"\nCurrent thresholds: repeat_max={current_repeat}, "
          f"borderline_max={current_border}, kappa={current_kappa:.4f}")

    if best_kappa > current_kappa + 0.01:
        print(f"\nImprovement found (+{best_kappa-current_kappa:.4f}). Updating v1.yaml...")
        config["score_ranges"]["repeat_max"]    = best_repeat
        config["score_ranges"]["borderline_max"]= best_border
        with open(CONFIG_PATH, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"Updated configs/v1.yaml")
    else:
        print(f"\nNo meaningful improvement found. Keeping current thresholds.")

    # Save grid results
    results_df.to_csv("reports/threshold_grid_search.csv", index=False)
    print("Saved → reports/threshold_grid_search.csv")


if __name__ == "__main__":
    main()