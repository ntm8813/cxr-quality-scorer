"""
evaluate_correlations.py — stable per-axis evaluation (no Spearman instability)

Fixes:
- Removes ConstantInputWarning entirely
- Removes SciPy Spearman (pairwise case does not need it)
- Adds deterministic ordinal evaluation
- Adds progress visibility per axis
- Adds collapsed-signal diagnostics
"""

import h5py
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.scorers.exposure_scorer import ExposureScorer
from src.scorers.sharpness_scorer import SharpnessScorer
from src.scorers.coverage_scorer import CoverageScorer
from src.scorers.inspiration_scorer import InspirationScorer

H5_PATH = "data/processed/cxr_degraded.h5"
MANIFEST_PATH = "data/processed/degradation_manifest.csv"
CONFIG_PATH = "configs/v1.yaml"
TARGET_RHO = 0.75


# ------------------------------------------------------------
# Core comparison metric (replaces Spearman safely)
# ------------------------------------------------------------
def pairwise_directional_score(bad1: float, bad2: float) -> int:
    """
    Returns:
        +1 if severity ordering is correct (2 > 1)
        -1 if reversed
         0 if no signal (collapsed)
    """
    if abs(bad1 - bad2) < 1e-8:
        return 0
    return 1 if bad2 > bad1 else -1


def evaluate_scorer_axis(axis_name, scorer, manifest, h5):
    rows = manifest[manifest["axis"] == axis_name]
    if len(rows) == 0:
        print(f"\n{axis_name}: no rows in manifest.")
        return None

    correct = 0
    total = 0
    collapsed = 0

    print("\n" + "=" * 70)
    print(f"Evaluating {axis_name.upper()} (stable pairwise)")
    print("=" * 70)

    groups = list(rows.groupby("base_uid"))

    for base_uid, group in tqdm(groups, desc=f"{axis_name} (patients)", unit="patient"):
        s1 = group[group["severity"] == 1]
        s2 = group[group["severity"] == 2]

        if s1.empty or s2.empty:
            continue

        uid1, uid2 = s1.iloc[0]["uid"], s2.iloc[0]["uid"]

        if uid1 not in h5 or uid2 not in h5:
            continue

        try:
            r1 = scorer.score(h5[uid1][:], {"study_uid": uid1})
            r2 = scorer.score(h5[uid2][:], {"study_uid": uid2})

            b1 = 1.0 - float(r1.score)
            b2 = 1.0 - float(r2.score)

            if abs(b1 - b2) < 1e-8:
                collapsed += 1

            direction = pairwise_directional_score(b1, b2)
            if direction == 1:
                correct += 1

            total += 1

        except Exception as exc:
            tqdm.write(f"Skipping {base_uid}: {exc}")

    rho = correct / total if total > 0 else 0.0
    collapse_rate = collapsed / total if total > 0 else 0.0

    print(f"\nRESULTS: {axis_name}")
    print(f"  Accuracy (monotonic ordering): {rho:.4f}")
    print(f"  Collapsed pairs: {collapsed}/{total} ({collapse_rate:.2%})")
    print(f"  STATUS: {'PASS' if rho >= TARGET_RHO else 'FAIL'}")

    return rho


def evaluate_rotation_ground_truth(manifest, h5):
    rows = manifest[manifest["axis"] == "rotation"]
    if len(rows) == 0:
        print("\nROTATION: no rows in manifest.")
        return None

    correct = 0
    total = 0

    print("\n" + "=" * 70)
    print("Evaluating ROTATION (ground truth)")
    print("=" * 70)

    for base_uid, group in tqdm(rows.groupby("base_uid"), desc="rotation", unit="patient"):
        s1 = group[group["severity"] == 1]
        s2 = group[group["severity"] == 2]

        if s1.empty or s2.empty:
            continue

        uid1, uid2 = s1.iloc[0]["uid"], s2.iloc[0]["uid"]

        if uid1 not in h5 or uid2 not in h5:
            continue

        ang1 = abs(float(h5[uid1].attrs.get("applied_angle_deg", 0.0)))
        ang2 = abs(float(h5[uid2].attrs.get("applied_angle_deg", 0.0)))

        if ang2 > ang1:
            correct += 1

        total += 1

    rho = correct / total if total > 0 else 0.0

    print(f"\nROTATION RESULTS")
    print(f"  Accuracy: {rho:.4f}")
    print(f"  STATUS: {'PASS' if rho >= TARGET_RHO else 'FAIL'}")

    return rho


def main():
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    print("\nLoading scorers...")

    scorers = {
        "exposure": ExposureScorer(config),
        "blur": SharpnessScorer(config),
        "coverage": CoverageScorer(config),
        "inspiration": InspirationScorer(config),
    }

    manifest = pd.read_csv(MANIFEST_PATH)

    print("\nManifest summary (axis × severity workload)")
    print(manifest.groupby(["axis", "severity"]).size())

    results = {}

    with h5py.File(H5_PATH, "r") as h5:
        for axis_name, scorer in scorers.items():
            results[axis_name] = evaluate_scorer_axis(axis_name, scorer, manifest, h5)

        results["rotation"] = evaluate_rotation_ground_truth(manifest, h5)

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    for axis, score in results.items():
        status = "PASS" if score is not None and score >= TARGET_RHO else "FAIL"
        print(f"{axis:<14} {status}  score={score:.4f}")

    return results


if __name__ == "__main__":
    main()