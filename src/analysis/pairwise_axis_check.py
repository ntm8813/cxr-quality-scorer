import h5py
import yaml
import pandas as pd

from src.scorers.exposure_scorer import ExposureScorer
from src.scorers.sharpness_scorer import SharpnessScorer
from src.scorers.rotation_scorer import RotationScorer


H5_PATH = "data/processed/cxr_degraded.h5"
MANIFEST_PATH = "data/processed/degradation_manifest.csv"
CONFIG_PATH = "configs/v1.yaml"


def circular_distance_deg(a: float, b: float, period: float = 180.0) -> float:
    """
    Smallest absolute angular difference under a circular period.
    For orientation estimates, 180° periodicity is usually correct.
    """
    diff = (a - b + period / 2.0) % period - period / 2.0
    return abs(float(diff))


def evaluate_rotation(rows, h5, scorer):
    total = 0
    correct = 0

    print("\n" + "=" * 70)
    print("ROTATION PAIRWISE CHECK (SIGNED DELTA METHOD)")
    print("=" * 70)

    for base_uid, group in rows.groupby("base_uid"):
        sev1 = group[group["severity"] == 1]
        sev2 = group[group["severity"] == 2]

        if len(sev1) == 0 or len(sev2) == 0:
            continue

        uid1 = sev1.iloc[0]["uid"]
        uid2 = sev2.iloc[0]["uid"]
        uid0 = f"{base_uid}_clean_0"

        if uid0 not in h5 or uid1 not in h5 or uid2 not in h5:
            continue

        try:
            angle0 = scorer.score(h5[uid0][:], {"study_uid": uid0}).raw_metrics["rotation_angle_deg_signed"]
            angle1 = scorer.score(h5[uid1][:], {"study_uid": uid1}).raw_metrics["rotation_angle_deg_signed"]
            angle2 = scorer.score(h5[uid2][:], {"study_uid": uid2}).raw_metrics["rotation_angle_deg_signed"]

            delta1 = circular_distance_deg(angle1, angle0, period=180.0)
            delta2 = circular_distance_deg(angle2, angle0, period=180.0)

            total += 1
            if delta2 > delta1:
                correct += 1
        except Exception:
            continue

    print(f"TOTAL PAIRS                    = {total}")
    print(f"SEV2 DELTA GREATER THAN SEV1    = {correct}")
    print(f"PERCENT                        = {100 * correct / total:.2f}%")

    return correct / total if total > 0 else 0.0


def evaluate_blur(rows, h5, scorer):
    total = 0
    correct = 0

    print("\n" + "=" * 70)
    print("BLUR PAIRWISE CHECK")
    print("=" * 70)

    for base_uid, group in rows.groupby("base_uid"):
        sev1 = group[group["severity"] == 1]
        sev2 = group[group["severity"] == 2]

        if len(sev1) == 0 or len(sev2) == 0:
            continue

        uid1 = sev1.iloc[0]["uid"]
        uid2 = sev2.iloc[0]["uid"]

        lap1 = scorer.score(h5[uid1][:], {"study_uid": uid1}).raw_metrics["laplacian_variance"]
        lap2 = scorer.score(h5[uid2][:], {"study_uid": uid2}).raw_metrics["laplacian_variance"]

        total += 1
        if lap2 < lap1:
            correct += 1

    print(f"TOTAL PAIRS            = {total}")
    print(f"SEV2 LOWER THAN SEV1   = {correct}")
    print(f"PERCENT                = {100 * correct / total:.2f}%")

    return correct / total if total > 0 else 0.0


def evaluate_exposure(rows, h5, scorer):
    total = 0
    correct = 0

    print("\n" + "=" * 70)
    print("EXPOSURE PAIRWISE CHECK")
    print("=" * 70)

    for base_uid, group in rows.groupby("base_uid"):
        sev1 = group[group["severity"] == 1]
        sev2 = group[group["severity"] == 2]

        if len(sev1) == 0 or len(sev2) == 0:
            continue

        uid1 = sev1.iloc[0]["uid"]
        uid2 = sev2.iloc[0]["uid"]

        score1 = scorer.score(h5[uid1][:], {"study_uid": uid1}).score
        score2 = scorer.score(h5[uid2][:], {"study_uid": uid2}).score

        total += 1
        if score2 < score1:
            correct += 1

    print(f"TOTAL PAIRS                 = {total}")
    print(f"SEV2 SCORE LOWER THAN SEV1  = {correct}")
    print(f"PERCENT                     = {100 * correct / total:.2f}%")

    return correct / total if total > 0 else 0.0


def main():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    exposure_scorer = ExposureScorer(config)
    blur_scorer = SharpnessScorer(config)
    rotation_scorer = RotationScorer(config)

    manifest = pd.read_csv(MANIFEST_PATH)

    with h5py.File(H5_PATH, "r") as h5:
        exposure_rows = manifest[manifest["axis"] == "exposure"]
        blur_rows = manifest[manifest["axis"] == "blur"]
        rotation_rows = manifest[manifest["axis"] == "rotation"]

        exposure_pct = evaluate_exposure(exposure_rows, h5, exposure_scorer)
        blur_pct = evaluate_blur(blur_rows, h5, blur_scorer)
        rotation_pct = evaluate_rotation(rotation_rows, h5, rotation_scorer)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Exposure : {exposure_pct * 100:.2f}%")
    print(f"Blur     : {blur_pct * 100:.2f}%")
    print(f"Rotation : {rotation_pct * 100:.2f}%")

    print("\nInterpretation:")
    print("~50%  = random")
    print("~70%  = weak")
    print("~85%  = good")
    print("~95%+ = excellent")


if __name__ == "__main__":
    main()