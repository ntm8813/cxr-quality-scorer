import h5py
import yaml
import pandas as pd
import numpy as np

from src.scorers.exposure_scorer import ExposureScorer
from src.scorers.sharpness_scorer import SharpnessScorer
from src.scorers.rotation_scorer import RotationScorer

H5_PATH = "data/processed/cxr_degraded.h5"
MANIFEST_PATH = "data/processed/degradation_manifest.csv"
CONFIG_PATH = "configs/v1.yaml"


def circular_distance_deg(a: float, b: float, period: float = 180.0) -> float:
    diff = (a - b + period / 2.0) % period - period / 2.0
    return abs(float(diff))


def dataset_exists(h5, uid: str) -> bool:
    if uid in h5:
        return True

    if "images" in h5 and uid in h5["images"]:
        return True

    return False


def load_image(h5, uid: str):
    if uid in h5:
        return h5[uid][:]

    if "images" in h5 and uid in h5["images"]:
        return h5["images"][uid][:]

    raise KeyError(uid)


def evaluate_axis(axis_name, scorer, manifest, h5):
    rows = manifest[manifest["axis"] == axis_name]

    if len(rows) == 0:
        print(f"\n{axis_name}: no rows found")
        return

    total_pairs = 0
    strict_correct = 0
    ties = 0
    margins = []

    print("\n" + "=" * 70)
    print(f"Evaluating {axis_name.upper()}")
    print("=" * 70)

    print("\nDEBUG")
    print(f"Rows found               = {len(rows)}")
    print(f"Unique base_uid          = {rows['base_uid'].nunique()}")
    print(f"Severities present       = {sorted(rows['severity'].unique())}")

    missing_severity_pairs = 0
    missing_h5_pairs = 0
    scorer_failures = 0

    for idx, (base_uid, group) in enumerate(rows.groupby("base_uid"), start=1):

        if idx % 100 == 0:
            print(
                f"{axis_name}: "
                f"{idx}/{rows['base_uid'].nunique()} "
                f"({100*idx/rows['base_uid'].nunique():.1f}%)"
            )

        sev1 = group[group["severity"] == 1]
        sev2 = group[group["severity"] == 2]

        if len(sev1) == 0 or len(sev2) == 0:
            missing_severity_pairs += 1
            continue

        uid1 = str(sev1.iloc[0]["uid"])
        uid2 = str(sev2.iloc[0]["uid"])
        uid0 = f"{base_uid}_clean_0"

        if not dataset_exists(h5, uid0):
            missing_h5_pairs += 1
            continue

        if not dataset_exists(h5, uid1):
            missing_h5_pairs += 1
            continue

        if not dataset_exists(h5, uid2):
            missing_h5_pairs += 1
            continue

        try:

            if axis_name == "rotation":

                clean_angle = scorer.score(
                    load_image(h5, uid0),
                    {"study_uid": uid0}
                ).raw_metrics["rotation_angle_deg_signed"]

                angle1 = scorer.score(
                    load_image(h5, uid1),
                    {"study_uid": uid1}
                ).raw_metrics["rotation_angle_deg_signed"]

                angle2 = scorer.score(
                    load_image(h5, uid2),
                    {"study_uid": uid2}
                ).raw_metrics["rotation_angle_deg_signed"]

                badness1 = circular_distance_deg(
                    angle1,
                    clean_angle,
                    period=180.0
                )

                badness2 = circular_distance_deg(
                    angle2,
                    clean_angle,
                    period=180.0
                )

            elif axis_name == "blur":

                result1 = scorer.score(
                    load_image(h5, uid1),
                    {"study_uid": uid1}
                )

                result2 = scorer.score(
                    load_image(h5, uid2),
                    {"study_uid": uid2}
                )

                score1 = float(result1.score)
                score2 = float(result2.score)

                badness1 = 1.0 - score1
                badness2 = 1.0 - score2

            else:

                result1 = scorer.score(
                    load_image(h5, uid1),
                    {"study_uid": uid1}
                )

                result2 = scorer.score(
                    load_image(h5, uid2),
                    {"study_uid": uid2}
                )

                score1 = float(result1.score)
                score2 = float(result2.score)

                badness1 = 1.0 - score1
                badness2 = 1.0 - score2

            total_pairs += 1

            margin = badness2 - badness1
            margins.append(margin)

            if np.isclose(badness1, badness2):
                ties += 1
            elif badness2 > badness1:
                strict_correct += 1

        except Exception as e:
            scorer_failures += 1

            if scorer_failures <= 20:
                print(
                    f"SCORER FAILURE [{axis_name}] "
                    f"{base_uid}: "
                    f"{type(e).__name__}: {e}"
                )

            continue

    print(f"Missing severity pairs   = {missing_severity_pairs}")
    print(f"Missing H5 pairs         = {missing_h5_pairs}")
    print(f"Scorer failures          = {scorer_failures}")

    if total_pairs == 0:
        print("No valid pairs")
        return

    strict_accuracy = strict_correct / total_pairs
    weighted_accuracy = (strict_correct + 0.5 * ties) / total_pairs
    mean_margin = float(np.mean(margins)) if margins else 0.0
    median_margin = float(np.median(margins)) if margins else 0.0
    tie_rate = ties / total_pairs

    print(f"Total pairs               = {total_pairs}")
    print(f"Strict correct pairs      = {strict_correct}")
    print(f"Ties                      = {ties}")
    print(f"Strict accuracy           = {strict_accuracy:.4f}")
    print(f"Weighted accuracy         = {weighted_accuracy:.4f}")
    print(f"Mean margin (sev2-sev1)   = {mean_margin:.6f}")
    print(f"Median margin             = {median_margin:.6f}")
    print(f"Tie rate                  = {tie_rate:.4f}")

    if weighted_accuracy >= 0.75:
        print("STATUS                    : PASS")
    else:
        print("STATUS                    : FAIL")


def main():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    scorers = {
        "rotation": RotationScorer(config),
        "exposure": ExposureScorer(config),
        "blur": SharpnessScorer(config),
    }

    manifest = pd.read_csv(MANIFEST_PATH)

    print("\nManifest summary")
    print(manifest.groupby(["axis", "severity"]).size())

    with h5py.File(H5_PATH, "r") as h5:

        print("\nH5 TOP LEVEL KEYS")
        print(list(h5.keys())[:20])

        evaluate_axis(
            "rotation",
            scorers["rotation"],
            manifest,
            h5
        )

        evaluate_axis(
            "exposure",
            scorers["exposure"],
            manifest,
            h5
        )

        evaluate_axis(
            "blur",
            scorers["blur"],
            manifest,
            h5
        )

    print("\nFinished.")


if __name__ == "__main__":
    main()