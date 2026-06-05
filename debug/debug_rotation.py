import os
import sys
import h5py
import yaml
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.scorers.rotation_scorer import RotationScorer

MANIFEST_PATH = os.path.join(project_root, "data/processed/degradation_manifest.csv")
H5_PATH = os.path.join(project_root, "data/processed/cxr_degraded.h5")
CONFIG_PATH = os.path.join(project_root, "configs/v1.yaml")


def circular_distance_deg(a: float, b: float, period: float = 180.0) -> float:
    diff = (a - b + period / 2.0) % period - period / 2.0
    return abs(float(diff))


def main():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    scorer = RotationScorer(config)
    df = pd.read_csv(MANIFEST_PATH)
    rows = df[df["axis"] == "rotation"]

    print("\n" + "=" * 88)
    print(f"{'Base UID':<14} | {'Clean':<8} | {'Sev1':<8} | {'Sev2':<8} | {'Δ1':<8} | {'Δ2':<8} | {'S2>S1?':<7}")
    print("=" * 88)

    count = 0
    with h5py.File(H5_PATH, "r") as h5:
        for base_uid, group in rows.groupby("base_uid"):
            sev1 = group[group["severity"] == 1]
            sev2 = group[group["severity"] == 2]

            if len(sev1) == 0 or len(sev2) == 0:
                continue

            uid0 = f"{base_uid}_clean_0"
            uid1, uid2 = sev1.iloc[0]["uid"], sev2.iloc[0]["uid"]

            if uid0 not in h5 or uid1 not in h5 or uid2 not in h5:
                continue

            angle0 = scorer.score(h5[uid0][:], {"study_uid": uid0}).raw_metrics["rotation_angle_deg_signed"]
            angle1 = scorer.score(h5[uid1][:], {"study_uid": uid1}).raw_metrics["rotation_angle_deg_signed"]
            angle2 = scorer.score(h5[uid2][:], {"study_uid": uid2}).raw_metrics["rotation_angle_deg_signed"]

            delta1 = circular_distance_deg(angle1, angle0, period=180.0)
            delta2 = circular_distance_deg(angle2, angle0, period=180.0)

            is_correct = delta2 > delta1

            print(
                f"{str(base_uid)[:14]:<14} | "
                f"{angle0:<8.2f} | "
                f"{angle1:<8.2f} | "
                f"{angle2:<8.2f} | "
                f"{delta1:<8.2f} | "
                f"{delta2:<8.2f} | "
                f"{str(is_correct):<7}"
            )

            count += 1
            if count >= 20:
                break


if __name__ == "__main__":
    main()