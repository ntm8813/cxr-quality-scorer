# test_blur_eval.py

import pandas as pd
import h5py
import yaml

from src.scorers.sharpness_scorer import SharpnessScorer

cfg = yaml.safe_load(open("configs/v1.yaml"))
scorer = SharpnessScorer(cfg)

manifest = pd.read_csv("data/processed/degradation_manifest.csv")
rows = manifest[manifest["axis"] == "blur"]

h5 = h5py.File("data/processed/cxr_degraded.h5", "r")

success = 0

for i, (_, row) in enumerate(rows.head(100).iterrows()):
    uid = row["uid"]

    try:
        result = scorer.score(
            h5[uid][:],
            {"study_uid": uid}
        )

        success += 1

    except Exception as e:
        print("FAILED:", uid)
        print(type(e).__name__, e)
        break

print("success =", success)