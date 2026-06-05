import h5py
import yaml
import pandas as pd

from src.scorers.rotation_scorer import RotationScorer

manifest = pd.read_csv("data/processed/degradation_manifest.csv")

rows = manifest[manifest["axis"] == "rotation"]

base_uid, group = next(iter(rows.groupby("base_uid")))

uid0 = f"{base_uid}_clean_0"

with open("configs/v1.yaml", "r") as f:
    cfg = yaml.safe_load(f)

scorer = RotationScorer(cfg)

with h5py.File("data/processed/cxr_degraded.h5", "r") as h5:
    print("loading image")
    image = h5[uid0][:]

    print("running scorer")
    result = scorer.score(image, {"study_uid": uid0})

    print("done")
    print(result.score)