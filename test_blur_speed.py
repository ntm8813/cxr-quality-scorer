# test_blur_speed.py

import pandas as pd
import h5py
import yaml
import time

from src.scorers.sharpness_scorer import SharpnessScorer

cfg = yaml.safe_load(open("configs/v1.yaml"))
scorer = SharpnessScorer(cfg)

manifest = pd.read_csv("data/processed/degradation_manifest.csv")
uid = manifest[manifest["axis"] == "blur"].iloc[0]["uid"]

h5 = h5py.File("data/processed/cxr_degraded.h5", "r")
img = h5[uid][:]

start = time.time()

for _ in range(100):
    scorer.score(img, {"study_uid": uid})

print((time.time() - start) / 100)