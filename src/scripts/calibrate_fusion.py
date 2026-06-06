# scripts/calibrate_fusion.py
# python -m src.scripts.calibrate_fusion

from __future__ import annotations

import h5py
import yaml
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.stats import spearmanr, ConstantInputWarning
from tqdm import tqdm

from src.scorers.exposure_scorer import ExposureScorer
from src.scorers.sharpness_scorer import SharpnessScorer
from src.scorers.metadata_scorer import MetadataScorer
from src.scorers.rotation_scorer import RotationScorer

CONFIG_PATH   = "configs/v1.yaml"
H5_PATH       = "data/processed/cxr_degraded.h5"
MANIFEST_PATH = "data/processed/degradation_manifest.csv"

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

# ── Load manifest ─────────────────────────────────────────────

df = pd.read_csv(MANIFEST_PATH)

print(f"Manifest rows: {len(df)}")
print(f"Columns      : {df.columns.tolist()}")
print(f"Sample:\n{df.head(3)}\n")

KEY_COL = next(
    (c for c in ["study_uid", "uid", "key", "filename", "image_id"] if c in df.columns),
    df.columns[0]
)

SEV_COL = next(
    (c for c in ["severity", "degradation_level", "level", "severity_level"] if c in df.columns),
    None
)

print(f"Using key_col='{KEY_COL}', sev_col='{SEV_COL}'")

if SEV_COL is None:
    print("WARNING: No severity column found. Assigning severity=0 to all rows.")
    df["severity"] = 0
    SEV_COL = "severity"

# ── Sample for speed ──────────────────────────────────────────

df_sample = df.sample(min(400, len(df)), random_state=42).reset_index(drop=True)

scorers = [
    ExposureScorer(config),
    SharpnessScorer(config),
    MetadataScorer(config),
    RotationScorer(config),
]

SCORER_TO_AXIS = {
    "ExposureScorer": "exposure",
    "SharpnessScorer": "sharpness",
    "MetadataScorer": "metadata",
    "RotationScorer": "rotation",
}

# ── Score each sampled image ──────────────────────────────────

records = []

print("Scoring images...")

with h5py.File(H5_PATH, "r") as f:
    h5_keys = set(f.keys())

    for _, row in tqdm(
        df_sample.iterrows(),
        total=len(df_sample),
        desc="Processing"
    ):
        key = str(row[KEY_COL])
        sev = int(row[SEV_COL])

        if key not in h5_keys:
            continue

        try:
            image = f[key][:]
            metadata = {"study_uid": key}

            axis_scores = {}

            for scorer in scorers:
                result = scorer.score(image, metadata)

                axis_key = SCORER_TO_AXIS[scorer.__class__.__name__]
                axis_scores[axis_key] = result.score

            records.append({
                "key": key,
                "severity": sev,
                **axis_scores,
            })

        except Exception:
            pass

print(f"Scored: {len(records)} / {len(df_sample)} images")

if len(records) == 0:
    print("\nERROR: No images scored.")
    print(f"Sample manifest keys : {df[KEY_COL].head(5).tolist()}")

    with h5py.File(H5_PATH, "r") as f:
        print(f"Sample HDF5 keys    : {list(f.keys())[:5]}")

    raise SystemExit(1)

scored_df = pd.DataFrame(records)

# ── Per-axis Spearman correlation ─────────────────────────────

print("\nPer-axis Spearman ρ (score vs severity — expect negative):")

for axis in ["exposure", "sharpness", "metadata", "rotation"]:
    if axis in scored_df.columns:

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConstantInputWarning)
            rho, p = spearmanr(scored_df[axis], scored_df["severity"])

        if np.isnan(rho):
            print(
                f"  {axis:<15}: ρ = N/A  "
                "(constant output — expected for metadata on synthetic data)"
            )
        else:
            print(f"  {axis:<15}: ρ = {rho:+.4f}  (p={p:.4f})")

# ── Default weights from config ───────────────────────────────

base_w = {k: float(v) for k, v in config["axis_weights"].items()}


def composite(row: dict, weights: dict) -> float:
    ws = 0.0
    wt = 0.0

    for axis, weight in weights.items():
        if axis in row:
            ws += row[axis] * weight
            wt += weight

    return ws / wt if wt > 0 else 0.5


# ── Neighbourhood grid search ─────────────────────────────────

step = 0.05

candidates = []

axes_to_search = [
    axis
    for axis in base_w.keys()
    if axis in scored_df.columns
]

for axis_key in axes_to_search:
    for delta in [-step, 0.0, step]:

        weights = dict(base_w)

        weights[axis_key] = max(
            0.01,
            weights[axis_key] + delta
        )

        total = sum(weights.values())

        candidates.append({
            k: v / total
            for k, v in weights.items()
        })

print(f"\nTesting {len(candidates)} weight configurations...")

best_rho = -999.0
best_w = dict(base_w)

for weights in candidates:

    comp_scores = [
        composite(record, weights)
        for record in records
    ]

    severities = [
        record["severity"]
        for record in records
    ]

    rho, _ = spearmanr(comp_scores, severities)

    rho_quality = -rho

    if rho_quality > best_rho:
        best_rho = rho_quality
        best_w = dict(weights)

print(f"\nBest Spearman ρ (composite vs quality): {best_rho:.4f}")

print("Best weights:")

for axis, value in best_w.items():
    marker = ""

    if abs(value - base_w.get(axis, value)) > 0.001:
        marker = " ← changed"

    print(f"  {axis:<15}: {value:.4f}{marker}")

# ── Calibration plot ──────────────────────────────────────────

comp_best = [
    composite(record, best_w)
    for record in records
]

sevs_best = [
    record["severity"]
    for record in records
]

Path("reports/figures").mkdir(
    parents=True,
    exist_ok=True
)

fig, (ax1, ax2) = plt.subplots(
    1,
    2,
    figsize=(13, 5)
)

ax1.scatter(
    sevs_best,
    comp_best,
    alpha=0.35,
    c="steelblue",
    s=16
)

ax1.set_xlabel(
    "Known Synthetic Severity (0=Clean, 1=Mild, 2=Severe)"
)

ax1.set_ylabel(
    "Composite Score (0–1)"
)

ax1.set_title(
    f"Calibration Plot  ρ_quality = {best_rho:.3f}"
)

ax1.axhline(
    config["score_ranges"]["borderline_max"] / 100,
    color="orange",
    linestyle="--",
    label="Borderline threshold"
)

ax1.axhline(
    config["score_ranges"]["repeat_max"] / 100,
    color="red",
    linestyle="--",
    label="Repeat threshold"
)

ax1.legend()

data_by_sev = {
    severity: [
        score
        for score, sev in zip(comp_best, sevs_best)
        if sev == severity
    ]
    for severity in sorted(set(sevs_best))
}

if data_by_sev:
    ax2.boxplot(
        data_by_sev.values(),
        tick_labels=[f"Severity {k}" for k in data_by_sev]
    )

    ax2.set_ylabel("Composite Score (0–1)")
    ax2.set_title("Score Distribution by Severity")

plt.tight_layout()

plt.savefig(
    "reports/figures/fusion_calibration.png",
    dpi=150
)

print("\nSaved: reports/figures/fusion_calibration.png")

# ── Save results ──────────────────────────────────────────────

output = {
    "best_rho_quality": round(best_rho, 4),
    "best_weights": {
        k: round(v, 4)
        for k, v in best_w.items()
    },
    "n_images_scored": len(records),
    "note": (
        "If best_rho_quality > current, copy best_weights "
        "into configs/v1.yaml axis_weights section."
    ),
}

Path("reports").mkdir(exist_ok=True)

with open("reports/calibration_results.json", "w") as f:
    json.dump(output, f, indent=2)

print("Saved: reports/calibration_results.json")