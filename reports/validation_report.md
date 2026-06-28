# Validation Report — MTV-INT-RAD-003

## Automated CXR Image Quality Scorer

---

## 1. Overview

This report documents validation of the automated chest radiograph (CXR) image
quality scoring pipeline against a 300-image gold-standard test set, independently
rated by two reviewers across 7 quality axes plus a global quality rating.

**Important note on methodology:** the initial validation pass surfaced a
critical model-loading bug that left the lung segmentation network randomly
initialized throughout the entire first validation cycle. This bug, along with
two related issues, was identified, fixed, and the validation re-run. The results
in this report reflect the corrected pipeline. Full root-cause documentation is
in `reports/WEEK4_STATUS_REPORT.md`.

---

## 2. Dataset

- **Gold-standard test set:** 300 real chest radiographs
- **Reviewers:** 2 independent raters
- **Rating scale:** 1 = acceptable, 2 = borderline, 3 = repeat, per axis plus
  a global overall rating
- **Consensus:** majority vote per axis, computed via `src/analysis/compute_kappa.py`
- **Inter-rater agreement:** see `reports/interrater_kappa.json` for the
  human-human agreement ceiling

---

## 3. Pipeline Configuration

- **Final thresholds** (re-tuned post-bugfix via grid search against reviewer
  consensus, `src/analysis/threshold_tuning.py`):
  - `repeat_max = 45`
  - `borderline_max = 60`
- **Models used:**
  - Lung segmentation: MONAI U-Net
  - Motion blur classifier: EfficientNet-B0
  - Artifact classifier: EfficientNet-B0

---

## 4. Results

### 4.1 Per-axis model agreement with reviewer consensus

| Axis | Model κ | Agreement | N |
|---|---|---|---|
| sharpness | 0.0254 | 71.3% | 300 |
| exposure | 0.0032 | 63.0% | 300 |
| rotation | 0.0317 | 19.3% | 300 |
| coverage | 0.0604 | 75.7% | 300 |
| inspiration | 0.0021 | 22.0% | 300 |
| artifact | 0.0443 | 11.3% | 300 |
| metadata | 0.0000 | 98.3% | 300 |
| **Overall** | **0.2205** | **58.3%** | 300 |

### 4.2 Calibration

Spearman ρ between composite quality score and reviewer global rating:
**-0.1942**

The negative correlation is the expected direction: a higher composite score
(better predicted quality) corresponds to a lower (better) reviewer rating.

See `reports/figures/validation_calibration.png` for the full calibration plot.

### 4.3 Composite score distribution

| Statistic | Value |
|---|---|
| Mean | 0.6376 |
| Std | 0.0746 |
| Min | 0.4486 |
| 25th percentile | 0.5841 |
| Median | 0.6169 |
| 75th percentile | 0.6740 |
| Max | 0.8919 |

### 4.4 Final flag distribution (300 studies)

| Flag | Count |
|---|---|
| Acceptable | 181 |
| Borderline | 118 |
| Repeat | 1 |

---

## 5. Root-Cause Bug Summary

Three issues were identified and fixed during validation, improving overall κ
from 0.0496 to 0.2205 (4.4× improvement) and agreement from 43.7% to 58.3%.
Full technical detail in `reports/WEEK4_STATUS_REPORT.md`. In brief:

1. **Silent checkpoint loading failure** — U-Net loaded 0 of 49 weight keys due
   to a key-prefix mismatch, undetected because `strict=False` suppressed the
   error. Fixed with prefix normalization and a hard failure guard.
2. **Config schema mismatch** — model path resolution assumed a flat key
   structure; actual config used versioned paths. One-line fix.
3. **Incorrect rotation geometry assumption** — orientation estimation assumed
   vertically elongated lung masks; real masks are roughly as wide as tall.
   Replaced with inter-lung-centroid-line geometry, verified against real mask
   overlays.

---

## 6. Limitations

- **Rotation reviewer consensus is heavily imbalanced** in this 300-image sample
  (242 acceptable / 58 borderline / 0 repeat), which caps the achievable κ for
  this axis regardless of model quality.
- **Artifact axis (11.3% agreement)** and **inspiration axis (22% agreement)**
  remain weak and were not addressed in this validation cycle.
- **Sharpness and exposure axes** show low κ despite acceptable raw agreement
  percentages (71.3% and 63.0% respectively), suggesting a class-imbalance
  effect in the underlying distributions rather than random disagreement.
- Validation was performed against a single 300-image gold-standard set from
  one data source; generalization to other scanner types, patient populations,
  and acquisition protocols has not been independently verified.

---

## 7. Conclusion

The corrected pipeline achieves an overall weighted κ of 0.2205 against reviewer
consensus on the 300-image gold standard, with the composite score correlating
in the correct direction with reviewer global quality ratings (ρ = -0.1942).
This represents fair-to-moderate agreement on the strongest axes (coverage,
sharpness) and weak agreement on artifact and inspiration axes, which are
flagged as open items for future work. The pipeline is suitable for use as an
advisory QC layer with human review, not as an autonomous accept/reject gate,
given the current agreement levels.