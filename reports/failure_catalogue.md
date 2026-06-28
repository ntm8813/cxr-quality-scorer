# Failure Mode Catalogue — MTV-INT-RAD-003

## Summary
- Studies analysed: **300**
- Per-axis disagreements: **1057** (50.3% of all axis evaluations)
- Overall flag disagreements: **147** (49.0% of studies)

## Rotation Estimation Method Distribution

_'rotation_estimation_method' not found in data\predictions\model_v1.csv. This check requires the pipeline to flatten raw_metrics['estimation_method'] into the predictions CSV. Skipping skew check._

## Failure Mode Definitions

- **FN_REPEAT**: False negative — model missed a repeat-quality image
- **FP_REPEAT**: False positive — model incorrectly flagged acceptable image as repeat
- **FN_BORDER**: False negative — model missed a borderline image (called it acceptable)
- **FP_BORDER**: False positive — model called borderline image acceptable
- **OVER_PENALISE**: Model more strict than reviewer consensus
- **UNDER_PENALISE**: Model more lenient than reviewer consensus

## Per-Axis Failure Counts

| Axis | FN_REPEAT | FP_REPEAT | FN_BORDER | FP_BORDER | OVER | UNDER |
|------|-----------|-----------|-----------|-----------|------|-------|
| sharpness | 1 | 0 | 58 | 52 | 0 | 0 |
| exposure | 7 | 0 | 100 | 4 | 0 | 0 |
| rotation | 0 | 217 | 4 | 31 | 0 | 0 |
| coverage | 2 | 0 | 67 | 6 | 0 | 0 |
| inspiration | 0 | 117 | 0 | 117 | 0 | 0 |
| artifact | 0 | 250 | 13 | 6 | 0 | 0 |
| metadata | 0 | 0 | 5 | 0 | 0 | 0 |

## Top 20 Worst Disagreements (by severity)

| Study UID | Axis | Model | Consensus | Mode |
|-----------|------|-------|-----------|------|
| 00006583_003 | inspiration | repeat | 1 | FP_REPEAT |
| 00006527_001 | rotation | repeat | 1 | FP_REPEAT |
| 00004006_016 | rotation | repeat | 1 | FP_REPEAT |
| 00004001_003 | rotation | repeat | 1 | FP_REPEAT |
| 00003990_016 | artifact | repeat | 1 | FP_REPEAT |
| 00003990_016 | rotation | repeat | 1 | FP_REPEAT |
| 00003990_016 | exposure | acceptable | 3 | FN_REPEAT |
| 00003967_000 | artifact | repeat | 1 | FP_REPEAT |
| 00006519_012 | rotation | repeat | 1 | FP_REPEAT |
| 00006517_000 | artifact | repeat | 1 | FP_REPEAT |
| 00006517_000 | inspiration | repeat | 1 | FP_REPEAT |
| 00006502_001 | rotation | repeat | 1 | FP_REPEAT |
| 00006482_000 | artifact | repeat | 1 | FP_REPEAT |
| 00006482_000 | rotation | repeat | 1 | FP_REPEAT |
| 00006476_000 | artifact | repeat | 1 | FP_REPEAT |
| 00006476_000 | rotation | repeat | 1 | FP_REPEAT |
| 00006469_001 | artifact | repeat | 1 | FP_REPEAT |
| 00006469_001 | inspiration | repeat | 1 | FP_REPEAT |
| 00006469_001 | rotation | repeat | 1 | FP_REPEAT |
| 00006469_000 | artifact | repeat | 1 | FP_REPEAT |

## Interpretation

### Critical failures (FN_REPEAT)
Images the model called acceptable or borderline that reviewers flagged as repeat. These are the most clinically dangerous disagreements — a poor-quality image passed to a clinician.

### False alarms (FP_REPEAT)
Images the model flagged as repeat that reviewers considered acceptable. These cause unnecessary repeat exposures.

### Borderline misses (FN_BORDER)
Borderline images scored as acceptable by the model. Lower clinical risk but indicates the model's thresholds are too lenient.

### Systematic bias
Compare OVER_PENALISE vs UNDER_PENALISE totals per axis. A consistent direction indicates a threshold calibration issue rather than a model accuracy issue.

## Recommended Threshold Adjustments

_To be filled in after Day 24 review session._

| Axis | Current repeat_max | Suggested adjustment | Rationale |
|------|--------------------|----------------------|-----------|
| sharpness | 40 | TBD | TBD |
| exposure  | 40 | TBD | TBD |
| rotation  | 40 | TBD | Re-evaluate after RotationScorer v2 (mask-anchored) |

---
_Generated automatically by src/analysis/error_analysis.py_