
# CXR QUALITY PIPELINE FINAL REPORT

====================================================

## 1. RUNTIME PERFORMANCE
CXR PIPELINE RUNTIME REPORT

Samples evaluated : 100
Average runtime   : 0.1351 sec
P95 runtime       : 0.1701 sec
Min runtime       : 0.0958 sec
Max runtime       : 0.2263 sec

Requirement (<2s) : PASS

====================================================

## 2. STATISTICAL METRICS (MAE + SPEARMAN)
=== MAE REPORT (vs composite_score) ===
exposure     | MAE=0.1829 | Spearman ρ=0.2370
sharpness    | MAE=0.1904 | Spearman ρ=0.5986
rotation     | MAE=0.6109 | Spearman ρ=0.2673
coverage     | MAE=0.2834 | Spearman ρ=0.4768
inspiration  | MAE=0.3096 | Spearman ρ=0.2994

=== GLOBAL COMPOSITE DISTRIBUTION ===
Mean : 0.6660059999999999
Std  : 0.07964151156275226
Min  : 0.4753
Max  : 0.8997

====================================================

## 3. SEGMENTATION CONSISTENCY (DICE)
CXR SEGMENTATION ROBUSTNESS REPORT

Samples evaluated : 100
Mean Dice score   : 0.9383


====================================================

## FINAL STATUS

- Runtime constraint (<2s): PASSED
- 100-study evaluation: COMPLETED
- MAE + Spearman analysis: COMPLETED
- Dice segmentation robustness: VERIFIED

====================================================

DATA SOURCES:
- evaluation_results.json
- NIH subset (100 samples)
====================================================
