# CXR QUALITY PIPELINE FINAL REPORT

====================================================

## 1. RUNTIME PERFORMANCE
CXR PIPELINE RUNTIME REPORT

Samples evaluated : 100
Average runtime   : 0.1063 sec
P95 runtime       : 0.1317 sec
Min runtime       : 0.0828 sec
Max runtime       : 0.1520 sec

Requirement (<2s) : PASS

====================================================

## 2. STATISTICAL METRICS (MAE + SPEARMAN)
=== MAE REPORT (vs composite_score) ===
exposure     | MAE=0.1829 | Spearman ρ=0.23699441042534036
sharpness    | MAE=0.1904 | Spearman ρ=0.5985586450833184
rotation     | MAE=0.6109 | Spearman ρ=0.2673190996770027
coverage     | MAE=0.2834 | Spearman ρ=0.476829028784129
inspiration  | MAE=0.3096 | Spearman ρ=0.2994271636622342

=== GLOBAL COMPOSITE DISTRIBUTION ===
Mean : 0.6660059999999999
Std  : 0.07964151156275226
Min  : 0.4753
Max  : 0.8997

====================================================

## 3. SEGMENTATION CONSISTENCY (DICE)
CXR SEGMENTATION ROBUSTNESS REPORT

Samples evaluated : 100
Mean Dice score   : 0.9388


====================================================

## FINAL STATUS

- Runtime constraint (<2s): VERIFIED
- 100-study evaluation: COMPLETED
- MAE + Spearman analysis: COMPLETED
- Dice robustness: VERIFIED

====================================================