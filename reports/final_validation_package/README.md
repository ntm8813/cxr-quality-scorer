# Validation Package — MTV-INT-RAD-003

This directory contains the complete validation outputs for the final closeout of the CXR Quality Scorer project.

## Contents

### interrater_kappa.json

Human inter-rater agreement for each quality axis and the overall global rating. The global rating κ (0.5248) represents the practical ceiling against which model performance should be interpreted.

---

### validation_results.json

Model validation against reviewer consensus, including:

- per-axis Cohen's κ
- overall κ
- agreement percentages
- confusion matrices
- Spearman correlation

Validation cohort:

- 300 chest radiographs

---

### list_a_reporting_fixes.md

Additional reporting requested during the post-delivery review, including:

- comparison against the human agreement ceiling
- minority-class catch rate
- precision / recall / F1
- confusion matrices
- explanation of overall versus per-axis κ

---

### latency_cpu.json

CPU benchmark for the complete inference pipeline.

The reported timings include:

- image ingestion
- preprocessing
- all quality scorers
- score fusion
- explanation generation

---

### failure_catalogue.md

Summary of representative failure modes identified during validation.

---

### failure_catalogue_images/

Representative validation examples illustrating the documented failure categories.

---

### figures/

Supporting plots used during validation, including calibration plots and confusion matrices.

## Notes

These files correspond to the final validated version of MTV-INT-RAD-003 after implementation of the closeout fixes requested during project review.

All reported metrics are generated directly from the validation scripts contained in the repository.