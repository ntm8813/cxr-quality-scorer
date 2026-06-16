# CXR Quality Scorer – Week 4 Validation Status Report

**Project:** MTV-INT-RAD-003
**Version:** v1.0
**Date:** June 2026

---

# 1. Project Objective

The objective of this project is to develop an automated Chest X-Ray (CXR) image quality assessment pipeline capable of evaluating radiographs across seven clinically relevant quality dimensions:

1. Sharpness
2. Exposure
3. Rotation
4. Coverage
5. Inspiration
6. Artifact Presence
7. Metadata Completeness

The system generates:

* Per-axis quality scores
* Per-axis quality flags
* Composite quality score
* Overall quality classification

Overall classifications:

| Flag       | Meaning                                        |
| ---------- | ---------------------------------------------- |
| Acceptable | Diagnostic quality                             |
| Borderline | Usable with deficiencies                       |
| Repeat     | Non-diagnostic; repeat acquisition recommended |

---

# 2. Validation Dataset Construction

## Source Dataset

NIH ChestX-ray Dataset

Validation subset created from:

```text
data/raw/nih_subset/
```

## Gold Standard Creation

A fixed validation set of:

```text
300 chest radiographs
```

was randomly sampled using a reproducible random seed.

Manifest:

```text
manifests/gold_standard_manifest.csv
```

contains the frozen study list used for all validation activities.

This ensures:

* Reproducibility
* Consistent reviewer ordering
* Identical evaluation set across reviewers

---

# 3. Human Review Process

Two independent reviewers evaluated all 300 studies.

Files:

```text
data/ratings/reviewer_1.csv
data/ratings/reviewer_2.csv
```

Each reviewer assigned:

| Axis          | Scale |
| ------------- | ----- |
| Sharpness     | 1–3   |
| Exposure      | 1–3   |
| Rotation      | 1–3   |
| Coverage      | 1–3   |
| Inspiration   | 1–3   |
| Artifact      | 1–3   |
| Metadata      | 1–3   |
| Global Rating | 1–3   |

Interpretation:

| Value | Meaning    |
| ----- | ---------- |
| 1     | Acceptable |
| 2     | Borderline |
| 3     | Repeat     |

Total ratings collected:

```text
300 studies × 2 reviewers
= 600 complete evaluations
```

---

# 4. Inter-Rater Agreement

Consensus generation was performed using:

```text
src/analysis/compute_kappa.py
```

Result:

| Axis          | Cohen's κ | Agreement |
| ------------- | --------- | --------- |
| Sharpness     | 0.2605    | 79.3%     |
| Exposure      | 0.5841    | 75.7%     |
| Rotation      | 0.3019    | 84.0%     |
| Coverage      | 0.3280    | 80.0%     |
| Inspiration   | -0.0494   | 64.7%     |
| Artifact      | 0.4530    | 74.0%     |
| Metadata      | -0.0071   | 98.3%     |
| Global Rating | 0.5248    | 72.0%     |

Observations:

* Exposure demonstrated the strongest reviewer agreement.
* Global rating achieved moderate agreement.
* Inspiration exhibited poor agreement, indicating subjective interpretation.
* Metadata agreement is not meaningful because PNG images do not contain DICOM metadata.

Consensus file generated:

```text
data/gold_standard_consensus.csv
```

Studies:

```text
300
```

---

# 5. Model Validation Dataset

Pipeline executed on all gold-standard studies.

Command:

```bash
python -m src.scripts.run_on_gold_standard
```

Output:

```text
data/predictions/model_v1.csv
```

Predictions generated:

```text
300 / 300
```

Failures:

```text
0
```

---

# 6. Model Output Distribution

Overall Flag Distribution:

| Class      | Count |
| ---------- | ----- |
| Acceptable | 35    |
| Borderline | 265   |
| Repeat     | 0     |

Composite Score Statistics:

| Metric | Value  |
| ------ | ------ |
| Count  | 300    |
| Mean   | 0.6077 |
| Std    | 0.0633 |
| Min    | 0.4062 |
| Median | 0.5954 |
| Max    | 0.8266 |

Observations:

* Model strongly favors the Borderline class.
* Very few studies are classified as Acceptable.
* Composite scores occupy a narrow operating range.

---

# 7. Validation Against Human Consensus

Performed using:

```text
src/analysis/compute_validation.py
```

## Per-Axis Agreement

| Axis        | Model κ | Agreement |
| ----------- | ------- | --------- |
| Sharpness   | -0.0872 | 57.0%     |
| Exposure    | 0.0780  | 61.3%     |
| Rotation    | 0.0026  | 1.7%      |
| Coverage    | 0.0364  | 74.3%     |
| Inspiration | -0.0087 | 35.7%     |
| Artifact    | 0.0447  | 10.7%     |
| Metadata    | 0.0000  | 98.3%     |

## Overall Classification

| Metric           | Value  |
| ---------------- | ------ |
| Weighted Cohen κ | 0.0496 |
| Agreement        | 39.0%  |

## Correlation

Spearman Correlation:

```text
ρ = -0.0368
```

Interpretation:

Higher composite scores correspond weakly and inconsistently with reviewer-assigned quality ratings.

---

# 8. Failure Analysis

Generated using:

```text
src/analysis/error_analysis.py
```

Studies analysed:

```text
300
```

Total disagreements:

```text
1083
```

---

## Major Failure Modes

### Rotation Scorer

Reviewer Consensus:

| Rating     | Count |
| ---------- | ----- |
| Acceptable | 242   |
| Borderline | 58    |
| Repeat     | 0     |

Model Predictions:

| Flag       | Count |
| ---------- | ----- |
| Acceptable | 2     |
| Borderline | 18    |
| Repeat     | 280   |

Finding:

The rotation scorer systematically classifies nearly all studies as severe rotation.

This is the largest single source of disagreement.

---

### Artifact Scorer

Observed behavior:

* Excessive repeat classifications.
* Poor alignment with reviewer judgement.
* Agreement only 10.7%.

Potential causes:

* Domain shift between training and validation images.
* Calibration mismatch.
* Threshold instability.

---

### Inspiration Scorer

Observed behavior:

* Weak agreement with reviewers.
* Human reviewers also exhibited poor agreement.

Interpretation:

This axis may inherently possess higher subjectivity.

---

### Metadata Scorer

Initial implementation assumed DICOM availability.

Validation images were PNG files.

Scorer was modified to:

```text
Return acceptable when metadata is unavailable.
```

Result:

```text
100% acceptable predictions
```

This axis cannot be meaningfully evaluated on PNG exports.

---

# 9. Current System Assessment

## Strengths

* End-to-end pipeline operational.
* Reproducible evaluation framework.
* Automated scoring across seven dimensions.
* Human gold-standard dataset created.
* Inter-rater agreement quantified.
* Failure catalogue generated.
* Validation workflow fully automated.

## Weaknesses

* Low agreement with expert consensus.
* Rotation estimation unreliable.
* Artifact detection over-sensitive.
* Composite score calibration weak.
* Limited discrimination between acceptable and borderline studies.

---

# 10. Deliverables Completed

| Deliverable                | Status   |
| -------------------------- | -------- |
| Gold-standard dataset      | Complete |
| Reviewer 1 annotations     | Complete |
| Reviewer 2 annotations     | Complete |
| Consensus generation       | Complete |
| Validation execution       | Complete |
| Kappa analysis             | Complete |
| Calibration analysis       | Complete |
| Failure mode analysis      | Complete |
| Validation figures         | Complete |
| Week 4 evaluation pipeline | Complete |

---

# 11. Key Conclusion

The project successfully demonstrates a complete automated chest radiograph quality assessment workflow including dataset construction, annotation management, consensus generation, validation, calibration analysis, and failure-mode investigation.

While current agreement with human reviewers remains low, the framework provides a reproducible foundation for future scorer refinement and model calibration.

The principal outcome of Week 4 is therefore not clinical deployment readiness, but the successful establishment of a complete validation and benchmarking framework capable of supporting subsequent model improvement efforts.
