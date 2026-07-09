# CXR Quality Scorer

Automated chest radiograph (CXR) image quality assessment pipeline developed for **MTV-INT-RAD-003**.

The project evaluates frontal chest X-ray quality across multiple image-quality dimensions and produces an overall quality assessment intended to support radiographers and quality-control workflows. The system accepts PNG images and DICOM studies, generates per-axis quality scores, fuses them into a composite score, and provides both interactive and batch reporting.

> **Important:** This software is intended as a quality-control support tool only. It is **not** a diagnostic system and must not be used as the sole basis for clinical decisions.

---

# Features

* PNG image support
* DICOM image support with metadata extraction
* Input validation and fail-safe rejection
* Lung segmentation
* Sharpness scoring
* Exposure scoring
* Rotation scoring
* Coverage scoring
* Inspiration scoring
* Artifact scoring
* Metadata scoring
* Composite score fusion
* Streamlit interface
* Batch inference
* Validation dashboard
* Study PDF generation
* Batch PDF generation
* Validation package generation

---

# Repository Overview

Typical repository structure:

```text
.
├── app.py
├── configs/
├── data/
├── reports/
├── schemas/
├── src/
├── tests/
├── environment.yml
├── README.md
└── best_lung_unet.pth.dvc
```

Main components:

| Folder     | Purpose                                 |
| ---------- | --------------------------------------- |
| `src/`     | Core scoring pipeline                   |
| `configs/` | Runtime configuration                   |
| `tests/`   | Unit and integration tests              |
| `reports/` | Validation outputs, figures and reports |
| `schemas/` | Pydantic data models                    |
| `data/`    | Local datasets                          |

---

# Requirements

* Python 3.10
* Conda
* Git
* DVC (for model weights)

---

# Installation

Clone the repository.

```bash
git clone <repository-url>
cd cxr-quality-scorer
```

Create the Conda environment.

```bash
conda env create -f environment.yml
```

Activate it.

```bash
conda activate cxr
```

---

# Model Weights (DVC)

The lung segmentation model is tracked using DVC rather than Git.

The repository contains a DVC pointer:

```
best_lung_unet.pth.dvc
```

The corresponding model file is:

```
best_lung_unet.pth
```

Configure the DVC remote.

```bash
dvc remote list
```

Configured remote:

```
gdrive_remote
```

Google Drive remote:

```
gdrive://157pOv3H4oIyrZg9TLZxks9Sb4vz1uCmA
```

Pull the model weights.

```bash
dvc pull
```

If you do not have access to the configured DVC remote, place the required model weights manually in the expected project location before running the pipeline.

---

# Running the Streamlit Application

```bash
streamlit run app.py
```

The application supports:

* single-study inference
* batch inference
* PDF report generation
* rejected-input reporting

---

# Running the Pipeline

Example:

```python
from src.pipeline import run_pipeline

result = run_pipeline("path/to/image.png")
```

For DICOM input:

```python
result = run_pipeline("path/to/study.dcm")
```

---

# Batch Processing

Generate batch reports with

```bash
python -m src.scripts.generate_batch_pdf
```

---

# Running Tests

Run the full test suite.

```bash
pytest
```

Generate coverage.

```bash
pytest --cov
```

The repository currently contains unit and integration tests covering:

* DICOM reader
* input validation
* scorer contracts
* sharpness scorer
* exposure scorer
* coverage scorer
* rotation scorer
* metadata scorer
* machine-learning scorers
* explanation module
* schema validation
* pipeline integration

---

# Configuration

Runtime configuration is stored in:

```
configs/v1.yaml
```

Configuration includes:

* scoring thresholds
* scorer weights
* model paths
* resize configuration
* confidence thresholds
* fusion parameters

Thresholds for exposure, coverage, artifact detection and motion-blur detection are configurable through this file.

---

# Input Validation

Every input passes through structural validation before scoring.

Validation checks include:

* image dimensions
* supported channel count
* dtype validation
* NaN / Inf detection
* pixel value range
* blank image detection
* required metadata fields

Invalid inputs return a `RejectedResult` rather than entering the scoring pipeline.

---

# DICOM Metadata

For DICOM studies, metadata is read directly from the DICOM header using `pydicom`.

Fields include:

* Study UID
* Modality
* Body Part
* View Position
* Exposure Index
* Deviation Index
* KVP

PNG images use placeholder metadata where DICOM metadata is unavailable.

---

# Validation Summary

Validation cohort:

* 300 chest radiographs

Human inter-rater ceiling:

* κ = **0.5248**

Model performance:

* Overall κ = **0.2205**
* Overall agreement = **58.3%**
* Spearman correlation = **−0.1942**
* Validation studies = **300**

CPU inference latency:

| Metric  |    Value |
| ------- | -------: |
| Mean    | 1.9588 s |
| Median  | 1.9520 s |
| P95     | 2.1535 s |
| Minimum | 1.7843 s |
| Maximum | 2.1895 s |

The reported latency represents complete CPU inference, including image loading, preprocessing, all quality scorers, score fusion and explanation generation.

---

# Validation Package

Validation outputs are available under the `reports/` directory.

Included artifacts:

* validation results
* inter-rater agreement
* latency benchmark
* failure catalogue
* calibration plots
* confusion matrices
* reporting fixes
* validation figures

---

# Limitations

This implementation intentionally does not perform:

* VOI-LUT processing
* windowing
* rescale slope/intercept normalization
* photometric interpretation handling
* 12–16 bit pixel normalization
* modality classification from image pixels

Upstream systems are expected to provide correctly filtered frontal chest radiographs.

The input validation layer performs structural validation but does not determine whether an image is clinically a chest radiograph.

---

# Future Work

Potential extensions include:

* CT quality assessment
* PACS integration
* multi-centre validation
* active learning
* additional quality dimensions
* improved artifact detection
* improved inspiration scoring

---

# Troubleshooting

### Missing model weights

Ensure the required model weights have been downloaded using DVC or copied manually into the expected location.

---

### Streamlit does not start

Verify that the Conda environment has been activated.

```bash
conda activate cxr
```

---

### DVC pull fails

Confirm that:

* DVC is installed
* the remote is configured
* you have access permissions to the configured Google Drive remote

---

### Import errors

Recreate the Conda environment.

```bash
conda env remove -n cxr
conda env create -f environment.yml
conda activate cxr
```

---

# Disclaimer

This software is intended for research and quality-control support.

It is **not** a medical device and must not be used as the sole basis for clinical decision-making.

---

# Acknowledgements

Developed as part of **MTV-INT-RAD-003** during the MedTatvaa internship.

Validation was performed against a manually reviewed 300-study chest radiograph dataset using independent reviewer consensus.
