# CXR Quality Scorer – Project Handoff

## Project

MTV-INT-RAD-003  
Chest X-Ray Quality Assessment System

## Current Status

Completed:

- PNG ingestion
- DICOM ingestion with header metadata extraction
- Input validation and fail-safe rejection pipeline
- Lung segmentation
- Sharpness scoring
- Exposure scoring
- Rotation scoring
- Coverage scoring
- Inspiration scoring
- Artifact scoring
- Metadata scoring
- Configurable scoring thresholds
- Composite score fusion
- Streamlit deployment
- Batch inference
- Validation dashboard
- Study PDF reports
- Batch PDF reports
- Final validation package generation

## Validation

Validation cohort:

300 studies

Human inter-rater ceiling κ:

0.5248 (moderate)

Model overall κ:

0.2205 (fair)

Overall agreement:

58.3%

Spearman correlation:

ρ = -0.1942

p-value:

0.000718

## Main Entry Points

Single inference:

```python
from src.pipeline import run_pipeline
```

Generate study PDF:

```python
from reports.pdf_report_generator import generate_study_report
```

Generate batch PDF:

```bash
python -m src.scripts.generate_batch_pdf
```

Launch app:

```bash
streamlit run app.py
```

## Deliverables

- Streamlit application
- Validation dashboard
- Batch inference
- Study PDF reports
- Batch PDF reports
- Validation metrics
- Failure catalogue
- Final validation package

## Implementation Notes

The pipeline extracts modality, body part, view position, exposure index, deviation index, and KVP directly from DICOM headers whenever DICOM input is provided. Placeholder metadata is used only for non-DICOM image formats.

Quality-scoring thresholds are configurable through `configs/v1.yaml`, allowing deployment-specific tuning without modifying scorer source code.

Input validation executes immediately after image loading. Invalid studies return a `RejectedResult` instead of entering the scoring pipeline.

## Future Work

- CT extension
- Active learning
- PACS integration
- Multi-center validation
- Additional quality dimensions

---

# Input Contract & Upstream Filtering (List A item, confirmed with Medtatvaa CEO)

**Modality / view-position / body-part filtering happens upstream, not in this engine.** At ingest, MistiQRad already reads the DICOM header and filters to frontal chest films by tag:

- Modality: `CR` / `DX`
- BodyPartExamined / Study Description: Chest
- ViewPosition: `PA` / `AP`
- `FOR PRESENTATION` images only

This was confirmed directly by the CEO (project correspondence, June 2026):

> "Don't worry about these tags etc. We'll do these checks at our end so that the correct modality and position is passed to you."

Consequently, **this engine does not implement a view/modality classifier**, and it does not attempt to determine whether an image is a chest radiograph from pixel content. That responsibility belongs to the upstream ingestion pipeline, and this engine trusts that contract.

What this engine **is** responsible for is failing safely whenever an input does not structurally match what the scoring pipeline expects, regardless of why that occurred (upstream filtering bypassed, malformed input, decoding failure, unexpected file, etc.).

This validation is implemented in:

```text
src/validation/input_validator.py
```

and executes inside `run_pipeline()` immediately after image loading and before any scorer or ML model is run.

## What "expected format" means

After loading (and after the configured resize step, if enabled), an input is considered valid only if all of the following are true:

- 2D grayscale image, or a 3D image with exactly 1, 3, or 4 channels.
- Square dimensions matching `config.image.resize` (default 1024×1024) when resizing is enabled. If resizing is disabled, square dimensions are not required because the resize stage normally guarantees them.
- Image dimensions between 64 and 8192 pixels on each side.
- NumPy dtype is `float32`.
- Every pixel is finite (no NaN or Inf).
- Every pixel value lies within `[0.0, 1.0]`.
- Image is not blank (pixel standard deviation greater than `1e-4`).
- Metadata dictionary exists and contains non-empty values for:
  - `study_uid`
  - `modality`
  - `view_position`
  - `body_part`
  - `exposure_index`
  - `deviation_index`
  - `kvp`

## What happens on failure

If any validation check fails, `run_pipeline()` returns a `RejectedResult` (defined in `schemas/rejected_result.py`) instead of a `StudyResult`.

A rejected study is intentionally represented by a separate schema, ensuring invalid studies cannot accidentally be interpreted as successfully scored examinations.

`RejectedResult` contains:

- `study_uid`
- `reason`
- `failed_checks`
- `details`

where `details` includes diagnostic information such as observed array shape, dtype, and available metadata keys.

Both execution paths in `app.py` explicitly branch on `RejectedResult`.

- Single-file mode displays the validation failure and diagnostic information.
- Batch mode records the study with `status = rejected` while allowing the remainder of the batch to continue.

## What this does NOT cover

This validation gate is structural rather than semantic.

It does not detect:

- Structurally valid but incorrect imaging modalities.
- Clinically poor yet structurally valid chest radiographs.
- Adversarial inputs intentionally crafted to satisfy structural validation.