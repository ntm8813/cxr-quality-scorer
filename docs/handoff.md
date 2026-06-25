# CXR Quality Scorer – Project Handoff

## Project

MTV-INT-RAD-003  
Chest X-Ray Quality Assessment System

## Current Status

Completed:

- DICOM ingestion
- PNG ingestion
- Lung segmentation
- Sharpness scoring
- Exposure scoring
- Rotation scoring
- Coverage scoring
- Inspiration scoring
- Artifact scoring
- Metadata scoring
- Composite score fusion
- Streamlit deployment
- Batch inference
- Validation dashboard
- PDF study reports
- PDF batch reports

## Validation

Human inter-rater ceiling κ:
0.5248

Model κ:
0.1119

Agreement:
43.7%

Spearman ρ:
-0.0425

Validation cohort:
300 studies

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

```python
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

## Future Work

- CT extension
- Active learning
- PACS integration
- Multi-center validation
- Additional quality dimensions