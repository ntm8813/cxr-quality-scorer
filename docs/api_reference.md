# API Reference

## run_pipeline

```python
from src.pipeline import run_pipeline
```

### Input

```python
run_pipeline(path_to_image)
```

Supported:

- DICOM (.dcm)
- PNG (.png)
- JPG (.jpg)

### Returns

StudyResult

Fields:

- study_uid
- overall_flag
- composite_score
- axis_results
- metadata_summary

---

## generate_study_report

```python
from reports.pdf_report_generator import generate_study_report
```

### Example

```python
result = run_pipeline(image_path)

generate_study_report(
    result=result,
    output_path="report.pdf",
    image_path=image_path,
)
```

---

## generate_batch_report

```python
from reports.pdf_report_generator import generate_batch_report
```

### Example

```python
generate_batch_report(
    predictions_csv,
    kappa_json,
    calibration_png,
    failure_md,
    output_path,
)
```