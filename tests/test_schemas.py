# tests/test_schemas.py
import pytest
from pydantic import ValidationError
from schemas.axis_result import AxisResult, AxisName, QualityFlag
from schemas.study_result import StudyResult

def test_axis_result_valid():
    """Verifies that a valid AxisResult passes Pydantic construction."""
    result = AxisResult(
        study_uid="1.2.3.4",
        axis=AxisName.EXPOSURE,
        score=0.85,
        flag=QualityFlag.ACCEPTABLE,
        raw_metrics={"mean_intensity": 128},
        rationale="Exposure looks well balanced"
    )
    assert result.score == 0.85
    assert result.flag == "acceptable"

def test_axis_result_score_out_of_range():
    """Verifies that an out-of-range score triggers a ValidationError."""
    with pytest.raises(ValidationError):
        AxisResult(
            study_uid="1.2.3.4",
            axis=AxisName.SHARPNESS,
            score=1.5,  # Invalid: Must be <= 1.0
            flag=QualityFlag.ACCEPTABLE
        )

def test_composite_score_valid():
    """Verifies that a complete StudyResult compiles correctly."""
    axis_data = [
        AxisResult(study_uid="1.2.3.4", axis=AxisName.EXPOSURE, score=0.9, flag=QualityFlag.ACCEPTABLE),
        AxisResult(study_uid="1.2.3.4", axis=AxisName.SHARPNESS, score=0.8, flag=QualityFlag.ACCEPTABLE)
    ]
    
    study = StudyResult(
        study_uid="1.2.3.4",
        composite_score=0.85,
        overall_flag=QualityFlag.ACCEPTABLE,
        axis_results=axis_data,
        metadata_summary={"modality": "CR"}
    )
    assert study.composite_score == 0.85
    assert len(study.axis_results) == 2