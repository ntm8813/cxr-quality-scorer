import pytest
from schemas.axis_result import AxisResult, QualityFlag, AxisName
from schemas.study_result import CompositeScore


def test_axis_result_valid():
    result = AxisResult(
        study_uid="1.2.3.4.5",
        axis=AxisName.SHARPNESS,
        score=0.85,
        flag=QualityFlag.ACCEPTABLE,
        raw_metrics={"laplacian_variance": 120.5},
        rationale="Sharpness is within acceptable range."
    )
    assert result.score == 0.85
    assert result.flag == "acceptable"


def test_axis_result_score_out_of_range():
    with pytest.raises(Exception):
        AxisResult(
            study_uid="1.2.3.4.5",
            axis=AxisName.SHARPNESS,
            score=1.5,  # invalid — must be 0.0 to 1.0
            flag=QualityFlag.ACCEPTABLE,
            raw_metrics={}
        )


def test_composite_score_valid():
    axis = AxisResult(
        study_uid="1.2.3.4.5",
        axis=AxisName.EXPOSURE,
        score=0.6,
        flag=QualityFlag.BORDERLINE,
        raw_metrics={"mean_pixel": 0.45}
    )
    study = CompositeScore(
        study_uid="1.2.3.4.5",
        composite_score=58.0,
        overall_flag=QualityFlag.BORDERLINE,
        axis_results=[axis],
        summary_rationale="Exposure is borderline."
    )
    assert study.composite_score == 58.0
    assert len(study.axis_results) == 1