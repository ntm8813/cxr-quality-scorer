import pytest
import numpy as np
import torch
from monai.networks.nets import UNet
from src.scorers.coverage_scorer import CoverageScorer
from src.scorers.inspiration_scorer import InspirationScorer
from schemas.axis_result import AxisName

@pytest.fixture(scope="module")
def mock_unet_model():
    """Instantiates a miniature UNet for rapid unit-test execution."""
    model = UNet(
        spatial_dims=2, in_channels=1, out_channels=1,
        channels=(8, 16), strides=(2,), num_res_units=1
    )
    model.eval()
    return model

@pytest.fixture(scope="module")
def baseline_test_config():
    """Matches the structure of configs/v1.yaml to avoid KeyErrors."""
    return {
        "score_ranges": {"repeat_max": 40.0, "borderline_max": 70.0},
        "thresholds": {
            "coverage_margin_min_px": 10,
            "inspiration_min_ratio": 0.18
        }
    }

def test_coverage_scorer_execution(mock_unet_model, baseline_test_config):
    """Verifies CoverageScorer instantiates and runs inference."""
    scorer = CoverageScorer(baseline_test_config, model=mock_unet_model)
    mock_image = np.ones((1024, 1024), dtype=np.float32) * 0.5
    
    result = scorer.score(mock_image, {"study_uid": "test_coverage"})
    
    assert result.axis == AxisName.COVERAGE
    assert 0.0 <= result.score <= 1.0
    assert "min_margin_px" in result.raw_metrics

def test_inspiration_scorer_execution(mock_unet_model, baseline_test_config):
    """Verifies InspirationScorer tracks lung-to-image area ratio."""
    scorer = InspirationScorer(baseline_test_config, model=mock_unet_model)
    mock_image = np.ones((1024, 1024), dtype=np.float32) * 0.5
    
    result = scorer.score(mock_image, {"study_uid": "test_inspiration"})
    
    assert result.axis == AxisName.INSPIRATION
    assert 0.0 <= result.score <= 1.0
    assert "lung_area_ratio" in result.raw_metrics