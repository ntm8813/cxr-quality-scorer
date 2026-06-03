import cv2
import numpy as np
import pytest

from src.scorers.sharpness_scorer import SharpnessScorer
from schemas.axis_result import AxisName


@pytest.fixture
def config():
    return {
        "thresholds": {
            "laplacian_variance": 80
        },
        "score_ranges": {
            "repeat_max": 40,
            "borderline_max": 70
        }
    }


def create_sharp_image():
    image = np.zeros((1024, 1024), dtype=np.float32)

    for i in range(0, 1024, 32):
        image[:, i:i+16] = 1.0

    return image


def create_blurred_image(kernel_size):
    image = create_sharp_image()

    blurred = cv2.GaussianBlur(
        image,
        (kernel_size, kernel_size),
        0
    )

    return blurred.astype(np.float32)


def test_sharpness_axis_name(config):

    scorer = SharpnessScorer(config)

    image = create_sharp_image()

    result = scorer.score(
        image,
        {"study_uid": "test"}
    )

    assert result.axis == AxisName.SHARPNESS


def test_blur_reduces_laplacian_variance(config):

    scorer = SharpnessScorer(config)

    sharp = create_sharp_image()
    blur1 = create_blurred_image(5)
    blur2 = create_blurred_image(15)

    sharp_result = scorer.score(sharp, {"study_uid": "test"})
    blur1_result = scorer.score(blur1, {"study_uid": "test"})
    blur2_result = scorer.score(blur2, {"study_uid": "test"})

    sharp_var = sharp_result.raw_metrics["laplacian_variance"]
    blur1_var = blur1_result.raw_metrics["laplacian_variance"]
    blur2_var = blur2_result.raw_metrics["laplacian_variance"]

    assert sharp_var > blur1_var > blur2_var


def test_blur_reduces_quality_score(config):

    scorer = SharpnessScorer(config)

    sharp = create_sharp_image()
    blur1 = create_blurred_image(5)
    blur2 = create_blurred_image(15)

    sharp_result = scorer.score(sharp, {"study_uid": "test"})
    blur1_result = scorer.score(blur1, {"study_uid": "test"})
    blur2_result = scorer.score(blur2, {"study_uid": "test"})

    assert sharp_result.score >= blur1_result.score >= blur2_result.score


def test_severe_blur_triggers_repeat_or_borderline(config):

    scorer = SharpnessScorer(config)

    image = create_blurred_image(31)

    result = scorer.score(
        image,
        {"study_uid": "test"}
    )

    assert result.flag in ["repeat", "borderline"]