import numpy as np
import cv2

from src.scorers.rotation_scorer import RotationScorer
from schemas.axis_result import AxisName


CONFIG = {
    "thresholds": {
        "rotation_tolerance_deg": 5
    },
    "score_ranges": {
        "repeat_max": 40,
        "borderline_max": 70
    }
}


def make_rotated_image(angle_deg):
    image = np.zeros((1024, 1024), dtype=np.float32)

    cv2.rectangle(
        image,
        (412, 312),
        (612, 712),
        1.0,
        -1
    )

    center = (512, 512)

    matrix = cv2.getRotationMatrix2D(
        center,
        angle_deg,
        1.0
    )

    rotated = cv2.warpAffine(
        image,
        matrix,
        (1024, 1024)
    )

    return rotated


def test_rotation_zero():

    scorer = RotationScorer(CONFIG)

    image = make_rotated_image(0)

    result = scorer.score(
        image,
        {"study_uid": "test"}
    )

    assert result.axis == AxisName.ROTATION
    assert result.score > 0.8


def test_rotation_mild():

    scorer = RotationScorer(CONFIG)

    image = make_rotated_image(5)

    result = scorer.score(
        image,
        {"study_uid": "test"}
    )

    assert result.axis == AxisName.ROTATION
    assert result.score > 0.4


def test_rotation_severe():

    scorer = RotationScorer(CONFIG)

    image = make_rotated_image(20)

    result = scorer.score(
        image,
        {"study_uid": "test"}
    )

    assert result.axis == AxisName.ROTATION
    assert result.score < 0.5