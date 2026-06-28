# tests/test_rotation_scorer.py
import numpy as np
import cv2
import pytest

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


# tests/test_rotation_scorer.py
# REPLACE ONLY this one helper function — leave everything else in the file as-is.

def make_synthetic_lung_mask(angle_deg: float, size: int = 1024) -> np.ndarray:
    """
    Builds a synthetic two-lung mask: two side-by-side ellipses at the
    SAME height, separated by a mediastinal gap, rotated by angle_deg
    around the image center.

    Matches real U-Net output geometry confirmed on actual CXRs: two
    lung fields with aspect ratio (bbox height / bbox width) in the
    0.65-1.30 range, NOT a single vertically-elongated blob. The
    earlier version of this fixture built tall narrow ellipses, which
    tested an orientation assumption that does not hold on real masks
    and masked the actual geometry bug.
    """
    mask = np.zeros((size, size), dtype=np.uint8)

    # Both lungs at the SAME y-center when angle_deg=0 -> horizontal
    # inter-centroid line, matching an upright patient.
    cv2.ellipse(mask, (size // 2 - 160, size // 2), (110, 220), 0, 0, 360, 1, -1)
    cv2.ellipse(mask, (size // 2 + 160, size // 2), (110, 220), 0, 0, 360, 1, -1)

    center = (size // 2, size // 2)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(mask, matrix, (size, size))

    return rotated.astype(bool)


class _FakeSegmentationModel:
    """
    Minimal stand-in for a loaded MONAI UNet. infer_lung_mask() expects
    something with .parameters() (for device lookup) and is callable
    returning a tensor-like prediction. We bypass that entirely by
    monkeypatching infer_lung_mask in the test instead — see below.
    """
    pass


def test_rotation_zero_no_model_fallback_path():
    """No model provided -> uses Tier 2 (gradient PCA) fallback."""
    scorer = RotationScorer(CONFIG)
    image = make_rotated_image(0)
    result = scorer.score(image, {"study_uid": "test"})

    assert result.axis == AxisName.ROTATION
    assert result.score > 0.8
    assert result.raw_metrics["estimation_method"] in (
        "fallback_gradient", "fallback_empty"
    )


def test_rotation_mild_no_model_fallback_path():
    scorer = RotationScorer(CONFIG)
    image = make_rotated_image(5)
    result = scorer.score(image, {"study_uid": "test"})

    assert result.axis == AxisName.ROTATION
    assert result.score > 0.4


def test_rotation_severe_no_model_fallback_path():
    scorer = RotationScorer(CONFIG)
    image = make_rotated_image(20)
    result = scorer.score(image, {"study_uid": "test"})

    assert result.axis == AxisName.ROTATION
    assert result.score < 0.5


def test_rotation_mask_anchored_upright_scores_high(monkeypatch):
    """
    With a model present and infer_lung_mask returning a clean upright
    two-lung mask, the scorer must use the lung_mask method and score high.
    """
    import src.scorers.rotation_scorer as rs_module

    upright_mask = make_synthetic_lung_mask(angle_deg=0)

    def fake_infer_lung_mask(model, image):
        return upright_mask, True

    monkeypatch.setattr(rs_module, "infer_lung_mask", fake_infer_lung_mask)

    scorer = RotationScorer(CONFIG, model=object())  # any non-None model
    image = np.zeros((1024, 1024), dtype=np.float32)
    result = scorer.score(image, {"study_uid": "test"})

    assert result.raw_metrics["estimation_method"] == "lung_mask"
    assert result.score > 0.75, (
        f"Upright synthetic lung mask scored too low: {result.score}. "
        f"angle_error_deg={result.raw_metrics['rotation_angle_deg']}"
    )
    assert result.flag == "acceptable"


def test_rotation_mask_anchored_rotated_scores_low(monkeypatch):
    """
    A lung mask rotated by 25 degrees must produce a low score and large
    angle_error_deg via the mask_anchored method.
    """
    import src.scorers.rotation_scorer as rs_module

    rotated_mask = make_synthetic_lung_mask(angle_deg=25)

    def fake_infer_lung_mask(model, image):
        return rotated_mask, True

    monkeypatch.setattr(rs_module, "infer_lung_mask", fake_infer_lung_mask)

    scorer = RotationScorer(CONFIG, model=object())
    image = np.zeros((1024, 1024), dtype=np.float32)
    result = scorer.score(image, {"study_uid": "test"})

    assert result.raw_metrics["estimation_method"] == "lung_mask"
    assert result.score < 0.5
    assert result.raw_metrics["rotation_angle_deg"] > 15


def test_rotation_mask_too_small_falls_back(monkeypatch):
    """
    If infer_lung_mask returns a mask with too few pixels, the scorer
    must fall back to Tier 2 rather than trust a noisy tiny mask.
    """
    import src.scorers.rotation_scorer as rs_module

    tiny_mask = np.zeros((1024, 1024), dtype=bool)
    tiny_mask[500:510, 500:510] = True  # only 100 pixels, below MIN_MASK_PIXELS

    def fake_infer_lung_mask(model, image):
        return tiny_mask, True

    monkeypatch.setattr(rs_module, "infer_lung_mask", fake_infer_lung_mask)

    scorer = RotationScorer(CONFIG, model=object())
    image = make_rotated_image(0)
    result = scorer.score(image, {"study_uid": "test"})

    assert result.raw_metrics["estimation_method"] != "lung_mask"


def test_rotation_no_mask_detected_falls_back(monkeypatch):
    """If infer_lung_mask reports detected=False, fall back to Tier 2."""
    import src.scorers.rotation_scorer as rs_module

    def fake_infer_lung_mask(model, image):
        return np.zeros((1024, 1024), dtype=bool), False

    monkeypatch.setattr(rs_module, "infer_lung_mask", fake_infer_lung_mask)

    scorer = RotationScorer(CONFIG, model=object())
    image = make_rotated_image(0)
    result = scorer.score(image, {"study_uid": "test"})

    assert result.raw_metrics["estimation_method"] != "lung_mask"


def test_regression_week4_bug_does_not_return(monkeypatch):
    """
    Regression guard for the Week 4 bug documented in
    reports/WEEK4_STATUS_REPORT.md Section 8: the scorer flagged
    280/300 real studies as severe rotation when reviewers flagged 0.

    This test locks in that an upright lung mask (the normal case for
    the vast majority of real studies) must NOT be scored as repeat.
    """
    import src.scorers.rotation_scorer as rs_module

    upright_mask = make_synthetic_lung_mask(angle_deg=0)

    def fake_infer_lung_mask(model, image):
        return upright_mask, True

    monkeypatch.setattr(rs_module, "infer_lung_mask", fake_infer_lung_mask)

    scorer = RotationScorer(CONFIG, model=object())
    image = np.zeros((1024, 1024), dtype=np.float32)
    result = scorer.score(image, {"study_uid": "test"})

    assert result.flag != "repeat", (
        "REGRESSION: an upright lung mask was flagged as 'repeat'. "
        "This is the exact Week 4 failure mode — orientation convention "
        "mismatch causing ~93% false repeat rate on real studies."
    )