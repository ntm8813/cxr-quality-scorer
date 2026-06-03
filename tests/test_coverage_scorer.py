import numpy as np
import torch

from src.scorers.coverage_scorer import CoverageScorer
from schemas.axis_result import AxisName


class DummyModelGood:
    def __call__(self, x):
        mask = torch.zeros((1, 1, 1024, 1024))
        mask[:, :, 50:974, 50:974] = 10.0
        return mask


class DummyModelBad:
    def __call__(self, x):
        mask = torch.zeros((1, 1, 1024, 1024))
        mask[:, :, 2:1022, 2:1022] = 10.0
        return mask


class DummyModelEmpty:
    def __call__(self, x):
        return torch.full((1, 1, 1024, 1024), -10.0)


CONFIG = {
    "thresholds": {
        "coverage_margin_min_px": 10
    },
    "score_ranges": {
        "repeat_max": 40,
        "borderline_max": 70
    }
}


def test_coverage_good_margin():

    scorer = CoverageScorer(CONFIG, DummyModelGood())

    image = np.random.rand(1024, 1024).astype(np.float32)

    result = scorer.score(
        image,
        {"study_uid": "test"}
    )

    assert result.axis == AxisName.COVERAGE
    assert result.score == 1.0
    assert result.raw_metrics["mask_detected"] is True


def test_coverage_truncated_margin():

    scorer = CoverageScorer(CONFIG, DummyModelBad())

    image = np.random.rand(1024, 1024).astype(np.float32)

    result = scorer.score(
        image,
        {"study_uid": "test"}
    )

    assert result.axis == AxisName.COVERAGE
    assert result.score < 1.0
    assert result.raw_metrics["min_margin_px"] < 10


def test_coverage_empty_mask():

    scorer = CoverageScorer(CONFIG, DummyModelEmpty())

    image = np.random.rand(1024, 1024).astype(np.float32)

    result = scorer.score(
        image,
        {"study_uid": "test"}
    )

    assert result.axis == AxisName.COVERAGE
    assert result.score == 0.0
    assert result.raw_metrics["mask_detected"] is False