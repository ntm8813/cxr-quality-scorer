import numpy as np
from src.scorers.coverage_scorer import CoverageScorer
from src.scorers.inspiration_scorer import InspirationScorer


class DummyConfig:
    def __init__(self):
        self.config = {
            "score_ranges": {"repeat_max": 40.0, "borderline_max": 70.0},
            "thresholds": {"coverage_margin_min_px": 10}
        }


def test_scorer_raw_metrics_contract():
    cfg = {
        "score_ranges": {"repeat_max": 40.0, "borderline_max": 70.0},
        "thresholds": {"coverage_margin_min_px": 10}
    }

    img = np.random.rand(1024, 1024).astype(np.float32)

    # ---- Coverage contract check ----
    cov = CoverageScorer(cfg, model=None)
    r1 = cov.score(img, {"study_uid": "test"})

    assert "mask_detected" in r1.raw_metrics
    assert isinstance(r1.raw_metrics["mask_detected"], (bool, np.bool_))

    assert "min_margin_px" in r1.raw_metrics
    assert isinstance(r1.raw_metrics["min_margin_px"], (float, int, np.floating))


    # ---- Inspiration contract check ----
    insp = InspirationScorer(cfg, model=None)
    r2 = insp.score(img, {"study_uid": "test"})

    assert "lung_area_ratio" in r2.raw_metrics
    assert isinstance(r2.raw_metrics["lung_area_ratio"], (float, int, np.floating))

    assert "upper_mass" in r2.raw_metrics
    assert "lower_mass" in r2.raw_metrics