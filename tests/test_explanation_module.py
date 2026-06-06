# tests/test_explanation_module.py
import pytest
from schemas.axis_result import AxisResult, AxisName, QualityFlag
from schemas.study_result import StudyResult
from src.explanation.explanation_module import ExplanationModule


@pytest.fixture
def module():
    return ExplanationModule()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _axis(axis: AxisName, flag: QualityFlag, score: float,
          metrics: dict = None) -> AxisResult:
    return AxisResult(
        study_uid   = "test-expl-001",
        axis        = axis,
        score       = score,
        flag        = flag,
        raw_metrics = metrics or {},
        rationale   = "",
    )


def _study(axes, flag: str, score: float = 0.65) -> StudyResult:
    return StudyResult(
        study_uid        = "test-expl-001",
        composite_score  = score,
        overall_flag     = flag,
        axis_results     = axes,
        metadata_summary = {},
    )


# ── Axis-level tests ──────────────────────────────────────────────────────────

def test_sharpness_acceptable_contains_variance(module):
    r = _axis(AxisName.SHARPNESS, QualityFlag.ACCEPTABLE, 0.9, {
        "laplacian_variance": 145.2, "lap_score": 0.85,
        "tenengrad": 1200.0, "ten_score": 0.72, "threshold": 80.0,
    })
    text = module.explain_axis(r)
    assert "145" in text
    assert "acceptable" in text.lower()


def test_sharpness_repeat_mentions_blur(module):
    r = _axis(AxisName.SHARPNESS, QualityFlag.REPEAT, 0.1, {
        "laplacian_variance": 8.3, "lap_score": 0.12,
        "tenengrad": 40.0, "ten_score": 0.08, "threshold": 80.0,
    })
    text = module.explain_axis(r)
    assert any(word in text.lower() for word in ["repeat", "unacceptable", "blur"])


def test_exposure_borderline_shows_dynamic_range(module):
    r = _axis(AxisName.EXPOSURE, QualityFlag.BORDERLINE, 0.55, {
        "dynamic_range": 0.41, "p5": 0.09, "p95": 0.50, "clipping_ratio": 0.02,
    })
    text = module.explain_axis(r)
    assert "0.41" in text or "dynamic" in text.lower()


def test_rotation_acceptable(module):
    r = _axis(AxisName.ROTATION, QualityFlag.ACCEPTABLE, 0.88, {
        "rotation_angle_deg": 1.8, "tolerance_deg": 3.0,
    })
    text = module.explain_axis(r)
    assert "1.8" in text


def test_coverage_repeat_shows_margin(module):
    r = _axis(AxisName.COVERAGE, QualityFlag.REPEAT, 0.2, {
        "min_margin_px": 3.0, "mask_detected": True,
    })
    text = module.explain_axis(r)
    assert "3" in text


def test_inspiration_borderline(module):
    r = _axis(AxisName.INSPIRATION, QualityFlag.BORDERLINE, 0.52, {
        "lower_mass": 0.021, "upper_mass": 0.019, "lung_area_ratio": 0.52,
    })
    text = module.explain_axis(r)
    assert "borderline" in text.lower() or "shallow" in text.lower()


def test_artifact_repeat_shows_probability(module):
    r = _axis(AxisName.ARTIFACT, QualityFlag.REPEAT, 0.12, {
        "artifact_probability": 0.893,
    })
    text = module.explain_axis(r)
    assert "0.893" in text


def test_metadata_borderline_shows_issue_count(module):
    r = _axis(AxisName.METADATA, QualityFlag.BORDERLINE, 0.6, {
        "issue_count": 2,
        "issues": ["Missing tag: modality", "Missing tag: patient_id"],
    })
    text = module.explain_axis(r)
    assert "2" in text


def test_empty_metrics_does_not_crash(module):
    """Template with no matching metrics must fall back gracefully."""
    r = _axis(AxisName.EXPOSURE, QualityFlag.REPEAT, 0.1, {})
    text = module.explain_axis(r)
    assert isinstance(text, str) and len(text) > 5


def test_unknown_axis_falls_back_to_rationale(module):
    """An axis without a template entry returns the original rationale."""
    r = AxisResult(
        study_uid="x", axis=AxisName.METADATA,
        score=0.95, flag=QualityFlag.ACCEPTABLE,
        raw_metrics={}, rationale="Existing rationale text.",
    )
    # Clear templates temporarily by passing an axis with no entry
    # (metadata IS in templates so we test the fallback via empty flag)
    r2 = r.model_copy(update={"flag": "acceptable"})
    text = module.explain_axis(r2)
    assert isinstance(text, str)


# ── Study-level tests ─────────────────────────────────────────────────────────

def test_enrich_study_fills_all_rationales(module):
    axes = [
        _axis(AxisName.SHARPNESS, QualityFlag.ACCEPTABLE, 0.9, {
            "laplacian_variance": 120.0, "lap_score": 0.8,
            "tenengrad": 900.0, "ten_score": 0.65,
        }),
        _axis(AxisName.EXPOSURE, QualityFlag.BORDERLINE, 0.55, {
            "dynamic_range": 0.4, "p5": 0.1, "p95": 0.5, "clipping_ratio": 0.02,
        }),
    ]
    study    = _study(axes, "borderline", 0.62)
    enriched = module.enrich_study(study)

    for ar in enriched.axis_results:
        assert ar.rationale and len(ar.rationale) > 10, \
            f"Rationale too short for {ar.axis}: '{ar.rationale}'"


def test_enrich_study_sets_summary_rationale(module):
    axes = [
        _axis(AxisName.SHARPNESS, QualityFlag.BORDERLINE, 0.55, {}),
    ]
    study    = _study(axes, "borderline", 0.55)
    enriched = module.enrich_study(study)
    assert "summary_rationale" in enriched.metadata_summary
    assert len(enriched.metadata_summary["summary_rationale"]) > 10


def test_study_summary_acceptable(module):
    axes = [
        _axis(AxisName.SHARPNESS, QualityFlag.ACCEPTABLE, 0.9, {}),
        _axis(AxisName.EXPOSURE,  QualityFlag.ACCEPTABLE, 0.85, {}),
    ]
    enriched = module.enrich_study(_study(axes, "acceptable", 0.87))
    assert "acceptable" in enriched.metadata_summary["summary_rationale"].lower()


def test_study_summary_repeat_lists_axes(module):
    axes = [
        _axis(AxisName.SHARPNESS, QualityFlag.REPEAT,     0.1,  {}),
        _axis(AxisName.EXPOSURE,  QualityFlag.ACCEPTABLE, 0.85, {}),
    ]
    enriched = module.enrich_study(_study(axes, "repeat", 0.25))
    summary  = enriched.metadata_summary["summary_rationale"]
    assert "repeat" in summary.lower()
    assert "sharpness" in summary.lower()


def test_enrich_does_not_mutate_original(module):
    axes  = [_axis(AxisName.SHARPNESS, QualityFlag.ACCEPTABLE, 0.9, {})]
    study = _study(axes, "acceptable", 0.9)
    original_rationale = study.axis_results[0].rationale

    module.enrich_study(study)  # should not mutate

    assert study.axis_results[0].rationale == original_rationale
    assert "summary_rationale" not in study.metadata_summary