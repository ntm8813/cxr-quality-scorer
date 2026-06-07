# tests/test_ml_scorers.py
import pytest
import numpy as np
import yaml
from src.ml.model_registry        import ModelRegistry
from src.scorers.motion_blur_scorer import MotionBlurScorer
from src.scorers.artifact_scorer    import ArtifactScorer
from schemas.axis_result            import AxisName


@pytest.fixture(scope="module")
def config():
    with open("configs/v1.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def registry():
    return ModelRegistry()


@pytest.fixture(scope="module")
def blur_scorer(config, registry):
    model = registry.load_blur_classifier(device="cpu")
    return MotionBlurScorer(config, model=model)


@pytest.fixture(scope="module")
def artifact_scorer(config, registry):
    model = registry.load_artifact_classifier(device="cpu")
    return ArtifactScorer(config, model=model)


def _meta():
    return {"study_uid": "test-w3-ml"}


def _rand_image():
    return np.random.rand(1024, 1024).astype(np.float32)


# ── MotionBlurScorer ──────────────────────────────────────────

def test_blur_scorer_returns_axis_result(blur_scorer):
    result = blur_scorer.score(_rand_image(), _meta())
    assert result is not None


def test_blur_scorer_axis_is_sharpness(blur_scorer):
    result = blur_scorer.score(_rand_image(), _meta())
    axis = result.axis if isinstance(result.axis, str) else result.axis.value
    assert axis == "sharpness"


def test_blur_scorer_score_in_range(blur_scorer):
    result = blur_scorer.score(_rand_image(), _meta())
    assert 0.0 <= result.score <= 1.0


def test_blur_scorer_flag_valid(blur_scorer):
    result = blur_scorer.score(_rand_image(), _meta())
    flag = result.flag if isinstance(result.flag, str) else result.flag.value
    assert flag in ("acceptable", "borderline", "repeat")


def test_blur_scorer_raw_metrics_has_probability(blur_scorer):
    result = blur_scorer.score(_rand_image(), _meta())
    assert "blur_probability" in result.raw_metrics
    assert 0.0 <= result.raw_metrics["blur_probability"] <= 1.0


def test_blur_scorer_rationale_is_string(blur_scorer):
    result = blur_scorer.score(_rand_image(), _meta())
    assert isinstance(result.rationale, str) and len(result.rationale) > 5


def test_blur_scorer_uniform_dark_image(blur_scorer):
    """Very dark image — should not crash."""
    image  = np.zeros((1024, 1024), dtype=np.float32)
    result = blur_scorer.score(image, _meta())
    assert 0.0 <= result.score <= 1.0


def test_blur_scorer_uniform_bright_image(blur_scorer):
    """Fully saturated image — should not crash."""
    image  = np.ones((1024, 1024), dtype=np.float32)
    result = blur_scorer.score(image, _meta())
    assert 0.0 <= result.score <= 1.0


# ── ArtifactScorer ────────────────────────────────────────────

def test_artifact_scorer_returns_axis_result(artifact_scorer):
    result = artifact_scorer.score(_rand_image(), _meta())
    assert result is not None


def test_artifact_scorer_axis_is_artifact(artifact_scorer):
    result = artifact_scorer.score(_rand_image(), _meta())
    axis = result.axis if isinstance(result.axis, str) else result.axis.value
    assert axis == "artifact"


def test_artifact_scorer_score_in_range(artifact_scorer):
    result = artifact_scorer.score(_rand_image(), _meta())
    assert 0.0 <= result.score <= 1.0


def test_artifact_scorer_flag_valid(artifact_scorer):
    result = artifact_scorer.score(_rand_image(), _meta())
    flag = result.flag if isinstance(result.flag, str) else result.flag.value
    assert flag in ("acceptable", "borderline", "repeat")


def test_artifact_scorer_raw_metrics_has_probability(artifact_scorer):
    result = artifact_scorer.score(_rand_image(), _meta())
    assert "artifact_probability" in result.raw_metrics
    assert 0.0 <= result.raw_metrics["artifact_probability"] <= 1.0


def test_artifact_scorer_rationale_is_string(artifact_scorer):
    result = artifact_scorer.score(_rand_image(), _meta())
    assert isinstance(result.rationale, str) and len(result.rationale) > 5


# ── Full pipeline integration ─────────────────────────────────

def test_pipeline_includes_artifact_axis():
    """
    End-to-end: run_pipeline on a sample PNG and verify
    artifact axis appears in axis_results.
    """
    from pathlib import Path
    from src.pipeline import run_pipeline

    sample_dir = Path("data/raw")
    pngs = list(sample_dir.rglob("*.png"))[:3]
    if not pngs:
        pytest.skip("No sample PNG images in data/raw/")

    for p in pngs:
        result = run_pipeline(str(p), explain=True)
        axes   = [
            (r.axis if isinstance(r.axis, str) else r.axis.value)
            for r in result.axis_results
        ]
        assert "artifact" in axes, \
            f"artifact axis missing from pipeline output. Got: {axes}"
        assert result.composite_score > 0
        flag = result.overall_flag
        assert flag in ("acceptable", "borderline", "repeat")

        # Explanation enrichment check
        summary = result.metadata_summary.get("summary_rationale", "")
        assert len(summary) > 10, "summary_rationale too short"