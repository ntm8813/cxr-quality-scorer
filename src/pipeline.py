from __future__ import annotations

import os
import yaml
import numpy as np
from PIL import Image
from collections import defaultdict

from src.fusion.score_fusion import ScoreFusion
from src.ml.model_registry import ModelRegistry

from src.scorers.coverage_scorer import CoverageScorer
from src.scorers.exposure_scorer import ExposureScorer
from src.scorers.inspiration_scorer import InspirationScorer
from src.scorers.metadata_scorer import MetadataScorer
from src.scorers.rotation_scorer import RotationScorer
from src.scorers.sharpness_scorer import SharpnessScorer
from src.explanation.explanation_module import ExplanationModule

from src.scorers.motion_blur_scorer import MotionBlurScorer
from src.scorers.artifact_scorer import ArtifactScorer

from schemas.study_result import StudyResult


# ─────────────────────────────────────────────────────────────
# Image utilities
# ─────────────────────────────────────────────────────────────

def _resize_float_image(image: np.ndarray, size: int) -> np.ndarray:
    if size is None:
        return image.astype(np.float32)
    if image.shape[0] == size and image.shape[1] == size:
        return image.astype(np.float32)

    resized = Image.fromarray(
        (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    )
    resized = resized.resize(
        (size, size),
        resample=Image.Resampling.LANCZOS
        if hasattr(Image, "Resampling")
        else Image.LANCZOS,
    )
    return (np.asarray(resized, dtype=np.float32) / 255.0)


def load_image_any(path: str, resize_to: int | None = None):
    if path.lower().endswith((".png", ".jpg", ".jpeg")):
        img = Image.open(path).convert("L")

        if resize_to is not None:
            img = img.resize(
                (resize_to, resize_to),
                resample=Image.Resampling.LANCZOS
                if hasattr(Image, "Resampling")
                else Image.LANCZOS,
            )

        image = np.array(img, dtype=np.float32) / 255.0
        metadata = {"study_uid": os.path.basename(path), "file_path": path}
        return image, metadata

    if path.lower().endswith(".dcm"):
        from src.io.dicom_reader import DICOMReader

        image, metadata = DICOMReader().load(path)
        if resize_to is not None:
            image = _resize_float_image(image, resize_to)
        return image, metadata

    raise ValueError(f"Unsupported file format: {path}")


# ─────────────────────────────────────────────────────────────
# FIX: resolve duplicate axis outputs (SHARPNESS collision)
# ─────────────────────────────────────────────────────────────

def _merge_duplicate_axes(axis_results):
    """
    If multiple scorers emit the same axis (e.g. SHARPNESS from:
    SharpnessScorer + MotionBlurScorer), merge by averaging scores.
    """
    grouped = defaultdict(list)

    for r in axis_results:
        key = r.axis.value if hasattr(r.axis, "value") else str(r.axis)
        grouped[key].append(r)

    merged = []

    for key, results in grouped.items():
        if len(results) == 1:
            merged.append(results[0])
        else:
            avg_score = float(np.mean([r.score for r in results]))
            base = results[0]

            merged.append(
                base.model_copy(
                    update={
                        "score": avg_score,
                        "raw_metrics": {
                            "merged_from": [
                                r.raw_metrics for r in results
                            ]
                        },
                        "rationale": " | ".join(
                            r.rationale or "" for r in results
                        ),
                    }
                )
            )

    return merged


# ─────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────

def run_pipeline(
    image_path: str,
    config_path: str = "configs/v1.yaml",
    explain: bool = False,
) -> StudyResult:

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    resize_to = int(config.get("image", {}).get("resize", 1024))

    image, metadata = load_image_any(image_path, resize_to=resize_to)

    # ── Models
    registry = ModelRegistry()
    unet_model = registry.load_lung_segmentation(device="cpu")
    blur_model = registry.load_blur_classifier(device="cpu")
    artifact_model = registry.load_artifact_classifier(device="cpu")

    # ── Scorers
    scorers = [
        ExposureScorer(config),
        SharpnessScorer(config),
        MetadataScorer(config),
        RotationScorer(config),

        CoverageScorer(config, model=unet_model),
        InspirationScorer(config, model=unet_model),

        MotionBlurScorer(config, model=blur_model),
        ArtifactScorer(config, model=artifact_model),
    ]

    # ── Run scoring
    axis_results = []

    for scorer in scorers:
        try:
            axis_results.append(scorer.score(image, metadata))
        except Exception as exc:
            print(f"[SCORER ERROR] {scorer.__class__.__name__}: {exc}")

    # ── FIX: merge duplicate axes BEFORE fusion
    axis_results = _merge_duplicate_axes(axis_results)

    # ── Fusion
    fusion = ScoreFusion(config_path)

    study = fusion.fuse(
        metadata.get("study_uid", "unknown"),
        axis_results,
    )

    # ── Explanation enrichment (REQUIRED FOR TESTS) ──
    try:
        explainer = ExplanationModule()
        study = explainer.enrich_study(study)
    except Exception as e:
        print(f"[EXPLANATION ERROR] {e}")

    return study