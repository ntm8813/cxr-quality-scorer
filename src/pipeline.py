# src/pipeline.py
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
from src.scorers.motion_blur_scorer import MotionBlurScorer
from src.scorers.artifact_scorer import ArtifactScorer
from src.explanation.explanation_module import ExplanationModule
from schemas.study_result import StudyResult
from src.validation.input_validator import validate_input
from schemas.rejected_result import RejectedResult


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
    return np.asarray(resized, dtype=np.float32) / 255.0


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


def _merge_duplicate_axes(axis_results):
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
                            "merged_from": [r.raw_metrics for r in results]
                        },
                        "rationale": " | ".join(
                            r.rationale or "" for r in results
                        ),
                    }
                )
            )

    return merged


def run_pipeline(
    image_path: str,
    config_path: str = "configs/v1.yaml",
) -> StudyResult | RejectedResult:

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    resize_to = (
        int(config.get("image", {}).get("resize", 1024))
        if config.get("image", {}).get("resize") is not None
        else None
    )

    image, metadata = load_image_any(image_path, resize_to=resize_to)

    # --- Fail-safe / input validation gate ---------------------------------
    # Modality / view-position / body-part filtering happens UPSTREAM in
    # MistiQRad at ingest time. This gate does NOT re-check clinical content;
    # it only verifies the array and metadata are structurally usable, so a
    # malformed or unexpected input fails safe instead of producing a
    # confident, meaningless score.
    #
    # expect_square is tied to whether resize_to is actually set. If resize
    # is disabled in config, a non-square image is expected and valid.
    validation = validate_input(
        image,
        metadata,
        expect_square=(resize_to is not None),
    )

    if not validation.is_valid:
        fallback_uid = "unknown"

        if metadata and metadata.get("study_uid"):
            fallback_uid = metadata["study_uid"]
        elif metadata and metadata.get("source_path"):
            fallback_uid = f"unidentified:{metadata['source_path']}"
        elif image_path:
            fallback_uid = f"unidentified:{image_path}"

        return RejectedResult(
            study_uid=fallback_uid,
            reason=validation.reason,
            failed_checks=validation.failed_checks,
            details=validation.details,
        )
    # -----------------------------------------------------------------------

    registry = ModelRegistry()

    unet_model = registry.load_lung_segmentation(device="cpu")
    blur_model = registry.load_blur_classifier(device="cpu")
    artifact_model = registry.load_artifact_classifier(device="cpu")

    scorers = [
        ExposureScorer(config),
        SharpnessScorer(config),
        MetadataScorer(config),
        RotationScorer(config, model=unet_model),
        CoverageScorer(config, model=unet_model),
        InspirationScorer(config, model=unet_model),
        MotionBlurScorer(config, model=blur_model),
        ArtifactScorer(config, model=artifact_model),
    ]

    axis_results = []

    for scorer in scorers:
        try:
            axis_results.append(scorer.score(image, metadata))
        except Exception as exc:
            print(f"[SCORER ERROR] {scorer.__class__.__name__}: {exc}")

    axis_results = _merge_duplicate_axes(axis_results)

    fusion = ScoreFusion(config_path)
    study = fusion.fuse(metadata.get("study_uid", "unknown"), axis_results)

    # Explanation toggle (safe + production control)
    explain = config.get("pipeline", {}).get("explain", True)

    if explain:
        try:
            study = ExplanationModule().enrich_study(study)
        except Exception as exc:
            print(f"[EXPLANATION ERROR] {exc}")

    return study