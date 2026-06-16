from __future__ import annotations

import numpy as np

from schemas.axis_result import AxisName, AxisResult
from src.scorers.base import BaseScorer
from src.scorers.quality_utils import (
    anatomical_foreground_mask,
    as_float_image,
    crop_to_mask,
    weighted_orientation_deg,
)


class RotationScorer(BaseScorer):
    """
    Rotation scorer based on ROI-restricted dominant-axis orientation.
    Returns a signed angle in degrees for pairwise comparisons.
    """

    def _estimate_orientation(self, image: np.ndarray):
        img = as_float_image(image)
        body_mask = anatomical_foreground_mask(img)
        crop_img, crop_mask = crop_to_mask(img, body_mask, pad_ratio=0.08)

        if crop_img.size == 0:
            return 0.0, 0.0, 0

        signed_angle_deg, confidence, fg_count = weighted_orientation_deg(crop_img)

        if fg_count <= 0:
            return 0.0, 0.0, 0

        return float(signed_angle_deg), float(confidence), int(fg_count)

    def score(self, image: np.ndarray, metadata: dict) -> AxisResult:
        tolerance = float(self.config.get("thresholds", {}).get("rotation_tolerance_deg", 5.0))

        signed_angle_deg, confidence, fg_count = self._estimate_orientation(image)

        if fg_count <= 0:
            signed_angle_deg = 0.0
            confidence = 0.0

        orientation_deg = abs(float(signed_angle_deg))

        angle_error_deg = min(
            abs(orientation_deg - 90.0),
            abs(orientation_deg)
        )

        denom = max(1.0, tolerance * 1.35)
        confidence_boost = 0.55 + 0.45 * float(np.clip(confidence, 0.0, 1.0))
        raw_score = float(
            np.clip(
                np.exp(-((angle_error_deg / denom) ** 1.15)) * confidence_boost,
                0.0,
                1.0,
            )
        )

        return AxisResult(
            study_uid=metadata.get("study_uid", "unknown"),
            axis=AxisName.ROTATION,
            score=raw_score,
            flag=self._flag_from_score(raw_score),
            raw_metrics={
                "rotation_angle_deg": round(float(angle_error_deg), 2),
                "rotation_angle_deg_signed": round(float(signed_angle_deg), 2),
                "orientation_confidence": round(float(confidence), 4),
                "foreground_pixel_count": int(fg_count),
                "tolerance_deg": float(tolerance),
            },
            rationale=(
                f"Signed orientation={signed_angle_deg:.2f}°, "
                f"deviation={angle_error_deg:.2f}°, confidence={confidence:.3f}."
            ),
        )