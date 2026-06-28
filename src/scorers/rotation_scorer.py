# src/scorers/rotation_scorer.py
from __future__ import annotations

import numpy as np

from schemas.axis_result import AxisName, AxisResult
from src.scorers.base import BaseScorer
from src.scorers.quality_utils import (
    anatomical_foreground_mask,
    as_float_image,
    crop_to_mask,
    heuristic_lung_mask,
    infer_lung_mask,
    mask_anchored_orientation_deg,
    weighted_orientation_deg,
)


class RotationScorer(BaseScorer):
    """
    Rotation scorer.

    Primary method: estimates patient rotation from the LINE CONNECTING
    THE LEFT AND RIGHT LUNG CENTROIDS in the segmentation mask. A
    correctly positioned patient's two lungs sit at the same height, so
    this line is horizontal; rotation tilts it.

    Fallback method (used only if no usable lung mask can be obtained,
    e.g. self.model is None or segmentation fails): falls back to the
    original whole-image gradient-PCA method.

    This two-tier design exists because gradient-PCA on the raw image is
    frequently dominated by non-anatomical edges (collar lines, shoulders,
    equipment, image borders) and produced near-90-degree errors on real
    CXRs (see reports/WEEK4_STATUS_REPORT.md, Section 8 — model flagged
    280/300 studies as severe rotation when reviewers flagged 0).
    """

    MIN_MASK_PIXELS = 200  # below this, the mask is too small to trust

    def _estimate_orientation(self, image: np.ndarray):
        """
        Returns (signed_angle_deg, confidence, foreground_count, method_used)
        """
        img = as_float_image(image)

        # ── Tier 1: mask-anchored orientation ──────────────────────────
        if self.model is not None:
            lung_mask, detected = infer_lung_mask(self.model, img)
            if detected and lung_mask is not None and lung_mask.sum() >= self.MIN_MASK_PIXELS:
                signed_angle_deg, confidence, fg_count = mask_anchored_orientation_deg(
                    img, lung_mask
                )
                if fg_count >= self.MIN_MASK_PIXELS:
                    return signed_angle_deg, confidence, fg_count, "lung_mask"

        # ── Tier 2: fallback to whole-image gradient PCA ───────────────
        body_mask = anatomical_foreground_mask(img)
        crop_img, crop_mask = crop_to_mask(img, body_mask, pad_ratio=0.08)

        if crop_img.size == 0:
            return 0.0, 0.0, 0, "fallback_empty"

        signed_angle_deg, confidence, fg_count = weighted_orientation_deg(crop_img)

        if fg_count <= 0:
            return 0.0, 0.0, 0, "fallback_empty"

        return float(signed_angle_deg), float(confidence), int(fg_count), "fallback_gradient"

    def score(self, image: np.ndarray, metadata: dict) -> AxisResult:
        tolerance = float(self.config.get("thresholds", {}).get("rotation_tolerance_deg", 5.0))

        signed_angle_deg, confidence, fg_count, method = self._estimate_orientation(image)

        if fg_count <= 0:
            signed_angle_deg = 0.0
            confidence = 0.0

        # signed_angle_deg from mask_anchored_orientation_deg is already
        # "deviation from vertical" — 0 means upright, larger magnitude
        # means more rotated. No further 90-degree correction is needed
        # for the mask-anchored path.
        if method == "lung_mask":
                    # mask_anchored_orientation_deg now returns deviation from
                    # HORIZONTAL directly (inter-centroid line angle). 0 = upright.
                    angle_error_deg = abs(float(signed_angle_deg))
        else:
                    # Fallback gradient-PCA path: dominant axis angle from
                    # horizontal, where an upright body is near 90 deg from
                    # horizontal (vertical spine). Correct for that convention.
            orientation_deg = abs(float(signed_angle_deg))
            angle_error_deg = min(abs(orientation_deg - 90.0), abs(orientation_deg),)
        denom = max(1.0, tolerance * 1.35)
        confidence_boost = 0.70 + 0.30 * float(np.clip(confidence, 0.0, 1.0))
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
                "estimation_method": method,
            },
            rationale=(
                f"Method={method}, signed deviation={signed_angle_deg:.2f}°, "
                f"error={angle_error_deg:.2f}°, confidence={confidence:.3f}."
            ),
        )