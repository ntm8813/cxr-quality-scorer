# src/scorers/motion_blur_scorer.py
from __future__ import annotations

import numpy as np
import torch
import cv2

from src.scorers.base import BaseScorer
from schemas.axis_result import AxisResult, AxisName

_INPUT_SIZE = 224


class MotionBlurScorer(BaseScorer):
    """
    EfficientNet-B0 motion blur classifier.

    NOTE:
    Emits AxisName.SHARPNESS intentionally to align with
    SharpnessScorer for later fusion-level aggregation.
    """

    def score(self, image: np.ndarray, metadata: dict) -> AxisResult:

        img_224 = cv2.resize(
            image.astype(np.float32),
            (_INPUT_SIZE, _INPUT_SIZE),
            interpolation=cv2.INTER_LINEAR,
        )

        tensor = torch.from_numpy(
            np.stack([img_224, img_224, img_224], axis=0)
        ).unsqueeze(0).float()

        with torch.no_grad():
            logit = self.model(tensor).squeeze()
            blur_prob = float(torch.sigmoid(logit).item())

        raw_score = float(np.clip(1.0 - blur_prob, 0.0, 1.0))

        return AxisResult(
            study_uid=metadata.get("study_uid", "unknown"),
            axis=AxisName.SHARPNESS,
            score=raw_score,
            flag=self._flag_from_score(raw_score),
            raw_metrics={
                "blur_probability": round(blur_prob, 4),
                "source": "efficientnet_b0_ml"
            },
            rationale=(
                f"ML blur classifier probability: {blur_prob:.3f}. "
                + (
                    "No significant motion blur detected."
                    if blur_prob < 0.4
                    else "Motion blur detected."
                )
            ),
        )