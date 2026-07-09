# src/scorers/artifact_scorer.py
from __future__ import annotations

import cv2
import numpy as np
import torch

from src.scorers.base import BaseScorer
from schemas.axis_result import AxisResult, AxisName

_INPUT_SIZE = 224


class ArtifactScorer(BaseScorer):
    """
    EfficientNet-B0 artifact classifier.

    NOTE:
    Outputs normalized quality score (1 - artifact probability).
    No post-hoc thresholding applied to avoid uncalibrated heuristics.
    """

    def _artifact_probability_threshold(self) -> float:
        return float(
            self.config.get("thresholds", {}).get(
                "artifact_probability_threshold",
                0.5,
            )
        )

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
            artifact_prob = float(torch.sigmoid(logit).item())

        raw_score = float(np.clip(1.0 - artifact_prob, 0.0, 1.0))
        threshold = self._artifact_probability_threshold()

        return AxisResult(
            study_uid=metadata.get("study_uid", "unknown"),
            axis=AxisName.ARTIFACT,
            score=raw_score,
            flag=self._flag_from_score(raw_score),
            raw_metrics={
                "artifact_probability": round(artifact_prob, 4),
                "artifact_probability_threshold": threshold,
                "artifact_likely": artifact_prob >= threshold,
                "source": "efficientnet_b0_ml",
            },
            rationale=(
                f"Artifact probability: {artifact_prob:.3f} "
                f"(threshold={threshold:.2f}). "
                + (
                    "Low likelihood of artifacts detected."
                    if artifact_prob < threshold
                    else "Possible artifacts detected."
                )
            ),
        )