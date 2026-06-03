import torch
import numpy as np

from src.scorers.base import BaseScorer
from schemas.axis_result import AxisResult, AxisName


class CoverageScorer(BaseScorer):
    """
    Evaluates anatomical lung coverage.

    Uses a segmentation mask to determine whether lung anatomy
    is truncated near the image borders.
    """

    def __init__(self, config: dict, model):
        super().__init__(config)
        self.model = model

    def score(self, image: np.ndarray, metadata: dict) -> AxisResult:

        height, width = image.shape[:2]

        tensor = (
            torch.from_numpy(image)
            .unsqueeze(0)
            .unsqueeze(0)
            .float()
        )

        with torch.no_grad():
            mask_logits = self.model(tensor)
            mask = torch.sigmoid(mask_logits).squeeze().cpu().numpy()

        binary_mask = (mask > 0.5).astype(np.uint8)

        rows = np.any(binary_mask, axis=1)
        cols = np.any(binary_mask, axis=0)

        row_idx = np.where(rows)[0]
        col_idx = np.where(cols)[0]

        # Empty mask protection
        if len(row_idx) == 0 or len(col_idx) == 0:

            raw_score = 0.0

            return AxisResult(
                study_uid=metadata.get("study_uid", "unknown"),
                axis=AxisName.COVERAGE,
                score=raw_score,
                flag=self._flag_from_score(raw_score),
                raw_metrics={
                    "min_margin_px": 0,
                    "mask_detected": False
                },
                rationale="No lung mask detected."
            )

        ymin, ymax = row_idx[[0, -1]]
        xmin, xmax = col_idx[[0, -1]]

        min_margin = min(
            xmin,
            width - xmax - 1,
            ymin,
            height - ymax - 1
        )

        thresholds = self.config.get("thresholds", {})

        target_margin = thresholds.get(
            "coverage_margin_min_px",
            thresholds.get("coverage_margin_px", 10)
        )

        if min_margin >= target_margin:
            raw_score = 1.0
        else:
            raw_score = float(min_margin / target_margin)

        raw_score = max(0.0, min(1.0, raw_score))

        return AxisResult(
            study_uid=metadata.get("study_uid", "unknown"),
            axis=AxisName.COVERAGE,
            score=raw_score,
            flag=self._flag_from_score(raw_score),
            raw_metrics={
                "min_margin_px": int(min_margin),
                "mask_detected": True
            },
            rationale=(
                f"Minimum anatomical margin is "
                f"{min_margin}px. Target ≥ {target_margin}px."
            )
        )