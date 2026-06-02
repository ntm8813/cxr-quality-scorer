import torch
import numpy as np
from src.scorers.base import BaseScorer
from schemas.axis_result import AxisResult, AxisName

class CoverageScorer(BaseScorer):
    """
    Evaluates anatomical lung coverage truncation thresholds.
    Uses a pre-trained U-Net segmentation mask to determine if lung 
    boundaries extend too close to the image edge.
    """

    def __init__(self, config: dict, model):
        super().__init__(config)
        self.model = model

    def score(self, image: np.ndarray, metadata: dict) -> AxisResult:
        # 1. Prepare tensor (add batch and channel dimensions)
        tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()
        
        # 2. Run inference
        with torch.no_grad():
            mask_logits = self.model(tensor)
            mask = torch.sigmoid(mask_logits).squeeze().numpy()
            
        binary_mask = (mask > 0.5).astype(np.uint8)
        
        # 3. Calculate margin (simple heuristic: distance from mask bounds to image edges)
        rows = np.any(binary_mask, axis=1)
        cols = np.any(binary_mask, axis=0)
        
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        
        min_margin = min(xmin, 1024 - xmax, ymin, 1024 - ymax)
        
        # 4. Normalize score based on config thresholds
        target_margin = self.config.get("thresholds", {}).get("coverage_margin_min_px", 10)
        raw_score = 1.0 if min_margin >= target_margin else float(min_margin / target_margin)
        
        return AxisResult(
            study_uid=metadata.get("study_uid", "unknown"),
            axis=AxisName.COVERAGE,
            score=raw_score,
            flag=self._flag_from_score(raw_score),
            raw_metrics={"min_margin_px": int(min_margin)},
            rationale=f"Minimum anatomical margin is {min_margin}px. Target ≥ {target_margin}px."
        )