# src/scorers/inspiration_scorer.py
import torch
import numpy as np
from src.scorers.base import BaseScorer
from schemas.axis_result import AxisResult, AxisName

class InspirationScorer(BaseScorer):
    """Evaluates adequate expansion/inspiration via segmented lung area heuristics."""
    
    def __init__(self, config: dict, model):
        super().__init__(config)
        self.model = model

    def score(self, image: np.ndarray, metadata: dict) -> AxisResult:
        # 1. Generate segmentation mask using the shared model instance
        tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()
        with torch.no_grad():
            mask = torch.sigmoid(self.model(tensor)).squeeze().numpy()
        binary_mask = (mask > 0.5).astype(np.uint8)
        
        # 2. Calculate lung area ratio against total image dimensions
        lung_pixels = np.sum(binary_mask)
        total_pixels = image.shape[0] * image.shape[1]
        lung_ratio = lung_pixels / total_pixels
        
        # Fetch minimum expectation threshold from config
        min_ratio = self.config.get("thresholds", {}).get("inspiration_min_ratio", 0.18)
        raw_score = 1.0 if lung_ratio >= min_ratio else float(lung_ratio / min_ratio)
        
        return AxisResult(
            study_uid=metadata.get("study_uid", "unknown"),
            axis=AxisName.INSPIRATION,
            score=raw_score,
            flag=self._flag_from_score(raw_score),
            raw_metrics={"lung_area_ratio": float(lung_ratio), "lung_pixels": int(lung_pixels)},
            rationale=f"Estimated lung expansion area ratio is {lung_ratio:.3f}. Target baseline ≥ {min_ratio}."
        )