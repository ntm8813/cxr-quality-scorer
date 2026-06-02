import cv2
import numpy as np
from src.scorers.base import BaseScorer
from schemas.axis_result import AxisResult, AxisName

class SharpnessScorer(BaseScorer):
    """Computes high-frequency spatial edge details using Laplacian and Sobel variance matrices[cite: 17, 18]."""

    def score(self, image: np.ndarray, metadata: dict) -> AxisResult:
        # Convert matrix back to uint8 for OpenCV operations 
        img_uint8 = (image * 255).astype(np.uint8) 

        # Laplacian variance calculation 
        laplacian = cv2.Laplacian(img_uint8, cv2.CV_64F)
        lap_var = float(laplacian.var()) 

        # Tenengrad matrix gradient magnitude [cite: 18]
        gx = cv2.Sobel(img_uint8, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(img_uint8, cv2.CV_64F, 0, 1, ksize=3)
        tenengrad = float(np.mean(gx**2 + gy**2)) 

        # Extraction from config file [cite: 18]
        threshold = self.config["thresholds"]["laplacian_variance"] 

        # Linear normalization mapping around threshold [cite: 18]
        ratio = lap_var / threshold 
        raw_score = float(np.clip(ratio / 2.0, 0.0, 1.0))

        return AxisResult(
            study_uid=metadata.get("study_uid", "unknown"), 
            axis=AxisName.SHARPNESS, 
            score=raw_score, 
            flag=self._flag_from_score(raw_score), 
            raw_metrics={
                "laplacian_variance": lap_var, 
                "tenengrad": tenengrad, 
                "threshold": threshold 
            },
            rationale=f"Laplacian variance {lap_var:.1f} vs threshold {threshold}." 
        )