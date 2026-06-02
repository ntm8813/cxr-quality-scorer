import cv2
import numpy as np
from src.scorers.base import BaseScorer
from schemas.axis_result import AxisResult, AxisName

class RotationScorer(BaseScorer):
    """Calculates anatomical orientation skew thresholds using second-order spatial moments[cite: 41, 42]."""

    def score(self, image: np.ndarray, metadata: dict) -> AxisResult:
        tolerance = self.config["thresholds"]["rotation_tolerance_deg"] 
        img_uint8 = (image * 255).astype(np.uint8) 
        _, binary = cv2.threshold(img_uint8, 30, 255, cv2.THRESH_BINARY) 

        moments = cv2.moments(binary) 
        if moments["mu20"] == moments["mu02"]: 
            angle = 0.0 
        else: 
            angle = 0.5 * np.degrees(
                np.arctan2(2 * moments["mu11"], moments["mu20"] - moments["mu02"]) 
            )

        abs_angle = abs(angle) 
        raw_score = float(np.clip(1.0 - (abs_angle / (tolerance * 3)), 0.0, 1.0)) 

        return AxisResult(
            study_uid=metadata.get("study_uid", "unknown"), 
            axis=AxisName.ROTATION, 
            score=raw_score, 
            flag=self._flag_from_score(raw_score), 
            raw_metrics={"rotation_angle_deg": round(angle, 2), "tolerance_deg": tolerance}, 
            rationale=f"Estimated rotation {angle:.1f}°. Tolerance ±{tolerance}°."
        )