import numpy as np
from src.scorers.base import BaseScorer
from schemas.axis_result import AxisResult, AxisName, QualityFlag

class ExposureScorer(BaseScorer):
    """Evaluates global histogram statistics and DICOM Deviation Index (DI) flags[cite: 5, 6]."""

    def score(self, image: np.ndarray, metadata: dict) -> AxisResult:
        # 1. Histogram statistics [cite: 5]
        mean_pixel = float(np.mean(image)) 
        p5 = float(np.percentile(image, 5))
        p95 = float(np.percentile(image, 95)) 
        clipping_ratio = float(np.mean(image > 0.95) + np.mean(image < 0.05)) 

        # 2. DICOM deviation index check [cite: 5]
        di = metadata.get("deviation_index") 
        di_ok = True 
        if di is not None: 
            di = float(di) 
            di_ok = -1.0 <= di <= 1.0 

        # 3. Score computation [cite: 6]
        mean_penalty = abs(mean_pixel - 0.5) * 2  # 0 = perfect, 1 = worst [cite: 6, 7]
        clip_penalty = min(clipping_ratio * 5, 1.0) 
        di_penalty = 0.0 if di_ok else 0.3 

        raw_score = 1.0 - (0.4 * mean_penalty + 0.4 * clip_penalty + 0.2 * di_penalty) 
        raw_score = float(np.clip(raw_score, 0.0, 1.0)) 

        return AxisResult(
            study_uid=metadata.get("study_uid", "unknown"),
            axis=AxisName.EXPOSURE, 
            score=raw_score, 
            flag=self._flag_from_score(raw_score), 
            raw_metrics={
                "mean_pixel": mean_pixel, 
                "p5": p5, 
                "p95": p95, 
                "clipping_ratio": clipping_ratio, 
                "deviation_index": di, 
                "di_within_bounds": di_ok 
            },
            rationale=f"Mean pixel intensity {mean_pixel:.2f}, clipping ratio {clipping_ratio:.3f}." 
        )