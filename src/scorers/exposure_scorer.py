import numpy as np

from src.scorers.base import BaseScorer
from schemas.axis_result import AxisResult, AxisName


class ExposureScorer(BaseScorer):
    """
    Evaluates global histogram quality for CXR exposure.

    Signal: p95 - p5 (dynamic range). A well-exposed CXR uses most of
    [0, 1]. Both under-exposure (image too dark, p95 low) and
    over-exposure (image too bright, p5 high, clipping at top) shrink
    the usable range.

    Secondary penalty: clipping ratio (pixels near 0 or 1).
    DICOM deviation index is used when available but is not the primary
    signal, because HDF5 datasets carry no DI metadata.
    """

    def _thresholds(self) -> dict:
        return self.config.get("thresholds", {})

    def score(self, image: np.ndarray, metadata: dict) -> AxisResult:
        thresholds = self._thresholds()

        p5 = float(np.percentile(image, 5))
        p95 = float(np.percentile(image, 95))
        mean_pixel = float(np.mean(image))

        dynamic_range = float(np.clip(p95 - p5, 0.0, 1.0))

        clip_low = float(thresholds.get("exposure_clip_low", 0.03))
        clip_high = float(thresholds.get("exposure_clip_high", 0.97))
        clip_lo = float(np.mean(image < clip_low))
        clip_hi = float(np.mean(image > clip_high))
        clipping_ratio = clip_lo + clip_hi

        di = metadata.get("deviation_index")
        di_penalty = 0.0
        di_ok = True

        di_min = float(thresholds.get("ei_deviation_min", -1.0))
        di_max = float(thresholds.get("ei_deviation_max", 1.0))

        if di is not None:
            try:
                di = float(di)
                di_ok = di_min <= di <= di_max
                if not di_ok:
                    di_penalty = min(abs(di) / 4.0, 0.3)
            except (TypeError, ValueError):
                di = None
                di_ok = False
                di_penalty = 0.3

        clip_penalty = float(np.clip(clipping_ratio * 4.0, 0.0, 1.0))

        w_dynamic = float(thresholds.get("exposure_dynamic_weight", 0.55))
        w_clipping = float(thresholds.get("exposure_clipping_weight", 0.30))
        w_di = float(thresholds.get("exposure_di_weight", 0.15))
        total_weight = max(w_dynamic + w_clipping + w_di, 1e-6)

        raw_score = (
            w_dynamic * dynamic_range
            + w_clipping * (1.0 - clip_penalty)
            + w_di * (1.0 - di_penalty)
        ) / total_weight
        raw_score = float(np.clip(raw_score, 0.0, 1.0))

        return AxisResult(
            study_uid=metadata.get("study_uid", "unknown"),
            axis=AxisName.EXPOSURE,
            score=raw_score,
            flag=self._flag_from_score(raw_score),
            raw_metrics={
                "p5": p5,
                "p95": p95,
                "dynamic_range": dynamic_range,
                "mean_pixel": mean_pixel,
                "clipping_ratio": clipping_ratio,
                "clip_low_threshold": clip_low,
                "clip_high_threshold": clip_high,
                "exposure_dynamic_weight": w_dynamic,
                "exposure_clipping_weight": w_clipping,
                "exposure_di_weight": w_di,
                "deviation_index": di,
                "deviation_index_min": di_min,
                "deviation_index_max": di_max,
                "di_within_bounds": di_ok,
            },
            rationale=(
                f"Dynamic range {dynamic_range:.3f} (p5={p5:.3f}, p95={p95:.3f}), "
                f"clipping ratio {clipping_ratio:.3f}."
            ),
        )