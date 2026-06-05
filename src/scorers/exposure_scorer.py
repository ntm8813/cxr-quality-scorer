import numpy as np
from src.scorers.base import BaseScorer
from schemas.axis_result import AxisResult, AxisName


class ExposureScorer(BaseScorer):
    """
    Evaluates global histogram quality for CXR exposure.

    Signal: p95 - p5 (dynamic range).  A well-exposed CXR uses most of
    [0, 1].  Both under-exposure (image too dark, p95 low) and
    over-exposure (image too bright, p5 high, clipping at top) shrink
    the usable range.  This makes the score monotonically sensitive to
    severity regardless of direction, fixing the NaN / random-rank issue
    that occurred when the previous mean-penalty was symmetric and the
    degrader randomly chose direction.

    Secondary penalty: clipping ratio (pixels near 0 or 1).
    DICOM deviation index is used when available but is not the primary
    signal, because HDF5 datasets carry no DI metadata.
    """

    def score(self, image: np.ndarray, metadata: dict) -> AxisResult:
        p5  = float(np.percentile(image, 5))
        p95 = float(np.percentile(image, 95))
        mean_pixel = float(np.mean(image))

        # Primary: dynamic range in [0, 1].  Perfect = 1.0, worst = 0.
        dynamic_range = p95 - p5  # [0, 1]

        # Secondary: clipping at either extreme hurts quality.
        clip_lo = float(np.mean(image < 0.03))
        clip_hi = float(np.mean(image > 0.97))
        clipping_ratio = clip_lo + clip_hi

        # DICOM DI (optional)
        di = metadata.get("deviation_index")
        di_penalty = 0.0
        di_ok = True
        if di is not None:
            di = float(di)
            di_ok = -1.0 <= di <= 1.0
            if not di_ok:
                di_penalty = min(abs(di) / 4.0, 0.3)   # proportional, capped

        # Composite score
        # 0.55 weight on dynamic range (primary), 0.30 on clipping, 0.15 on DI
        clip_penalty = min(clipping_ratio * 4.0, 1.0)
        raw_score = (
            0.55 * dynamic_range
            + 0.30 * (1.0 - clip_penalty)
            + 0.15 * (1.0 - di_penalty)
        )
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
                "deviation_index": di,
                "di_within_bounds": di_ok,
            },
            rationale=(
                f"Dynamic range {dynamic_range:.3f} (p5={p5:.3f}, p95={p95:.3f}), "
                f"clipping ratio {clipping_ratio:.3f}."
            ),
        )