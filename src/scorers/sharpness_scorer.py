import cv2
import numpy as np
from src.scorers.base import BaseScorer
from schemas.axis_result import AxisResult, AxisName


class SharpnessScorer(BaseScorer):
    """
    Evaluates sharpness using Laplacian variance and Tenengrad.

    Root cause of previous weak Spearman ρ
    ----------------------------------------
    The previous normalization was:
        raw_score = clip(lap_var / threshold / 2.0, 0, 1)

    For a 1024×1024 CXR blurred at σ=5, lap_var is often already > 80
    (the threshold), so the score clips to 1.0 and all discriminative
    signal above threshold is lost.  σ=5 and σ=10 images both score 1.0
    → rank order is random → Spearman ρ ≈ 0 or weakly negative.

    Fix: log-domain normalization.
    Laplacian variance spans multiple orders of magnitude across the
    sharp → heavy-blur range on real CXRs (typically 10–2000).
    A log mapping preserves separation across the full range:

        score = clip(log(1 + lap_var) / log(1 + scale), 0, 1)

    where `scale` is the variance value that should map to score=1.0
    (a fully sharp CXR).  We use tenengrad as a second signal and
    blend them so the scorer is robust to images where one measure
    saturates independently.
    """

    # Empirical scale constants for 1024×1024 float32 CXRs.
    # lap_var for a sharp CXR ≈ 300–600; for σ=10 blur ≈ 5–30.
    # Setting scale to 500 puts sharp images at ~0.9 and heavy blur at ~0.3.
    LAP_SCALE = 500.0
    # Tenengrad (mean of squared Sobel gradients) for sharp ≈ 800–3000.
    TEN_SCALE = 2000.0

    def score(self, image: np.ndarray, metadata: dict) -> AxisResult:
        img_uint8 = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)

        # Laplacian variance
        laplacian = cv2.Laplacian(img_uint8, cv2.CV_64F)
        lap_var   = float(laplacian.var())

        # Tenengrad
        gx = cv2.Sobel(img_uint8, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(img_uint8, cv2.CV_64F, 0, 1, ksize=3)
        tenengrad = float(np.mean(gx ** 2 + gy ** 2))

        # Log-domain scores — monotone, never saturates prematurely
        lap_score = float(np.clip(
            np.log1p(lap_var) / np.log1p(self.LAP_SCALE), 0.0, 1.0
        ))
        ten_score = float(np.clip(
            np.log1p(tenengrad) / np.log1p(self.TEN_SCALE), 0.0, 1.0
        ))

        # Blend: Laplacian carries more weight (more noise-robust on CXRs)
        raw_score = 0.65 * lap_score + 0.35 * ten_score

        # Config threshold kept for _flag_from_score comparisons only
        threshold = self.config["thresholds"]["laplacian_variance"]

        return AxisResult(
            study_uid=metadata.get("study_uid", "unknown"),
            axis=AxisName.SHARPNESS,
            score=raw_score,
            flag=self._flag_from_score(raw_score),
            raw_metrics={
                "laplacian_variance": lap_var,
                "tenengrad":          tenengrad,
                "lap_score":          lap_score,
                "ten_score":          ten_score,
                "threshold":          threshold,
            },
            rationale=(
                f"Laplacian variance {lap_var:.1f} → log-score {lap_score:.3f}; "
                f"Tenengrad {tenengrad:.1f} → log-score {ten_score:.3f}."
            ),
        )