import numpy as np
from src.scorers.base import BaseScorer
from schemas.axis_result import AxisResult, AxisName


class CoverageScorer(BaseScorer):
    """
    FINAL STABLE COVERAGE SCORER (CONSTRAINT-BASED)

    Core insight:
    replicate-padding enforces a HARD MATHEMATICAL CONSTRAINT:
        ∂I/∂x = 0 or ∂I/∂y = 0 in padded regions

    Coverage loss is detected by measuring how much of the image
    exhibits this constraint violation reduction.
    """

    BORDER_FRAC = 0.12  # slightly above max crop (0.12)

    def score(self, image: np.ndarray, metadata: dict) -> AxisResult:
        img = image.astype(np.float32)

        # normalize robustly
        img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-6)

        h, w = img.shape
        bh = max(1, int(h * self.BORDER_FRAC))
        bw = max(1, int(w * self.BORDER_FRAC))

        # gradients
        gy = np.abs(np.diff(img, axis=0))
        gx = np.abs(np.diff(img, axis=1))

        # border regions
        top = gy[:bh, :]
        bot = gy[-bh:, :]
        left = gx[:, :bw]
        right = gx[:, -bw:]

        # center region (true anatomy reference)
        center_gy = gy[bh:-bh, :]
        center_gx = gx[:, bw:-bw]

        # robust safety (avoid division collapse)
        eps = 1e-6

        border_signal = (
            np.mean(top) +
            np.mean(bot) +
            np.mean(left) +
            np.mean(right)
        ) / 4.0

        center_signal = (
            np.mean(center_gy) +
            np.mean(center_gx)
        ) / 2.0

        # key invariant ratio
        ratio = border_signal / (center_signal + eps)

        # invert so severity increases → score decreases
        score = float(np.clip(ratio, 0.0, 1.0))

        return AxisResult(
            study_uid=metadata.get("study_uid", "unknown"),
            axis=AxisName.COVERAGE,
            score=score,
            flag=self._flag_from_score(score),
            raw_metrics={
                "border_signal": float(border_signal),
                "center_signal": float(center_signal),
                "ratio": float(ratio),
            },
            rationale=(
                f"Boundary constraint ratio={ratio:.4f}. "
                f"Lower values indicate stronger edge replication effect."
            ),
        )