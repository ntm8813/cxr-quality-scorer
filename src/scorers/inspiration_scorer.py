import numpy as np
from src.scorers.base import BaseScorer
from schemas.axis_result import AxisResult, AxisName


class InspirationScorer(BaseScorer):
    """
    Inspiration = vertical distribution of anatomical structure.

    Correct idea:
    - good inspiration → lung structure extends downward
    - poor inspiration → diaphragm rises → structure shifts upward
    - must be invariant to brightness/exposure → use gradients
    """

    def score(self, image: np.ndarray, metadata: dict) -> AxisResult:
        img = image.astype(np.float32)

        # normalize intensity
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)

        # ---- STRUCTURAL MAP ----
        gx = np.abs(np.diff(img, axis=1, prepend=img[:, :1]))
        gy = np.abs(np.diff(img, axis=0, prepend=img[:1, :]))
        structure = gx + gy

        h = structure.shape[0]

        upper = structure[:h // 2, :]
        lower = structure[h // 2:, :]

        upper_mass = float(np.mean(upper))
        lower_mass = float(np.mean(lower))

        score = lower_mass / (upper_mass + lower_mass + 1e-6)
        score = float(np.clip(score, 0.0, 1.0))

        return AxisResult(
            study_uid=metadata.get("study_uid", "unknown"),
            axis=AxisName.INSPIRATION,
            score=score,
            flag=self._flag_from_score(score),
            raw_metrics={
                "upper_mass": upper_mass,
                "lower_mass": lower_mass,
            },
            rationale="Inspiration measured via vertical gradient mass distribution."
        )