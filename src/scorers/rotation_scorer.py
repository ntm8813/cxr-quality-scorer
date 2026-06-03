import cv2
import numpy as np

from src.scorers.base import BaseScorer
from schemas.axis_result import AxisResult, AxisName


class RotationScorer(BaseScorer):
    """
    Estimates image rotation using second-order image moments.
    A perfectly vertical anatomical structure corresponds to 0° rotation error.
    """

    def score(self, image: np.ndarray, metadata: dict) -> AxisResult:

        tolerance = self.config["thresholds"]["rotation_tolerance_deg"]

        img_uint8 = (image * 255).astype(np.uint8)

        _, binary = cv2.threshold(
            img_uint8,
            30,
            255,
            cv2.THRESH_BINARY
        )

        moments = cv2.moments(binary)

        if moments["m00"] == 0:
            angle_error = 90.0

        elif moments["mu20"] == moments["mu02"]:
            angle_error = 0.0

        else:
            raw_angle = 0.5 * np.degrees(
                np.arctan2(
                    2 * moments["mu11"],
                    moments["mu20"] - moments["mu02"]
                )
            )

            # Convert orientation into deviation from vertical.
            angle_error = abs(90.0 - abs(raw_angle))

        raw_score = float(
            np.clip(
                1.0 - (angle_error / (tolerance * 3)),
                0.0,
                1.0
            )
        )

        return AxisResult(
            study_uid=metadata.get("study_uid", "unknown"),
            axis=AxisName.ROTATION,
            score=raw_score,
            flag=self._flag_from_score(raw_score),
            raw_metrics={
                "rotation_angle_deg": round(float(angle_error), 2),
                "tolerance_deg": tolerance
            },
            rationale=(
                f"Estimated rotation error "
                f"{angle_error:.1f}°. "
                f"Tolerance ±{tolerance}°."
            )
        )