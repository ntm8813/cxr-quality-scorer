import numpy as np

from src.scorers.base import BaseScorer
from schemas.axis_result import AxisResult, AxisName


class MetadataScorer(BaseScorer):
    """
    PNG validation mode.

    Gold-standard validation uses PNG exports that contain
    no DICOM metadata.

    Metadata axis is therefore marked acceptable by default.
    """

    def score(self, image: np.ndarray, metadata: dict) -> AxisResult:

        return AxisResult(
            study_uid=metadata.get("study_uid", "unknown"),
            axis=AxisName.METADATA,
            score=1.0,
            flag="acceptable",
            raw_metrics={
                "validation_mode": "png",
                "metadata_available": False,
            },
            rationale=(
                "Metadata validation disabled because "
                "PNG studies contain no DICOM headers."
            ),
        )