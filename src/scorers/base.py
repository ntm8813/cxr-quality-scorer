from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np
from schemas.axis_result import AxisResult


class BaseScorer(ABC):
    """
    Every scoring module (Exposure, Sharpness, etc.) inherits from this.
    Intern A implements: ExposureScorer, SharpnessScorer, CoverageScorer, RotationScorer, InspirationScorer
    Intern B implements: ArtifactScorer, MotionBlurScorer
    Both call super().__init__() and implement the score() method.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        config: the loaded v1.yaml as a Python dict.
        Store thresholds here so every scorer reads from one source of truth.
        """
        self.config = config

    @abstractmethod
    def score(self, image: np.ndarray, metadata: Dict[str, Any]) -> AxisResult:
        """
        image    : float32 NumPy array, shape (1024, 1024), values in [0, 1]
        metadata : dict extracted from DICOM header by the DICOMReader
        returns  : a fully populated AxisResult
        """
        pass

    def _flag_from_score(self, score: float) -> str:
        """
        Converts a normalised score (0-1) to a flag string.
        Uses thresholds from config (converted from 0-100 to 0-1 internally).
        """
        repeat_threshold = self.config["score_ranges"]["repeat_max"] / 100.0
        borderline_threshold = self.config["score_ranges"]["borderline_max"] / 100.0

        if score <= repeat_threshold:
            return "repeat"
        elif score <= borderline_threshold:
            return "borderline"
        else:
            return "acceptable"