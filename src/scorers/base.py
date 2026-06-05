from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np
from schemas.axis_result import AxisResult


class BaseScorer(ABC):
    """
    Base class for all scorers.

    Supports BOTH:
    - BaseScorer(config)
    - BaseScorer(config, model=...)
    - BaseScorer(config, model)
    """

    def __init__(self, config: Dict[str, Any], model: Optional[Any] = None):
        """
        config: loaded v1.yaml as dict
        model : optional ML model (UNet, etc.) used by DL scorers
        """
        self.config = config
        self.model = model

    @abstractmethod
    def score(self, image: np.ndarray, metadata: Dict[str, Any]) -> AxisResult:
        """
        image    : float32 NumPy array, shape (1024, 1024), values in [0, 1]
        metadata : dict extracted from DICOM header
        returns  : AxisResult
        """
        pass

    def _flag_from_score(self, score: float) -> str:
        """
        Converts normalized score (0–1) into quality flag.
        Thresholds are defined in config (0–100 scale).
        """
        score_ranges = self.config.get("score_ranges", {})

        repeat_threshold = score_ranges.get("repeat_max", 0) / 100.0
        borderline_threshold = score_ranges.get("borderline_max", 0) / 100.0

        if score <= repeat_threshold:
            return "repeat"
        elif score <= borderline_threshold:
            return "borderline"
        else:
            return "acceptable"