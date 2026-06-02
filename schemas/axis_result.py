# schemas/axis_result.py
from enum import Enum
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

class AxisName(str, Enum):
    EXPOSURE = "exposure"
    SHARPNESS = "sharpness"
    METADATA = "metadata"
    ROTATION = "rotation"
    COVERAGE = "coverage"
    INSPIRATION = "inspiration"
    ARTIFACT = "artifact"

class QualityFlag(str, Enum):
    ACCEPTABLE = "acceptable"
    BORDERLINE = "borderline"
    REPEAT = "repeat"

class AxisResult(BaseModel):
    """
    Standardised assessment schema for a single quality axis evaluation.
    """
    study_uid: str
    axis: AxisName
    score: float = Field(..., ge=0.0, le=1.0, description="Normalized score from 0.0 (worst) to 1.0 (best)")
    flag: QualityFlag = Field(..., description="Status flag: acceptable, borderline, or repeat")
    raw_metrics: Dict[str, Any] = Field(default_factory=dict, description="Raw dictionary measurements for debugging")
    rationale: Optional[str] = Field(None, description="Human-readable text string describing the system's reasoning")