from pydantic import BaseModel, Field, ConfigDict
from enum import Enum
from typing import Optional, Dict, Any


class QualityFlag(str, Enum):
    ACCEPTABLE = "acceptable"
    BORDERLINE = "borderline"
    REPEAT = "repeat"


class AxisName(str, Enum):
    SHARPNESS = "sharpness"
    EXPOSURE = "exposure"
    ROTATION = "rotation"
    COVERAGE = "coverage"
    INSPIRATION = "inspiration"
    ARTIFACT = "artifact"
    METADATA = "metadata"


class AxisResult(BaseModel):
    study_uid: str = Field(..., description="DICOM StudyInstanceUID")
    axis: AxisName = Field(..., description="Which quality axis this result covers")
    score: float = Field(..., ge=0.0, le=1.0, description="Normalised score, 1.0 = perfect quality")
    flag: QualityFlag = Field(..., description="accept / borderline / repeat")
    raw_metrics: Dict[str, Any] = Field(default_factory=dict, description="Raw numeric values used to compute score")
    rationale: str = Field(default="", description="Human-readable explanation of this score")

    # Upgraded to Pydantic v2 format to clear deprecation warnings
    model_config = ConfigDict(use_enum_values=True)