from pydantic import BaseModel, Field
from typing import List, Optional
from schemas.axis_result import AxisResult, QualityFlag


class CompositeScore(BaseModel):
    study_uid: str = Field(..., description="DICOM StudyInstanceUID")
    composite_score: float = Field(..., ge=0.0, le=100.0, description="Weighted composite quality score 0-100")
    overall_flag: QualityFlag = Field(..., description="Final accept / borderline / repeat decision")
    axis_results: List[AxisResult] = Field(..., description="Per-axis breakdown")
    summary_rationale: str = Field(default="", description="One-paragraph human-readable summary")

    class Config:
        use_enum_values = True