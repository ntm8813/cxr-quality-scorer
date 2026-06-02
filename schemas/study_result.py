# schemas/study_result.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from schemas.axis_result import AxisResult

class StudyResult(BaseModel):
    study_uid: str
    composite_score: float
    overall_flag: str
    axis_results: List[AxisResult]
    metadata_summary: Dict[str, Any] = Field(default_factory=dict)