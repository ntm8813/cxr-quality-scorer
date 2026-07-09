# schemas/rejected_result.py
"""
Schema for a study that failed input validation and was NOT scored.

This is deliberately a different shape than StudyResult — it must never
be confused with, or silently coerced into, a real scored result. Code
that consumes pipeline output (app.py, batch scripts, API callers) should
check which type it received before reading composite_score / axis_results.
"""
from __future__ import annotations

from pydantic import BaseModel, Field
from typing import List, Dict, Any


class RejectedResult(BaseModel):
    study_uid: str
    status: str = Field("rejected", description="Always 'rejected' — distinguishes this from a scored StudyResult")
    reason: str = Field(..., description="Human-readable summary of why scoring was skipped")
    failed_checks: List[str] = Field(default_factory=list, description="Machine-readable list of failed validation checks")
    details: Dict[str, Any] = Field(default_factory=dict, description="Diagnostic context: shape, dtype, metadata keys present, etc.")