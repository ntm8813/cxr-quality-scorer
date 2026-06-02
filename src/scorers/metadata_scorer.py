import numpy as np
from src.scorers.base import BaseScorer
from schemas.axis_result import AxisResult, AxisName

REQUIRED_TAGS = [
    "study_uid", "modality", "view_position",
    "body_part", "patient_id", "bits_stored" 
]

VALID_MODALITIES = {"CR", "DX"} 
VALID_VIEW_POSITIONS = {"PA", "AP"} 

class MetadataScorer(BaseScorer):
    """Checks presence and categorical validity of crucial clinical DICOM header tags."""

    def score(self, image: np.ndarray, metadata: dict) -> AxisResult:
        issues = [] 

        # Check required tags present and non-empty 
        for tag in REQUIRED_TAGS: 
            val = metadata.get(tag) 
            if val is None or str(val).strip() == "": 
                issues.append(f"Missing tag: {tag}")

        # Check modality is CXR-appropriate [cite: 12]
        modality = str(metadata.get("modality", "")).upper() 
        if modality and modality not in VALID_MODALITIES: 
            issues.append(f"Unexpected modality: {modality}") 

        # Check view position [cite: 13]
        view = str(metadata.get("view_position", "")).upper() 
        if view and view not in VALID_VIEW_POSITIONS: 
            issues.append(f"Unexpected view position: {view}") 

        # Score based on issue count [cite: 13]
        issue_count = len(issues) 
        if issue_count == 0: 
            raw_score = 1.0 
        elif issue_count <= 2: 
            raw_score = 0.6 
        else: 
            raw_score = 0.2 

        return AxisResult(
            study_uid=metadata.get("study_uid", "unknown"), 
            axis=AxisName.METADATA, 
            score=raw_score,
            flag=self._flag_from_score(raw_score), 
            raw_metrics={"issue_count": issue_count, "issues": issues}, 
            rationale=f"{issue_count} metadata issue(s) found." if issues else "All required DICOM tags present and valid." 
        )