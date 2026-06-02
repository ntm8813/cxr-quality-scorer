import yaml
from typing import List
from schemas.axis_result import AxisResult
from schemas.study_result import StudyResult

class ScoreFusion:
    """Aggregates per-axis results using custom configured weights."""

    def __init__(self, config_path: str = "configs/v1.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.weights = self.config["axis_weights"] 

    def fuse(self, study_uid: str, axis_results: List[AxisResult]) -> StudyResult:
        results_by_axis = {r.axis: r for r in axis_results} 
        total_weight = 0.0 
        weighted_score = 0.0

        for axis_name, weight in self.weights.items(): 
            if axis_name in results_by_axis: 
                weighted_score += results_by_axis[axis_name].score * weight 
                total_weight += weight 

        if total_weight == 0: 
            raise ValueError("No axis results matched config weights") 

        composite = (weighted_score / total_weight) 
        composite = round(float(composite), 4)

        # Map to QualityFlag configurations
        repeat_max = self.config["score_ranges"]["repeat_max"] / 100.0 
        borderline_max = self.config["score_ranges"]["borderline_max"] / 100.0 

        if composite <= repeat_max: 
            overall_flag = "repeat" 
        elif composite <= borderline_max: 
            overall_flag = "borderline" 
        else: 
            overall_flag = "acceptable" 

        # Return fully compatible StudyResult schema profile
        return StudyResult(
            study_instance_uid=study_uid,
            axes=axis_results,
            composite_score=composite,
            passed=(overall_flag != "repeat")
        )