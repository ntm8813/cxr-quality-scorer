import yaml
from typing import List

from schemas.axis_result import AxisResult
from schemas.study_result import StudyResult


class ScoreFusion:
    """Aggregates per-axis results using configured weights."""

    def __init__(self, config_path: str = "configs/v1.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.weights = self.config["axis_weights"]

    def fuse(
        self,
        study_uid: str,
        axis_results: List[AxisResult]
    ) -> StudyResult:

        results_by_axis = {str(r.axis): r for r in axis_results}

        weighted_score = 0.0
        total_weight = 0.0

        for axis_name, weight in self.weights.items():
            if axis_name in results_by_axis:
                weighted_score += results_by_axis[axis_name].score * weight
                total_weight += weight

        if total_weight == 0:
            raise ValueError(
                "No axis results matched configured axis weights."
            )

        composite_score = round(
            float(weighted_score / total_weight),
            4
        )

        repeat_max = (
            self.config["score_ranges"]["repeat_max"] / 100.0
        )

        borderline_max = (
            self.config["score_ranges"]["borderline_max"] / 100.0
        )

        if composite_score <= repeat_max:
            overall_flag = "repeat"
        elif composite_score <= borderline_max:
            overall_flag = "borderline"
        else:
            overall_flag = "acceptable"

        return StudyResult(
            study_uid=study_uid,
            composite_score=composite_score,
            overall_flag=overall_flag,
            axis_results=axis_results,
            metadata_summary={}
        )