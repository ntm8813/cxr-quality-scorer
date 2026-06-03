import yaml
from typing import List

from schemas.axis_result import AxisResult
from schemas.study_result import StudyResult


class ScoreFusion:
    """Aggregates per-axis results using configured weights (robust + case-safe)."""

    def __init__(self, config_path: str = "configs/v1.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Normalize config keys to uppercase for safe matching
        self.weights = {
            k.upper(): float(v)
            for k, v in self.config["axis_weights"].items()
        }

    def fuse(
        self,
        study_uid: str,
        axis_results: List[AxisResult]
    ) -> StudyResult:

        # Normalize AxisResult keys to uppercase (enum-safe)
        results_by_axis = {
            r.axis.name.upper(): r for r in axis_results
        }

        weighted_score = 0.0
        total_weight = 0.0

        missing_axes = []

        for axis_name, weight in self.weights.items():

            if axis_name in results_by_axis:
                weighted_score += results_by_axis[axis_name].score * weight
                total_weight += weight
            else:
                missing_axes.append(axis_name)

        # HARD FAIL if nothing matched (prevents silent garbage outputs)
        if total_weight == 0:
            raise ValueError(
                f"Fusion failed: no matching axes found. Missing in output: {missing_axes}"
            )

        composite_score = float(weighted_score / total_weight)

        # Normalize score
        composite_score = round(composite_score, 4)

        repeat_max = self.config["score_ranges"]["repeat_max"] / 100.0
        borderline_max = self.config["score_ranges"]["borderline_max"] / 100.0

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
            metadata_summary={
                "missing_axes": missing_axes
            }
        )