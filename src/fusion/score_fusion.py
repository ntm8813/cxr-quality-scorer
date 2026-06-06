# src/fusion/score_fusion.py

from __future__ import annotations

from typing import List, Dict
from collections import defaultdict
import yaml

from schemas.axis_result import AxisResult
from schemas.study_result import StudyResult


class ScoreFusion:
    """
    FIXED VERSION:
    - Supports multiple scorers per axis
    - No silent overwrites
    - Proper aggregation (mean score per axis)
    - Preserves full traceability
    """

    def __init__(self, config_path: str = "configs/v1.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.weights = {
            k.upper(): float(v)
            for k, v in self.config["axis_weights"].items()
        }

    def _group_by_axis(self, axis_results: List[AxisResult]):
        grouped = defaultdict(list)

        for r in axis_results:
            axis = r.axis.name.upper()
            grouped[axis].append(r)

        return grouped

    def _aggregate_axis(self, results: List[AxisResult]) -> float:
        """
        Default aggregation strategy: simple mean.
        (Later we can upgrade to learned weights)
        """
        return sum(r.score for r in results) / len(results)

    def fuse(self, study_uid: str, axis_results: List[AxisResult]) -> StudyResult:

        grouped = self._group_by_axis(axis_results)

        weighted_score = 0.0
        total_weight = 0.0
        missing_axes = []

        axis_summary = {}

        for axis_name, weight in self.weights.items():

            if axis_name in grouped:

                aggregated_score = self._aggregate_axis(grouped[axis_name])

                weighted_score += aggregated_score * weight
                total_weight += weight

                axis_summary[axis_name] = {
                    "score": aggregated_score,
                    "n_models": len(grouped[axis_name]),
                }

            else:
                missing_axes.append(axis_name)

        if total_weight == 0:
            raise ValueError(
                f"Fusion failed: no matching axes found. Missing: {missing_axes}"
            )

        composite_score = round(weighted_score / total_weight, 4)

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
                "missing_axes": missing_axes,
                "axis_summary": axis_summary
            }
        )