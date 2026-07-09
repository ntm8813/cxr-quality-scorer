# src/fusion/score_fusion.py
from __future__ import annotations

from typing import List, Dict
from collections import defaultdict
import yaml

from schemas.axis_result import AxisResult
from schemas.study_result import StudyResult


class ScoreFusion:
    """
    Weighted aggregation of per-axis AxisResult objects into a single
    StudyResult composite score.

    FIX APPLIED (post-delivery review): the denominator used to normalise
    the weighted score is now ALWAYS the full configured weight sum
    (sum of all axis_weights in config, computed once at init), not the
    sum of only the weights belonging to axes that happened to produce a
    result for this particular study.

    Why this matters: previously, if any axis's scorer failed to return a
    result (an exception, a missing model, a corrupted intermediate step),
    that axis's weight silently dropped out of the denominator too. This
    meant a study with a missing axis was scored as if that axis had never
    existed in the weighting scheme at all, which inflates the composite
    score by redistributing the missing axis's weight across whichever
    axes did score — invisibly, with no signal that it happened. After
    this fix, a missing axis contributes 0 to the weighted sum but its
    weight still counts in the denominator, so a missing axis correctly
    drags the composite score down rather than being silently excluded
    from the average.

    Supports multiple scorers per axis, no silent overwrites, proper
    aggregation (mean score per axis), and preserves full traceability
    via metadata_summary.
    """

    def __init__(self, config_path: str = "configs/v1.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.weights = {
            k.upper(): float(v)
            for k, v in self.config["axis_weights"].items()
        }

        # Computed ONCE from config, not from whatever happens to score on
        # a given study. This is the fixed denominator.
        self.full_weight_sum = sum(self.weights.values())

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
        missing_axes = []

        axis_summary = {}

        for axis_name, weight in self.weights.items():

            if axis_name in grouped:

                aggregated_score = self._aggregate_axis(grouped[axis_name])
                contribution = aggregated_score * weight

                weighted_score += contribution

                axis_summary[axis_name] = {
                    "score": round(aggregated_score, 4),
                    "weight": weight,
                    "weighted_contribution": round(contribution, 4),
                    "n_models": len(grouped[axis_name]),
                    "status": "scored",
                }

            else:
                missing_axes.append(axis_name)
                # Missing axis contributes 0 to weighted_score, but its
                # weight is still part of self.full_weight_sum below —
                # this is the fix. It does NOT shrink the denominator.
                axis_summary[axis_name] = {
                    "score": None,
                    "weight": weight,
                    "weighted_contribution": 0.0,
                    "n_models": 0,
                    "status": "missing",
                }

        if self.full_weight_sum == 0:
            raise ValueError(
                "Fusion config error: all axis_weights sum to 0. "
                "Check configs/v1.yaml axis_weights."
            )

        if len(missing_axes) == len(self.weights):
            raise ValueError(
                f"Fusion failed: no matching axes found. All axes missing: {missing_axes}"
            )

        # Denominator is ALWAYS the full configured weight sum — this is
        # the actual fix. Previously this was sum(weight for axis in
        # grouped), which shrank whenever an axis was missing and
        # silently inflated the composite score for that study.
        composite_score = round(weighted_score / self.full_weight_sum, 4)

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
                "axis_summary": axis_summary,
                "fusion_method": (
                    f"composite_score = sum(axis_score * axis_weight) / "
                    f"{self.full_weight_sum:.4f} (full configured weight sum). "
                    f"Missing axes contribute 0 to the numerator; the "
                    f"denominator does NOT shrink to compensate — see "
                    f"axis_summary for the per-axis breakdown."
                ),
            }
        )