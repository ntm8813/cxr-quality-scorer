# src/explanation/explanation_module.py
from __future__ import annotations
from typing import Dict, Any
from schemas.axis_result import AxisResult
from schemas.study_result import StudyResult


# ── Template library ──────────────────────────────────────────────────────────
# Outer key  : axis name string (lowercase, matches AxisName enum values)
# Inner key  : flag string (matches QualityFlag enum values)
# Template   : uses .format(**safe_metrics) — every key has a safe default

_TEMPLATES: Dict[str, Dict[str, str]] = {

    "sharpness": {
        "acceptable": (
            "Sharpness acceptable — Laplacian variance {laplacian_variance:.1f} "
            "(log-score {lap_score:.3f}). No significant blur detected."
        ),
        "borderline": (
            "Sharpness borderline — Laplacian variance {laplacian_variance:.1f} "
            "(log-score {lap_score:.3f}), below expected range. "
            "Possible mild motion or slight defocus. Review before sharing."
        ),
        "repeat": (
            "Sharpness unacceptable — Laplacian variance {laplacian_variance:.1f} "
            "(log-score {lap_score:.3f}). Significant motion blur or defocus. "
            "Repeat acquisition recommended."
        ),
    },

    "exposure": {
        "acceptable": (
            "Exposure adequate — dynamic range {dynamic_range:.3f} "
            "(p5={p5:.3f}, p95={p95:.3f}), clipping ratio {clipping_ratio:.3f}."
        ),
        "borderline": (
            "Exposure borderline — dynamic range {dynamic_range:.3f} "
            "(p5={p5:.3f}, p95={p95:.3f}). "
            "Image may be slightly under- or over-exposed. "
            "Consider protocol review."
        ),
        "repeat": (
            "Exposure unacceptable — dynamic range {dynamic_range:.3f} "
            "(p5={p5:.3f}, p95={p95:.3f}), clipping ratio {clipping_ratio:.3f}. "
            "Image is significantly under- or over-exposed. Repeat required."
        ),
    },

    "rotation": {
        "acceptable": (
            "Positioning acceptable — deviation {rotation_angle_deg:.1f}° "
            "from vertical axis, within tolerance of ±{tolerance_deg:.0f}°."
        ),
        "borderline": (
            "Positioning borderline — deviation {rotation_angle_deg:.1f}° "
            "from vertical axis, marginally exceeding ±{tolerance_deg:.0f}° tolerance. "
            "Patient may be slightly rotated. Review clavicle symmetry."
        ),
        "repeat": (
            "Positioning unacceptable — deviation {rotation_angle_deg:.1f}° "
            "from vertical axis, well beyond ±{tolerance_deg:.0f}° tolerance. "
            "Significant patient rotation. Repeat with correct positioning."
        ),
    },

    "coverage": {
        "acceptable": (
            "Coverage adequate — lung field fully visible, "
            "minimum boundary margin {min_margin_px:.0f}px. "
            "No costophrenic angle or apex truncation detected."
        ),
        "borderline": (
            "Coverage borderline — minimum boundary margin {min_margin_px:.0f}px. "
            "Possible partial clipping of lung field. Review before sharing."
        ),
        "repeat": (
            "Coverage unacceptable — minimum boundary margin {min_margin_px:.0f}px. "
            "Significant lung field clipping detected. "
            "Repeat with correct collimation."
        ),
    },

    "inspiration": {
        "acceptable": (
            "Inspiration adequate — lower-zone gradient mass ratio {lower_mass:.4f}, "
            "indicating sufficient lung inflation."
        ),
        "borderline": (
            "Inspiration borderline — lower-zone gradient mass ratio {lower_mass:.4f}. "
            "Possible shallow breath. May affect basal zone assessment."
        ),
        "repeat": (
            "Inspiration inadequate — lower-zone gradient mass ratio {lower_mass:.4f}. "
            "Significant diaphragm elevation suspected. "
            "Repeat with full inspiration instruction."
        ),
    },

    "artifact": {
        "acceptable": (
            "No significant artifacts detected — "
            "artifact probability {artifact_probability:.3f}."
        ),
        "borderline": (
            "Artifact presence borderline — probability {artifact_probability:.3f}. "
            "Minor artifacts may be present. Review image before clinical use."
        ),
        "repeat": (
            "Artifacts detected — probability {artifact_probability:.3f}. "
            "Significant artifacts present (grid lines, foreign objects, or "
            "processing artifacts). Repeat acquisition recommended."
        ),
    },

    "metadata": {
        "acceptable": (
            "All required DICOM metadata present and valid. "
            "No tag issues found."
        ),
        "borderline": (
            "Metadata issues — {issue_count} problem(s) detected: {issues_str}. "
            "Study may have incomplete tagging."
        ),
        "repeat": (
            "Critical metadata missing — {issue_count} required tag(s) absent "
            "or invalid: {issues_str}. "
            "Study cannot be reliably identified or routed."
        ),
    },
}

# Safe defaults for every key that any template references
_DEFAULTS: Dict[str, Any] = {
    "laplacian_variance"  : 0.0,
    "lap_score"           : 0.0,
    "ten_score"           : 0.0,
    "tenengrad"           : 0.0,
    "threshold"           : 80.0,
    "dynamic_range"       : 0.0,
    "p5"                  : 0.0,
    "p95"                 : 0.0,
    "mean_pixel"          : 0.0,
    "clipping_ratio"      : 0.0,
    "deviation_index"     : None,
    "di_within_bounds"    : True,
    "rotation_angle_deg"  : 0.0,
    "tolerance_deg"       : 5.0,
    "min_margin_px"       : 0.0,
    "mask_detected"       : False,
    "lower_mass"          : 0.0,
    "upper_mass"          : 0.0,
    "lung_area_ratio"     : 0.0,
    "artifact_probability": 0.0,
    "blur_probability"    : 0.0,
    "issue_count"         : 0,
    "issues"              : [],
    "issues_str"          : "none",
}


class ExplanationModule:
    """
    Maps each AxisResult (score + flag + raw_metrics) to a
    human-readable rationale string using per-axis, per-flag templates.

    Usage:
        module  = ExplanationModule()
        study   = fusion.fuse(uid, axis_results)
        study   = module.enrich_study(study)
        # study.axis_results[i].rationale is now a full sentence
        # study.metadata_summary["summary_rationale"] is the paragraph summary
    """

    def explain_axis(self, result: AxisResult) -> str:
        """
        Returns an enriched rationale string for a single AxisResult.
        Falls back to the existing rationale if no template is found.
        """
        # Safely extract string values from either str or Enum
        axis_key = result.axis if isinstance(result.axis, str) else result.axis.value
        flag_key = result.flag if isinstance(result.flag, str) else result.flag.value

        axis_templates = _TEMPLATES.get(axis_key)
        if not axis_templates:
            return result.rationale or f"{axis_key}: score {result.score:.3f} ({flag_key})."

        template = axis_templates.get(flag_key)
        if not template:
            return result.rationale or f"{axis_key}: score {result.score:.3f} ({flag_key})."

        # Build a safe metrics dict: defaults first, then actual values override
        safe: Dict[str, Any] = dict(_DEFAULTS)
        safe.update(result.raw_metrics or {})

        # Convert list fields to readable strings for template insertion
        issues_list = safe.get("issues", [])
        safe["issues_str"] = (
            "; ".join(str(x) for x in issues_list) if issues_list else "none"
        )
        # Ensure no list values remain — they break .format()
        for k, v in list(safe.items()):
            if isinstance(v, list):
                safe[k] = str(v)
            if v is None:
                safe[k] = "N/A"

        try:
            return template.format(**safe)
        except (KeyError, ValueError):
            # Graceful fallback — never crash the pipeline over a missing key
            return result.rationale or f"{axis_key}: score {result.score:.3f} ({flag_key})."

    def explain_study(self, study: StudyResult) -> str:
        """
        Generates a one-paragraph study-level summary from all axis results.
        """
        score = study.composite_score
        flag  = study.overall_flag
        axes  = study.axis_results

        # score is 0–1 in your fusion; display as 0–100
        score_display = score * 100.0 if score <= 1.0 else score

        def flag_val(r):
            return r.flag if isinstance(r.flag, str) else r.flag.value

        def axis_val(r):
            return r.axis if isinstance(r.axis, str) else r.axis.value

        repeat_axes     = [axis_val(r) for r in axes if flag_val(r) == "repeat"]
        borderline_axes = [axis_val(r) for r in axes if flag_val(r) == "borderline"]

        if flag == "acceptable":
            return (
                f"Overall quality acceptable (composite score {score_display:.1f}/100). "
                f"All {len(axes)} evaluated axes are within acceptable bounds. "
                f"Study is ready for sharing."
            )
        elif flag == "borderline":
            return (
                f"Overall quality borderline (composite score {score_display:.1f}/100). "
                f"Axes requiring review: {', '.join(borderline_axes) or 'none'}. "
                f"Radiologist review recommended before sharing."
            )
        else:
            msg = (
                f"Overall quality unacceptable (composite score {score_display:.1f}/100). "
                f"Axes flagged for repeat: {', '.join(repeat_axes) or 'unknown'}. "
            )
            if borderline_axes:
                msg += f"Borderline axes: {', '.join(borderline_axes)}. "
            msg += "Repeat acquisition recommended."
            return msg

    def enrich_study(self, study: StudyResult) -> StudyResult:
        """
        Returns a new StudyResult with:
          - enriched per-axis rationale on each AxisResult
          - study.metadata_summary["summary_rationale"] set to the paragraph summary

        Uses model_copy() — Pydantic v2 compatible.
        Does NOT mutate the original StudyResult.
        """
        enriched_axes = []
        for ar in study.axis_results:
            enriched_axes.append(
                ar.model_copy(update={"rationale": self.explain_axis(ar)})
            )

        updated_metadata = dict(study.metadata_summary)
        updated_metadata["summary_rationale"] = self.explain_study(study)

        return study.model_copy(update={
            "axis_results"   : enriched_axes,
            "metadata_summary": updated_metadata,
        })