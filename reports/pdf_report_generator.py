# src/reports/pdf_report_generator.py
"""
Day 25 — PDF QA report generator.
Generates per-study PDF reports and per-batch PDF QA logs.

Usage:
    from src.reports.pdf_report_generator import (
        generate_study_report,
        generate_batch_report,
    )
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional
import json

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image as RLImage, PageBreak,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

from schemas.study_result import StudyResult

# ── Colour palette matching the Streamlit UI ──────────────────
_GREEN  = colors.HexColor("#155724")
_AMBER  = colors.HexColor("#856404")
_RED    = colors.HexColor("#721c24")
_GREEN_BG = colors.HexColor("#d4edda")
_AMBER_BG = colors.HexColor("#fff3cd")
_RED_BG   = colors.HexColor("#f8d7da")
_HEADER_BG= colors.HexColor("#2c3e50")
_LIGHT_GREY = colors.HexColor("#f8f9fa")

FLAG_COLOR = {
    "acceptable": (_GREEN,  _GREEN_BG),
    "borderline": (_AMBER,  _AMBER_BG),
    "repeat"    : (_RED,    _RED_BG),
}


def _styles():
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "title", parent=base["Title"],
            fontSize=18, textColor=colors.white,
            alignment=TA_CENTER, spaceAfter=4,
        ),
        "subtitle": ParagraphStyle(
            "subtitle", parent=base["Normal"],
            fontSize=10, textColor=colors.HexColor("#aaaaaa"),
            alignment=TA_CENTER, spaceAfter=12,
        ),
        "heading": ParagraphStyle(
            "heading", parent=base["Heading2"],
            fontSize=12, textColor=_HEADER_BG,
            spaceBefore=10, spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "body", parent=base["Normal"],
            fontSize=9, spaceAfter=4,
        ),
        "rationale": ParagraphStyle(
            "rationale", parent=base["Normal"],
            fontSize=8, textColor=colors.HexColor("#444444"),
            spaceAfter=2, leading=12,
        ),
        "flag_acceptable": ParagraphStyle(
            "flag_ok", parent=base["Normal"],
            fontSize=22, textColor=_GREEN,
            alignment=TA_CENTER,
        ),
        "flag_borderline": ParagraphStyle(
            "flag_bl", parent=base["Normal"],
            fontSize=22, textColor=_AMBER,
            alignment=TA_CENTER,
        ),
        "flag_repeat": ParagraphStyle(
            "flag_rp", parent=base["Normal"],
            fontSize=22, textColor=_RED,
            alignment=TA_CENTER,
        ),
    }
    return styles


def _flag_badge(flag: str, score: float, styles: dict) -> Table:
    """Renders a coloured score card for the overall flag."""
    fg, bg = FLAG_COLOR.get(flag, (_RED, _RED_BG))
    score_pct = score * 100 if score <= 1.0 else score

    icon  = {"acceptable": "✔", "borderline": "⚠", "repeat": "✗"}.get(flag, "?")
    label = flag.upper()

    data = [
        [Paragraph(f"{icon} {label}", styles.get(f"flag_{flag}", styles["body"]))],
        [Paragraph(f"{score_pct:.1f} / 100", ParagraphStyle(
            "score_num", fontSize=28, textColor=fg, alignment=TA_CENTER
        ))],
    ]
    t = Table(data, colWidths=[9*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("BOX",        (0, 0), (-1, -1), 1, fg),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
    ]))
    return t


def _axis_table(result: StudyResult, styles: dict) -> Table:
    """Renders the per-axis breakdown as a formatted table."""
    header = ["Axis", "Score", "Flag", "Rationale"]
    rows   = [header]

    for ar in result.axis_results:
        ax_name = ar.axis if isinstance(ar.axis, str) else ar.axis.value
        ax_flag = ar.flag if isinstance(ar.flag, str) else ar.flag.value
        score_pct = ar.score * 100 if ar.score <= 1.0 else ar.score
        rows.append([
            Paragraph(ax_name.capitalize(), styles["body"]),
            Paragraph(f"{score_pct:.1f}", styles["body"]),
            Paragraph(ax_flag.upper(), styles["body"]),
            Paragraph(ar.rationale or "—", styles["rationale"]),
        ])

    col_widths = [3*cm, 2*cm, 3*cm, 10*cm]
    t = Table(rows, colWidths=col_widths, repeatRows=1)

    style_cmds = [
        ("BACKGROUND",  (0, 0), (-1, 0), _HEADER_BG),
        ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, 0), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, _LIGHT_GREY]),
        ("GRID",        (0, 0), (-1, -1), 0.25, colors.grey),
        ("VALIGN",      (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING",  (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ("FONTSIZE",    (0, 1), (-1, -1), 8),
    ]

    # Colour the flag column
    flag_col = 2
    flag_map = {
        "ACCEPTABLE": _GREEN_BG,
        "BORDERLINE": _AMBER_BG,
        "REPEAT"    : _RED_BG,
    }
    for row_idx, ar in enumerate(result.axis_results, start=1):
        ax_flag = ar.flag if isinstance(ar.flag, str) else ar.flag.value
        bg = flag_map.get(ax_flag.upper(), colors.white)
        style_cmds.append(("BACKGROUND", (flag_col, row_idx), (flag_col, row_idx), bg))

    t.setStyle(TableStyle(style_cmds))
    return t


def generate_study_report(
    result: StudyResult,
    output_path: str | Path,
    image_path: Optional[str | Path] = None,
) -> Path:
    """
    Generate a per-study PDF QA report.

    Args:
        result      : StudyResult from run_pipeline()
        output_path : where to save the PDF
        image_path  : optional path to the CXR PNG/DICOM for thumbnail

    Returns:
        Path to the generated PDF
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc    = SimpleDocTemplate(
        str(output_path),
        pagesize    = A4,
        leftMargin  = 1.5*cm,
        rightMargin = 1.5*cm,
        topMargin   = 1.5*cm,
        bottomMargin= 1.5*cm,
    )
    styles = _styles()
    story  = []

    # Header bar
    header_data = [[
        Paragraph("CXR Quality Assessment Report", styles["title"]),
    ]]
    header_table = Table(header_data, colWidths=[18*cm])
    header_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), _HEADER_BG),
        ("TOPPADDING",    (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 0.4*cm))

    # Study metadata
    story.append(Paragraph("Study Information", styles["heading"]))
    meta_rows = [
        ["Study UID", result.study_uid],
        ["Project",   "MTV-INT-RAD-003 — Medtatvaa Healthcare"],
        ["Pipeline",  "v1.0"],
    ]
    meta_table = Table(meta_rows, colWidths=[4*cm, 14*cm])
    meta_table.setStyle(TableStyle([
        ("FONTNAME",  (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE",  (0, 0), (-1, -1), 9),
        ("GRID",      (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, _LIGHT_GREY]),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 0.4*cm))

    # Image thumbnail + score card side by side
    story.append(Paragraph("Quality Assessment", styles["heading"]))
    flag  = result.overall_flag
    badge = _flag_badge(flag, result.composite_score, styles)
    summary = result.metadata_summary.get("summary_rationale", "")

    if image_path and Path(image_path).exists():
        try:
            img      = RLImage(str(image_path), width=7*cm, height=7*cm)
            side_data= [[img, badge]]
            side_table = Table(side_data, colWidths=[8*cm, 10*cm])
            side_table.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN",  (1, 0), (1, 0),  "CENTER"),
            ]))
            story.append(side_table)
        except Exception:
            story.append(badge)
    else:
        story.append(badge)

    story.append(Spacer(1, 0.3*cm))
    if summary:
        story.append(Paragraph(summary, styles["body"]))
    story.append(Spacer(1, 0.4*cm))

    # Per-axis table
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
    story.append(Paragraph("Per-Axis Breakdown", styles["heading"]))
    story.append(_axis_table(result, styles))

    # Disclaimer
    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
    story.append(Paragraph(
        "<i>This report is generated by an automated QC advisory system and does not "
        "constitute a clinical or diagnostic determination. All image acceptability "
        "decisions remain the responsibility of the reviewing clinician or technologist.</i>",
        ParagraphStyle("disclaimer", fontSize=7, textColor=colors.grey, leading=10),
    ))

    doc.build(story)
    return output_path


def generate_batch_report(
    predictions_csv : str | Path,
    kappa_json      : str | Path,
    calibration_png : str | Path,
    failure_md      : str | Path,
    output_path     : str | Path,
) -> Path:
    """
    Generate a per-batch PDF QA log.

    Args:
        predictions_csv : path to data/predictions/model_v1.csv
        kappa_json      : path to reports/interrater_kappa.json
        calibration_png : path to reports/figures/validation_calibration.png
        failure_md      : path to reports/failure_catalogue.md
        output_path     : where to save the PDF

    Returns:
        Path to the generated PDF
    """
    import pandas as pd

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    preds = pd.read_csv(predictions_csv) if Path(predictions_csv).exists() else pd.DataFrame()

    if Path(kappa_json).exists():
        with open(kappa_json) as f:
            kappa_data = json.load(f)
    else:
        kappa_data = {}

    doc    = SimpleDocTemplate(
        str(output_path),
        pagesize     = A4,
        leftMargin   = 1.5*cm,
        rightMargin  = 1.5*cm,
        topMargin    = 1.5*cm,
        bottomMargin = 1.5*cm,
    )
    styles = _styles()
    story  = []

    # Header
    hdr = Table(
        [[Paragraph("CXR Batch QA Report — MTV-INT-RAD-003", styles["title"])]],
        colWidths=[18*cm],
    )
    hdr.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), _HEADER_BG),
        ("TOPPADDING",    (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
    ]))
    story.append(hdr)
    story.append(Spacer(1, 0.4*cm))

    # Batch statistics
    story.append(Paragraph("Batch Statistics", styles["heading"]))
    if len(preds) > 0:
        flag_counts = preds["overall_flag"].value_counts().to_dict() if "overall_flag" in preds else {}
        score_mean  = preds["composite_score"].mean() if "composite_score" in preds else 0
        score_std   = preds["composite_score"].std()  if "composite_score" in preds else 0

        stats_data = [
            ["Metric", "Value"],
            ["Total studies", str(len(preds))],
            ["Acceptable",    str(flag_counts.get("acceptable", 0))],
            ["Borderline",    str(flag_counts.get("borderline", 0))],
            ["Repeat",        str(flag_counts.get("repeat", 0))],
            ["Mean composite score", f"{score_mean:.2f}"],
            ["Std composite score",  f"{score_std:.2f}"],
        ]
        st = Table(stats_data, colWidths=[8*cm, 8*cm])
        st.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (-1, 0), _HEADER_BG),
            ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
            ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",    (0, 0), (-1, -1), 9),
            ("GRID",        (0, 0), (-1, -1), 0.25, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, _LIGHT_GREY]),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(st)
    story.append(Spacer(1, 0.5*cm))

    # Inter-rater kappa
    story.append(Paragraph("Inter-Rater Agreement (κ)", styles["heading"]))
    if kappa_data:
        ceiling = kappa_data.get("ceiling_kappa")
        n       = kappa_data.get("n_studies", 0)
        kappa_rows = [["Axis", "κ", "Agreement %", "Interpretation"]]
        per_axis = kappa_data.get("per_axis", {})
        for axis, row in per_axis.items():
            kappa_rows.append([
                axis,
                str(row.get("kappa", "N/A")),
                f"{row.get('agreement_pct', 0):.1f}%",
                row.get("interpretation", ""),
            ])
        kappa_rows.append([
            "CEILING (global)", str(ceiling), "", ""
        ])
        kt = Table(kappa_rows, colWidths=[4*cm, 3*cm, 4*cm, 7*cm])
        kt.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (-1, 0), _HEADER_BG),
            ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
            ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",    (0, 0), (-1, -1), 8),
            ("GRID",        (0, 0), (-1, -1), 0.25, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -2), [colors.white, _LIGHT_GREY]),
            ("BACKGROUND",  (0, -1), (-1, -1), colors.HexColor("#e8ecf0")),
            ("FONTNAME",    (0, -1), (-1, -1), "Helvetica-Bold"),
            ("TOPPADDING",  (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING",(0,0), (-1, -1), 3),
        ]))
        story.append(kt)
    story.append(Spacer(1, 0.5*cm))

    # Calibration plot
    if Path(calibration_png).exists():
        story.append(PageBreak())
        story.append(Paragraph("Calibration Plot", styles["heading"]))
        story.append(RLImage(str(calibration_png), width=16*cm, height=8*cm))
        story.append(Spacer(1, 0.4*cm))

    # Failure mode summary
    if Path(failure_md).exists():
        story.append(Paragraph("Failure Mode Summary", styles["heading"]))
        failure_text = Path(failure_md).read_text(encoding="utf-8")
        # Extract just the summary table section
        lines = failure_text.split("\n")
        summary_lines = []
        in_table = False
        for line in lines:
            if "Per-Axis Failure Counts" in line:
                in_table = True
            if in_table:
                summary_lines.append(line)
            if in_table and line.strip() == "" and len(summary_lines) > 5:
                break
        if summary_lines:
            story.append(Paragraph(
                "See reports/failure_catalogue.md for full analysis.",
                styles["body"],
            ))

    # Disclaimer
    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
    story.append(Paragraph(
        "<i>This batch QA report is advisory only. Not for clinical use.</i>",
        ParagraphStyle("disclaimer", fontSize=7, textColor=colors.grey),
    ))

    doc.build(story)
    return output_path