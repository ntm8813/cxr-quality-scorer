# app.py
# streamlit run app.py

from __future__ import annotations

import streamlit as st
import tempfile
import os
import time
import json
from pathlib import Path

st.set_page_config(
    page_title="CXR Quality Scorer — MTV-INT-RAD-003",
    page_icon="🫁",
    layout="wide",
)

_FLAG_ICON = {"acceptable": "🟢", "borderline": "🟡", "repeat": "🔴"}
_FLAG_BG = {"acceptable": "#d4edda", "borderline": "#fff3cd", "repeat": "#f8d7da"}
_FLAG_TEXT = {"acceptable": "#155724", "borderline": "#856404", "repeat": "#721c24"}


@st.cache_resource
def _get_pipeline():
    from src.pipeline import run_pipeline
    return run_pipeline


st.title("🫁 CXR Image Quality Scorer")
st.caption("Medtatvaa Healthcare · MTV-INT-RAD-003 · v1.0 Week 3 MVP")
st.divider()

uploaded = st.file_uploader(
    "Upload a chest radiograph (DICOM or PNG/JPG)",
    type=["dcm", "png", "jpg", "jpeg"],
)

if uploaded is not None:

    suffix = Path(uploaded.name).suffix.lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    with st.spinner("Running quality assessment pipeline..."):
        t0 = time.time()
        try:
            run_pipeline = _get_pipeline()
            result = run_pipeline(tmp_path)
            elapsed = time.time() - t0
            error = None
        except Exception as e:
            result = None
            elapsed = time.time() - t0
            error = str(e)
        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass

    if error:
        st.error(f"Pipeline error: {error}")
        st.stop()

    flag = result.overall_flag
    score = result.composite_score
    score_pct = score * 100.0 if score <= 1.0 else score

    icon = _FLAG_ICON.get(flag, "⚪")
    bg = _FLAG_BG.get(flag, "#f8f9fa")
    tc = _FLAG_TEXT.get(flag, "#000")
    summary = result.metadata_summary.get("summary_rationale", "")

    st.markdown(
        f"""
        <div style="background:{bg};padding:18px 24px;
        border-radius:10px;margin-bottom:16px;">
        <h2 style="margin:0;color:{tc};">{icon} Overall: <b>{flag.upper()}</b></h2>
        <h1 style="margin:4px 0;font-size:2.8em;color:{tc};">
            {score_pct:.1f}
            <span style="font-size:0.35em;color:#666;">/100</span>
        </h1>
        <p style="margin:4px 0;color:#444;">{summary}</p>
        <p style="margin:0;font-size:0.8em;color:#888;">
            Processed in {elapsed:.2f}s · {len(result.axis_results)} axes evaluated
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Per-Axis Breakdown")

    h0, h1, h2, h3 = st.columns([2, 1, 1, 4])
    h0.markdown("**Axis**")
    h1.markdown("**Score**")
    h2.markdown("**Flag**")
    h3.markdown("**Rationale**")

    st.divider()

    for ar in result.axis_results:
        ax_name = ar.axis if isinstance(ar.axis, str) else ar.axis.value
        ax_flag = ar.flag if isinstance(ar.flag, str) else ar.flag.value

        c0, c1, c2, c3 = st.columns([2, 1, 1, 4])
        c0.markdown(f"**{ax_name.capitalize()}**")
        c1.markdown(f"`{ar.score:.3f}`")
        c2.markdown(f"{_FLAG_ICON.get(ax_flag,'⚪')} {ax_flag}")
        c3.markdown(ar.rationale or "—")
        st.divider()

    with st.expander("Raw JSON output"):
        json_str = result.model_dump_json(indent=2)
        st.code(json_str, language="json")

        st.download_button(
            "⬇ Download JSON report",
            data=json_str,
            file_name=f"{result.study_uid}_quality_report.json",
            mime="application/json",
        )