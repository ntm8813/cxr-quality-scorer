# app.py
# streamlit run app.py

from __future__ import annotations

import streamlit as st
import tempfile
import os
import time
import json
import pandas as pd
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

# =========================
# SINGLE FILE MODE
# =========================
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


# =========================
# BATCH MODE (ADDED BLOCK)
# =========================

st.divider()
st.subheader("📦 Batch Processing")
st.caption("Upload a CSV with a 'path' column. Each row should be an absolute or relative path to a .png or .dcm file.")

batch_csv = st.file_uploader(
    "Upload batch CSV",
    type=["csv"],
    key="batch_uploader",
)

if batch_csv is not None:

    df_batch = pd.read_csv(batch_csv)

    if "path" not in df_batch.columns:
        st.error("CSV must have a column named 'path'.")
        st.stop()

    paths = df_batch["path"].tolist()
    st.info(f"Found {len(paths)} files. Processing...")

    progress_bar = st.progress(0)
    status_text = st.empty()
    results_list = []

    run_pipeline = _get_pipeline()

    for i, fpath in enumerate(paths):
        status_text.text(f"Processing {i+1}/{len(paths)}: {Path(fpath).name}")

        try:
            r = run_pipeline(str(fpath), explain=True)

            flag_v = r.overall_flag
            score_v = r.composite_score
            score_pct = score_v * 100.0 if score_v <= 1.0 else score_v

            row = {
                "file": Path(fpath).name,
                "study_uid": r.study_uid,
                "composite_score": round(score_pct, 1),
                "overall_flag": flag_v,
            }

            for ar in r.axis_results:
                ax = ar.axis if isinstance(ar.axis, str) else ar.axis.value
                row[f"{ax}_score"] = round(ar.score, 3)
                row[f"{ax}_flag"] = ar.flag if isinstance(ar.flag, str) else ar.flag.value

            results_list.append(row)

        except Exception as e:
            results_list.append({
                "file": Path(fpath).name,
                "error": str(e),
            })

        progress_bar.progress((i + 1) / len(paths))

    status_text.text("Batch complete.")
    df_results = pd.DataFrame(results_list)

    c1, c2, c3 = st.columns(3)
    if "overall_flag" in df_results.columns:
        c1.metric("🟢 Acceptable", int((df_results["overall_flag"] == "acceptable").sum()))
        c2.metric("🟡 Borderline", int((df_results["overall_flag"] == "borderline").sum()))
        c3.metric("🔴 Repeat", int((df_results["overall_flag"] == "repeat").sum()))

    st.dataframe(df_results, use_container_width=True)

    st.subheader("Per-Study Detail")

    for row in results_list:
        if "error" in row:
            st.error(f"{row['file']}: {row['error']}")
            continue

        flg = row.get("overall_flag", "")
        icon = _FLAG_ICON.get(flg, "⚪")

        with st.expander(f"{icon} {row['file']} — {flg} ({row.get('composite_score','?')}/100)"):

            axis_scores = {k.replace("_score", ""): v
                           for k, v in row.items() if k.endswith("_score")}

            for ax_name, ax_score in axis_scores.items():
                ax_flag = row.get(f"{ax_name}_flag", "")
                st.markdown(
                    f"**{ax_name}**: `{ax_score:.3f}` "
                    f"{_FLAG_ICON.get(ax_flag,'⚪')} {ax_flag}"
                )

    st.download_button(
        "⬇ Download batch results CSV",
        data=df_results.to_csv(index=False),
        file_name="batch_results.csv",
        mime="text/csv",
    )