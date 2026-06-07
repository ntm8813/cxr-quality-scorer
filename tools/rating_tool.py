# tools/rating_tool.py
# Intern A: streamlit run tools/rating_tool.py
# Intern B: streamlit run tools/rating_tool.py  (change Reviewer ID in sidebar)

from __future__ import annotations
import streamlit as st
import pandas as pd
import cv2
from pathlib import Path
from datetime import datetime

st.set_page_config(page_title="CXR Reviewer Rating Tool", layout="wide")

AXES = ["sharpness", "exposure", "rotation", "coverage",
        "inspiration", "artifact", "metadata"]
RATING_LABELS = {
    1: "1 — Acceptable",
    2: "2 — Borderline",
    3: "3 — Repeat",
}

# ── Reviewer identity ──────────────────────────────────────────
reviewer_id = st.sidebar.selectbox("Reviewer ID", [1, 2], index=0)

RATINGS_DIR  = Path("data/ratings")
RATINGS_DIR.mkdir(parents=True, exist_ok=True)
RATINGS_FILE = RATINGS_DIR / f"reviewer_{reviewer_id}.csv"

# ── Load existing ratings ──────────────────────────────────────
if RATINGS_FILE.exists():
    ratings_df = pd.read_csv(RATINGS_FILE)
else:
    ratings_df = pd.DataFrame(
        columns=["study_uid", "timestamp"] + AXES + ["global_rating", "notes"]
    )

# ── Find images ────────────────────────────────────────────────
IMAGE_DIR = Path("data/raw/sample_dicoms")
all_images = sorted(IMAGE_DIR.glob("*.png"))

st.title("🫁 CXR Quality Reviewer Tool")
st.caption(
    f"Reviewer {reviewer_id}  ·  "
    f"{len(all_images)} images  ·  "
    f"{len(ratings_df)} rated"
)

if not all_images:
    st.error(f"No PNG images found in {IMAGE_DIR}. "
             "Add images to data/raw/ first.")
    st.stop()

# ── Filter out already rated ───────────────────────────────────
rated_uids = set(ratings_df["study_uid"].tolist())
unrated    = [p for p in all_images if p.stem not in rated_uids]

st.sidebar.markdown(f"**Progress:** {len(rated_uids)} / {len(all_images)}")
st.sidebar.progress(len(rated_uids) / max(len(all_images), 1))

if not unrated:
    st.success("✅ All images rated!")
    st.download_button(
        "Download completed CSV",
        data      = ratings_df.to_csv(index=False),
        file_name = f"reviewer_{reviewer_id}.csv",
        mime      = "text/csv",
    )
    st.stop()

# ── Show current image ─────────────────────────────────────────
current = unrated[0]
img_arr = cv2.imread(str(current), cv2.IMREAD_GRAYSCALE)

idx_now = len(rated_uids) + 1
st.markdown(
    f"**Image {idx_now} / {len(all_images)}:** `{current.name}`"
)

col_img, col_form = st.columns([1, 1])

with col_img:
    if img_arr is not None:
        st.image(
            img_arr,
            caption=current.stem,
            width="stretch",
            clamp=True,
        )  
    else:
        st.warning(f"Could not load {current.name}")

    with st.expander("Scoring guide"):
        st.markdown("""
**1 — Acceptable:** No significant quality issue. Ready for reading.

**2 — Borderline:** Some degradation present but image may still be usable.
Radiologist review recommended.

**3 — Repeat:** Quality unacceptable. Repeat acquisition required.
        """)

with col_form:
    st.markdown("### Rate each axis")
    ratings = {}
    for axis in AXES:
        ratings[axis] = st.radio(
            axis.capitalize(),
            options     = [1, 2, 3],
            format_func = lambda x: RATING_LABELS[x],
            horizontal  = True,
            key         = f"{axis}_{current.stem}",
        )

    st.divider()
    global_rating = st.radio(
        "**Global overall quality**",
        options     = [1, 2, 3],
        format_func = lambda x: RATING_LABELS[x],
        horizontal  = True,
        key         = f"global_{current.stem}",
    )
    notes = st.text_input("Notes (optional)",
                          key=f"notes_{current.stem}",
                          placeholder="Any observations about this image...")

    col_sub, col_skip = st.columns([1, 1])
    with col_sub:
        if st.button("✅ Submit → Next", type="primary",
                     use_container_width=True):
            new_row = {
                "study_uid"    : current.stem,
                "timestamp"    : datetime.now().isoformat(),
                "global_rating": global_rating,
                "notes"        : notes,
                **ratings,
            }
            ratings_df = pd.concat(
                [ratings_df, pd.DataFrame([new_row])],
                ignore_index=True,
            )
            ratings_df.to_csv(RATINGS_FILE, index=False)
            st.rerun()

    with col_skip:
        if st.button("⏭ Skip", use_container_width=True):
            st.rerun()

# ── Always-available download ──────────────────────────────────
if len(ratings_df) > 0:
    st.sidebar.download_button(
        "💾 Download CSV so far",
        data      = ratings_df.to_csv(index=False),
        file_name = f"reviewer_{reviewer_id}.csv",
        mime      = "text/csv",
    )