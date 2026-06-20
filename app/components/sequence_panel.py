"""State-of-play panel: play back an attack's build-up and read its numerical balance.

The single-frame models judge a frozen instant; this panel shows the **evolution** the
temporal model (:mod:`models.temporal`) reasons over — the last few on-ball frames before
the trigger, with attackers-minus-defenders ahead of the ball per frame (the developing
3v2). Reads the torch-free ``app/assets/sequences.parquet`` written by ``app.precompute``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from app.components.pitch_viewer import close, sequence_figure

SEQUENCES_PATH = Path("app/assets/sequences.parquet")


@st.cache_data
def load_sequences() -> pd.DataFrame | None:
    """Load the precomputed build-up sequence geometry, if present."""
    if SEQUENCES_PATH.exists():
        return pd.read_parquet(SEQUENCES_PATH)
    return None


def render(match_id: int, possession: int) -> None:
    """Render the build-up playback + superiority trace for one possession."""
    seqs = load_sequences()
    if seqs is None:
        st.info("No sequence cache yet. Run `python -m app.precompute` after training.")
        return
    hit = seqs[(seqs["match_id"] == match_id) & (seqs["possession"] == possession)]
    if hit.empty:
        st.caption("No build-up sequence available for this possession.")
        return

    row = hit.iloc[0]
    frames = json.loads(row["frames_json"])
    superiority = json.loads(row["superiority_json"])
    n = len(frames)

    st.markdown("**State of play — the build-up**")
    st.caption(
        "The last few on-ball frames before the trigger. The temporal model reads this "
        "whole sequence, not just the final snapshot. **Ahead of the ball** = attackers "
        "minus defenders past the ball — a positive, rising number is a developing overload."
    )

    # A compact superiority trace so the whole build-up reads at a glance.
    trace = pd.DataFrame(
        {
            "frame": [f"t-{n - 1 - i}" if i < n - 1 else "trigger" for i in range(n)],
            "ahead of ball": superiority,
        }
    ).set_index("frame")
    st.bar_chart(trace, height=160, color="#10b981")

    if n > 1:
        step = st.slider("Build-up frame", 1, n, n, key=f"seq_{match_id}_{possession}") - 1
    else:
        step = 0
    fig = sequence_figure(frames[step], step, n, superiority[step])
    st.pyplot(fig, use_container_width=True)
    close(fig)
