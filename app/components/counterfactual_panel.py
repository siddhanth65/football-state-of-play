"""Counterfactual "instinct" panel: where should the most advanced attacker move?

Reads the torch-free ``app/assets/counterfactual.parquet`` (written by ``app.precompute``
via :mod:`eval.counterfactual`) and renders the success-probability surface produced by
sweeping the key attacker, plus the success-maximising move. This is the project's "instinct"
read grounded in the *strong* success head rather than the weak run head.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from app.components.pitch_viewer import close, counterfactual_figure

COUNTERFACTUAL_PATH = Path("app/assets/counterfactual.parquet")


@st.cache_data
def load_counterfactual() -> pd.DataFrame | None:
    """Load the precomputed counterfactual surfaces, if present."""
    if COUNTERFACTUAL_PATH.exists():
        return pd.read_parquet(COUNTERFACTUAL_PATH)
    return None


def render(row: pd.Series) -> None:
    """Render the counterfactual instinct surface for one cached possession row."""
    cf_df = load_counterfactual()
    if cf_df is None:
        st.info("No counterfactual cache yet. Run `python -m app.precompute` after training.")
        return
    hit = cf_df[
        (cf_df["match_id"] == int(row["match_id"]))
        & (cf_df["possession"] == int(row["possession"]))
    ]
    if hit.empty:
        st.caption("No counterfactual surface available for this possession.")
        return

    r = hit.iloc[0]
    cf = {
        "surface": json.loads(r["surface_json"]),
        "dxs": json.loads(r["dxs_json"]),
        "dys": json.loads(r["dys_json"]),
        "base": float(r["base"]),
        "best": float(r["best"]),
        "best_dx": float(r["best_dx"]),
        "best_dy": float(r["best_dy"]),
        "key_index": int(r["key_index"]),
    }
    st.markdown("**Instinct, the other way round — where *should* they run?**")
    st.caption(
        "Predicting where the runner *will* go is weak, so we ask the strong, calibrated "
        "**success** head the inverse: sweep the most advanced attacker across the pitch and "
        "read modelled P(success) at each spot (bright = better). The white ring is their "
        "current position; the arrow points to the success-maximising move."
    )
    fig = counterfactual_figure(row, cf)
    st.pyplot(fig, use_container_width=True)
    close(fig)
    gain = cf["best"] - cf["base"]
    st.metric(
        "Best modelled move",
        f"{cf['best_dx']:+.0f} m x, {cf['best_dy']:+.0f} m y",
        delta=f"{gain:+.1%} P(success)",
        help="Offset that maximises the success head, and the gain over staying put. "
        "Exploratory (TacticAI-style counterfactual), not prescriptive.",
    )
