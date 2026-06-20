"""The metrics side panel: what the model believes about one possession.

Leads with a plain-language verdict, then the numbers: success probabilities (both
models) against the actual outcome, the xT estimate, the run prediction error (the
"instinct gap"), and the next-receiver candidates as probability bars. Every block
carries a one-line explanation for first-time viewers.
"""

from __future__ import annotations

import numpy as np
import streamlit as st

from app.components.prediction_overlay import key_attacker


def instinct_gap_m(row, model: str = "tfm") -> float:
    """Distance (m) between the model's run prediction and the actual position."""
    dx = float(row[f"{model}_run_x"]) - float(row["run_true_x"])
    dy = float(row[f"{model}_run_y"]) - float(row["run_true_y"])
    return float(np.hypot(dx, dy))


def stay_put_error_m(row) -> float:
    """Error of the 'no-move' baseline: the runner's actual displacement in 1.5 s.

    The baseline predicts the run subject stays where they are, so its error equals
    how far they actually travelled. On the held-out set this averages ~6.46 m — the
    honest reference the (displacement-reparametrised) run head now narrowly beats.
    """
    key = key_attacker(row)
    if key is None:
        return float("nan")
    px = np.asarray(row["px"], dtype=float)
    py = np.asarray(row["py"], dtype=float)
    return float(np.hypot(px[key] - float(row["run_true_x"]), py[key] - float(row["run_true_y"])))


def verdict(row) -> str:
    """One plain sentence: what the model said vs what actually happened."""
    p = float(row["tfm_p_success"])
    success = float(row["y_success"]) > 0.5
    conf = "high" if p >= 0.65 else ("low" if p <= 0.35 else "moderate")
    if success and p >= 0.5:
        return (
            f"The model saw it coming: {conf} confidence ({p:.0%}) and the attack reached threat."
        )
    if not success and p < 0.5:
        return f"The model read the danger correctly: only {p:.0%}, and the attack broke down."
    if success and p < 0.5:
        return f"An upset: the model gave this just {p:.0%}, but the attack reached threat anyway."
    return f"A let-off: the model rated this {p:.0%}, but the defence recovered."


def render(row) -> None:
    """Render the prediction panel for one cached possession row."""
    st.markdown(f"#### {verdict(row)}")
    st.caption(
        '"Reached threat" means a shot within 15 seconds or the ball entering the '
        "penalty box within 10 seconds of this freeze-frame."
    )

    c1, c2 = st.columns(2)
    c1.metric("P(success) - Transformer", f"{row['tfm_p_success']:.0%}")
    c2.metric("P(success) - GAT", f"{row['gat_p_success']:.0%}")
    st.caption(
        "Two independent models score the same frozen moment. They see player geometry "
        "with no names or identities — plus a little context: the play's restart type "
        "(corner, throw-in, counter…) and the clock."
    )

    c3, c4 = st.columns(2)
    c3.metric("Threat added (predicted)", f"{row['tfm_xt']:+.3f}")
    c4.metric("Threat added (actual)", f"{row['y_xt']:+.3f}")
    st.caption(
        "xT progression: how much closer to a goal the possession moved the ball, on a "
        "0-to-~0.4 scale where 0.4 is a shot from the six-yard box."
    )

    st.divider()
    st.markdown("**The run head, honestly**")
    g1, g2, g3 = st.columns(3)
    g1.metric("Transformer off by", f"{instinct_gap_m(row, 'tfm'):.1f} m")
    g2.metric("GAT off by", f"{instinct_gap_m(row, 'gat'):.1f} m")
    g3.metric(
        "If they'd stayed put",
        f"{stay_put_error_m(row):.1f} m",
        help="How far the runner actually moved — the error of simply assuming no movement.",
    )
    st.caption(
        "The white-circled attacker is the most advanced runner; the arrows are each "
        "model's guess for where they will be 1.5 s later, the X is where they actually "
        "went. The head predicts a *displacement* from the runner's position (an earlier "
        "absolute version collapsed to the squad centroid and lost to 'stay put' by ~2×). "
        "After the fix both models **narrowly beat** the ~6.46 m no-move baseline (GAT "
        "6.35 m, hit-within-3 m 0.24 vs 0.22) — but the margin is small, so read the arrow "
        "as a calibrated direction-of-travel prior, not a pinpoint forecast."
    )

    presser = row.get("gat_presser_probs")
    if presser is not None and len(presser):
        top = float(np.max(presser))
        st.divider()
        st.markdown("**Defensive read**")
        d1, d2, d3 = st.columns(3)
        d1.metric("Top presser confidence", f"{top:.0%}")
        if row.get("gat_p_defstop") is not None:
            d2.metric("P(defence wins it back)", f"{float(row['gat_p_defstop']):.0%}")
        if row.get("gat_dxt") is not None:
            d3.metric("Dynamic xT (config threat)", f"{float(row['gat_dxt']):.3f}")
        st.caption(
            "Three defensive/threat reads from the same graph: which defender contests the "
            "next ball (presser softmax), the chance the defence makes a recovery within ~8 s, "
            "and the **Dynamic xT** — a configuration-conditioned threat the head learns well "
            "(R² 0.733) where realized xT could not (R² ≈ 0)."
        )

    probs = row.get("gat_receiver_probs") if hasattr(row, "get") else row["gat_receiver_probs"]
    if probs is not None and len(probs):
        probs = np.asarray(probs, dtype=float)
        top = np.argsort(probs)[::-1][:3]
        st.divider()
        st.markdown("**Who gets the next pass?**")
        px, py = row["px"], row["py"]
        for i in top:
            if i < len(px) and probs[i] > 0:
                st.progress(
                    min(float(probs[i]) * 2.5, 1.0),
                    text=f"player at ({px[i]:.0f}, {py[i]:.0f}) - p = {probs[i]:.2f}",
                )
        st.caption(
            "The model ranks every teammate by how likely they are to receive the next "
            "pass (the rings on the pitch). On open play the real receiver is the model's "
            "top pick 38% / top three 76% — well above the 20% / 59% you'd get by guessing "
            "uniformly, and on par with a simple 'nearest teammate' rule (~42%)."
        )
