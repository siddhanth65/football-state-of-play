"""Football State-of-Play dashboard (CLAUDE.md §10 Week 6).

Five modes:
- **Start here** — a guided walkthrough for first-time viewers.
- **Browse** — competition → match → possession, with highlight quick-picks.
- **Case studies** — the hand-picked famous-final sequences.
- **Compare** — two possessions side by side.
- **Model report** — the headline results, figures, and what they mean.

Run ``streamlit run app/streamlit_app.py``. Requires the prediction cache
(``python -m app.precompute``) — the app itself never imports torch.

Widget options are always scalar keys (match ids, row positions), never row dicts:
the cache rows contain numpy arrays, and Streamlit compares options with ``==`` when
a selection persists across reruns, which raises on arrays.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from app.components import counterfactual_panel, instinct_panel, sequence_panel
from app.components.pitch_viewer import (
    attention_figure,
    before_after_figure,
    close,
    frame_figure,
    team_radar_figure,
)

PREDICTIONS_PATH = Path("app/assets/predictions.parquet")
CASE_STUDIES_PATH = Path("results/case_studies/case_studies.json")
FIGURES_DIR = Path("results/figures")
RESULTS_DIR = Path("results")

ACCENT = "#b5653b"

st.set_page_config(page_title="State of Play", page_icon=":soccer:", layout="wide")

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

      :root {
        --accent: #b5653b; --accent2: #6d7a40; --bg: #f3ede1; --card: #fbf7ee;
        --line: #e4dac4; --ink: #3a352c; --muted: #8c8472;
      }

      html, body, .stApp, [data-testid="stAppViewContainer"], [class*="css"] {
        font-family: 'Outfit', system-ui, -apple-system, sans-serif; color: var(--ink);
      }

      /* Heading hierarchy by weight + colour, not scale (no scream) */
      h1, h2, h3, h4 {
        font-family: 'Outfit', sans-serif; letter-spacing: -0.02em; font-weight: 600;
        color: #2c281f;
      }
      h1 { font-size: 2.0rem !important; letter-spacing: -0.03em; }
      h2 { font-size: 1.38rem !important; }
      h3 { font-size: 1.1rem !important; color: #4a4436; }

      /* Numerals + code in mono (tabular) */
      [data-testid="stMetricValue"], [data-testid="stMetricDelta"],
      [data-testid="stDataFrame"] td, code {
        font-family: 'JetBrains Mono', ui-monospace, monospace; font-variant-numeric: tabular-nums;
      }
      [data-testid="stMetricValue"] {
        font-size: 1.5rem; font-weight: 500; letter-spacing: -0.01em; color: #2c281f;
      }
      [data-testid="stMetricLabel"] {
        color: var(--muted); font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.05em;
      }
      code { background: #ece2cc; border-radius: 4px; }

      /* Layout rhythm */
      .block-container { padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1320px; }
      [data-testid="stSidebar"] { border-right: 1px solid var(--line); background: #efe7d6; }
      hr { border-color: var(--line); margin: 1.4rem 0; }

      /* Metric cards: warm paper, soft shadow, terracotta accent bar */
      [data-testid="stMetric"] {
        background: var(--card); border: 1px solid var(--line);
        border-left: 3px solid var(--accent); border-radius: 16px; padding: 0.85rem 1.1rem;
        box-shadow: 0 14px 30px -20px rgba(120,90,50,0.30), inset 0 1px 0 rgba(255,255,255,0.6);
      }

      /* Tabs: single terracotta active accent */
      button[data-baseweb="tab"] { font-size: 0.92rem; font-weight: 500; letter-spacing: -0.01em; }
      button[data-baseweb="tab"][aria-selected="true"] { color: var(--accent) !important; }
      [data-baseweb="tab-highlight"] { background-color: var(--accent) !important; }

      /* Expanders + dataframes: quiet warm containers */
      [data-testid="stExpander"] {
        border: 1px solid var(--line); border-radius: 14px; background: var(--card);
      }
      [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 12px; }

      /* Hero stat chips */
      .chips { display: flex; gap: 0.55rem; flex-wrap: wrap; margin: 0.6rem 0 0.3rem; }
      .chip {
        border: 1px solid var(--line); background: rgba(255,255,255,0.55); border-radius: 999px;
        padding: 0.3rem 0.9rem; font-size: 0.8rem; color: #5a5347; white-space: nowrap;
      }
      .chip b { color: var(--accent); font-family: 'JetBrains Mono', monospace; }

      /* Tactile buttons */
      .stButton > button {
        border-radius: 10px; border: 1px solid var(--line); background: var(--card);
        color: var(--ink); transition: transform .12s ease;
      }
      .stButton > button:active { transform: translateY(1px); }
      [data-testid="stCaptionContainer"] { color: #93897a; }

      /* Warm page wash (top corners) */
      [data-testid="stAppViewContainer"] {
        background:
          radial-gradient(90% 55% at 8% -5%, rgba(181,101,59,0.10) 0%, rgba(181,101,59,0) 55%),
          radial-gradient(80% 50% at 100% -8%, rgba(109,122,64,0.08) 0%, rgba(109,122,64,0) 50%),
          var(--bg);
      }

      /* Hero banner — warm golden-hour gradient */
      .hero {
        position: relative; border-radius: 22px; padding: 2.1rem 2.3rem 1.5rem; margin: 0 0 1.3rem;
        overflow: hidden; border: 1px solid #e8dcc2;
        background:
          radial-gradient(130% 150% at 0% 0%, rgba(181,101,59,0.22) 0%, rgba(181,101,59,0) 48%),
          radial-gradient(120% 140% at 100% 8%, rgba(109,122,64,0.16) 0%, rgba(109,122,64,0) 46%),
          linear-gradient(180deg, #fbf6ea 0%, #f1e7d2 100%);
        box-shadow: 0 24px 48px -30px rgba(120,90,50,0.40), inset 0 1px 0 rgba(255,255,255,0.7);
      }
      .hero-kicker {
        font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; font-weight: 500;
        letter-spacing: 0.24em; color: var(--accent); margin-bottom: 0.35rem;
      }
      .hero-title {
        font-size: 2.8rem !important; font-weight: 700; letter-spacing: -0.04em;
        line-height: 1.0; margin: 0 0 0.45rem; color: #2c281f;
      }
      .hero-sub { color: #6a6253; font-size: 0.97rem; max-width: 64ch; margin: 0 0 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- data ---------------------------------------------------------------------
@st.cache_data
def load_cache() -> pd.DataFrame:
    """Load the precomputed prediction cache."""
    return pd.read_parquet(PREDICTIONS_PATH)


@st.cache_data
def load_case_studies() -> list[dict]:
    """Load the curated case-study list, if built."""
    if CASE_STUDIES_PATH.exists():
        return json.loads(CASE_STUDIES_PATH.read_text(encoding="utf-8"))
    return []


@st.cache_data
def load_csv(name: str) -> pd.DataFrame | None:
    """Load a results table if present."""
    p = RESULTS_DIR / name
    return pd.read_csv(p) if p.exists() else None


# --- labels -------------------------------------------------------------------
def minute_of(row) -> int:
    """Match minute from period-relative seconds (handles extra time)."""
    offset = {1: 0, 2: 45, 3: 90, 4: 105}.get(int(row["period"]), 0)
    return int(float(row["trigger_time_s"]) // 60) + offset


def match_label(df: pd.DataFrame, match_id: int) -> str:
    """Selectbox label for a match id."""
    r = df[df["match_id"] == match_id].iloc[0]
    return f"{r['home_team']} vs {r['away_team']}  ({r['match_date']})"


def possession_label(df: pd.DataFrame, idx: int) -> str:
    """Selectbox label for a cache row position."""
    r = df.loc[idx]
    tag = "counter" if r["from_counter"] else str(r["play_pattern"]).lower()
    return f"{minute_of(r)}'  ·  {tag}  ·  model says {r['tfm_p_success']:.0%}"


def figure_title(row) -> str:
    """In-figure title strip."""
    return f"{row['home_team']} vs {row['away_team']}  ·  {minute_of(row)}'  ·  {row['match_date']}"


# --- pickers (scalar options only — see module docstring) ----------------------
HIGHLIGHTS = {
    "Pick my own": None,
    "Most threatening attack": ("tfm_p_success", False),
    "Best counter-attack": ("counter", False),
    "Biggest upset (model said no, attack scored through)": ("upset", False),
    "Model's safest call": ("safe", False),
}


def apply_highlight(poss: pd.DataFrame, choice: str) -> int | None:
    """Resolve a highlight quick-pick to a row index, or None for manual."""
    if HIGHLIGHTS.get(choice) is None:
        return None
    if choice == "Most threatening attack":
        return int(poss["tfm_p_success"].idxmax())
    if choice == "Best counter-attack":
        c = poss[poss["from_counter"]]
        return int(c["tfm_p_success"].idxmax()) if not c.empty else None
    if choice == "Biggest upset (model said no, attack scored through)":
        u = poss[poss["y_success"] > 0.5]
        return int(u["tfm_p_success"].idxmin()) if not u.empty else None
    if choice == "Model's safest call":
        correct = poss[(poss["tfm_p_success"] < 0.5) == (poss["y_success"] < 0.5)]
        if correct.empty:
            return None
        return int((correct["tfm_p_success"] - 0.5).abs().idxmax())
    return None


def pick_possession(df: pd.DataFrame, key: str, highlights: bool = False) -> pd.Series | None:
    """Competition → match → possession picker. Returns the chosen row."""
    comps = sorted(df["competition_name"].dropna().unique())
    comp = st.selectbox("Competition", comps, key=f"{key}-comp")
    sub = df[df["competition_name"] == comp]

    match_ids = (
        sub.drop_duplicates("match_id").sort_values("match_date")["match_id"].astype(int).tolist()
    )
    mid = st.selectbox(
        "Match", match_ids, format_func=lambda m: match_label(sub, m), key=f"{key}-match"
    )
    poss = sub[sub["match_id"] == mid].sort_values("trigger_time_s")
    if poss.empty:
        st.info("No held-out possessions for this match.")
        return None

    idx = None
    if highlights:
        choice = st.radio(
            "Quick picks",
            list(HIGHLIGHTS),
            horizontal=True,
            key=f"{key}-hl",
            label_visibility="collapsed",
        )
        idx = apply_highlight(poss, choice)
        if idx is None and choice != "Pick my own":
            st.caption("No possession of that kind in this match; pick manually below.")
    if idx is None:
        idx = st.selectbox(
            "Possession",
            poss.index.tolist(),
            format_func=lambda i: possession_label(poss, i),
            key=f"{key}-poss",
        )
    return df.loc[idx]


# --- shared view ----------------------------------------------------------------
def show_possession(row: pd.Series, show_control: bool, models: tuple[str, ...]) -> None:
    """Before/after pitch pair + instinct panel for one possession."""
    st.markdown(f"###### {figure_title(row)}")
    left, right = st.columns([2.25, 1.0], gap="large")
    with left:
        fig = before_after_figure(row, models=models, show_control=show_control)
        st.pyplot(fig, use_container_width=True)
        close(fig)
        st.caption(
            "Left = the frozen moment the model scores. Right = the real player positions "
            "1.5 s later (StatsBomb 360 records coordinates, not video). The solid arrow is "
            "the model's predicted run; the dashed white arrow and X are what actually "
            "happened."
        )
        with st.expander("Read the single frame in detail (receiver rings, full pitch control)"):
            detail = frame_figure(
                row, show_control=show_control, models=models, title=figure_title(row)
            )
            st.pyplot(detail, use_container_width=True)
            close(detail)
            st.markdown(
                "- **Emerald dots** attackers, **grey** defenders, **star** the player on "
                "the ball, **white circle** the most advanced runner.\n"
                "- **Solid coloured arrow + diamond** = where the model expects the runner "
                "in 1.5 s. **Dashed white arrow + X** = where they actually went. The label "
                "between them is the miss, in metres.\n"
                "- **Amber rings** mark the model's most likely next-pass receivers "
                "(bigger = more likely).\n"
                "- **Pitch control**: brighter green = ground the attacking team would win "
                "a race to (Spearman 2018)."
            )
        with st.expander("State of play — play back the build-up (the temporal view)"):
            sequence_panel.render(int(row["match_id"]), int(row["possession"]))
        with st.expander("Instinct — where *should* the runner move? (counterfactual)"):
            counterfactual_panel.render(row)
        imp = row.get("gat_node_importance")
        if imp is not None and len(imp):
            with st.expander("Why — which players the model focused on (attention)"):
                afig = attention_figure(row, imp)
                st.pyplot(afig, use_container_width=True)
                close(afig)
                st.caption(
                    "Player marker size = the GAT's attention weight on that player (summed "
                    "across message-passing layers). The white ring marks the most-attended "
                    "player — a peek inside the black box, not a causal claim."
                )
    with right:
        instinct_panel.render(row)


# --- modes -----------------------------------------------------------------------
def start_here(df: pd.DataFrame, show_control: bool, models: tuple[str, ...]) -> None:
    """Guided walkthrough for first-time viewers."""
    st.subheader("What you are looking at, in four steps")
    s1, s2, s3, s4 = st.tabs(
        ["1 · The data", "2 · The model", "3 · The predictions", "4 · How good is it?"]
    )
    with s1:
        st.markdown(
            "Every time something happens in a match, StatsBomb's **360 data** records a "
            "snapshot of every visible player's position - a *freeze-frame*. This project "
            "collected **11,841 open-play attacking-third possessions** from **426 matches** "
            "across World Cups, Euros and the Bundesliga, men's and women's (set-pieces are "
            "excluded). Each snapshot is anonymous geometry: dots on a pitch, with no names."
        )
    with s2:
        st.markdown(
            "Each freeze-frame becomes a **graph**: players are nodes, and every pair of "
            "players is connected by an edge that knows the distance between them and how "
            "open the passing lane is (how far the nearest defender is from the line "
            "between them). Two neural networks read this graph - a **graph attention "
            "network** and a **transformer** - and each produces a compact summary of the "
            "situation: the *state of play*."
        )
    with s3:
        st.markdown(
            "From that summary, each model answers three questions about the next few "
            "seconds:\n\n"
            "1. **Will this attack reach threat?** A shot within 15 s or the ball in the "
            "box within 10 s.\n"
            "2. **How much threat is added?** Measured in xT, the standard "
            "expected-threat scale.\n"
            "3. **Where is the most advanced attacker about to run?** The *instinct* "
            "question. After reparametrising the head to predict a *displacement* (an earlier "
            "absolute version lost to 'stay put' by ~2×), both models now narrowly beat the "
            "no-move baseline — but the margin is small, so it is shown as a calibrated "
            "direction-of-travel prior, not a pinpoint forecast."
        )
        st.markdown("Here is a live example - Argentina's counter in the World Cup final:")
        cases = load_case_studies()
        example = None
        if cases:
            c = cases[0]
            sel = df[(df["match_id"] == c["match_id"]) & (df["possession"] == c["possession"])]
            example = sel.iloc[0] if not sel.empty else None
        if example is None:
            example = df.loc[df["tfm_p_success"].idxmax()]
        show_possession(example, show_control, models)
    with s4:
        st.markdown(
            "On **1,924 open-play possessions from matches the models never saw**, both "
            "models call success at **~0.74 AUC** (multi-seed: transformer 0.746, GAT 0.733; "
            "a coin flip is 0.5, the best non-learned baseline is 0.699) and their "
            "probabilities are honest: situations they rate 70% succeed about 70% of the "
            "time. (An earlier draft reported 0.797, but that pooled in easier set-pieces.) "
            "Full numbers live in the **Model report** tab."
        )
        rel = FIGURES_DIR / "reliability_success.png"
        if rel.exists():
            st.image(str(rel), caption="Calibration: predicted probability vs reality.")


def browse_mode(df: pd.DataFrame, show_control: bool, models: tuple[str, ...]) -> None:
    """Mode: competition → match → possession browser with quick-picks."""
    st.subheader("Browse held-out possessions")
    st.caption(
        "Every possession here comes from a match the models never trained on. "
        "Use the quick picks to jump to the dramatic ones."
    )
    row = pick_possession(df, key="browse", highlights=True)
    if row is not None:
        show_possession(row, show_control, models)


def case_study_mode(df: pd.DataFrame, show_control: bool, models: tuple[str, ...]) -> None:
    """Mode: the hand-picked case-study sequences."""
    st.subheader("Case studies")
    cases = load_case_studies()
    if not cases:
        st.warning("No case studies built yet. Run `python -m eval.case_studies`.")
        return
    titles = [c["title"] for c in cases]
    chosen = st.selectbox("Sequence", range(len(cases)), format_func=lambda i: titles[i])
    case = cases[chosen]
    sel = df[(df["match_id"] == case["match_id"]) & (df["possession"] == case["possession"])]
    if sel.empty:
        st.error("This case study is missing from the prediction cache. Re-run app.precompute.")
        return
    st.markdown(f"**{case['title']}**")
    st.markdown(case["note"])
    show_possession(sel.iloc[0], show_control, models)


def compare_mode(df: pd.DataFrame, show_control: bool, models: tuple[str, ...]) -> None:
    """Mode: two possessions side by side."""
    st.subheader("Compare two possessions")
    st.caption("Put a counter next to a slow build-up and watch the probabilities move.")
    c1, c2 = st.columns(2, gap="large")
    for col, key in ((c1, "left"), (c2, "right")):
        with col:
            row = pick_possession(df, key=f"cmp-{key}")
            if row is not None:
                fig = frame_figure(
                    row, show_control=show_control, models=models, title=figure_title(row)
                )
                st.pyplot(fig, use_container_width=True)
                close(fig)
                m1, m2 = st.columns(2)
                m1.metric("P(success) - Transformer", f"{row['tfm_p_success']:.0%}")
                m2.metric("Actual", "reached threat" if row["y_success"] > 0.5 else "broke down")


_ATTACK_AXES = ["attack_success", "attack_dxt", "option_richness", "counter_share"]
_DEFEND_AXES = ["solidity", "recovery_rate", "press_decisiveness", "lane_suppression"]


def team_identity_mode() -> None:
    """Mode: per-team attacking + defensive fingerprints derived from the learned relations."""
    st.subheader("Team identity — from the relations")
    table = load_csv("team_metrics.csv")
    if table is None:
        st.warning("No team metrics yet. Run `python -m eval.team_metrics`.")
        return
    st.markdown(
        "Each team's **fingerprint**, aggregated from the model's per-possession reads of the "
        "*relations* between players — attacking metrics over their own possessions, defensive "
        "metrics over possessions where they defend. This is the supervisor's *'derive "
        "attacking and defensive team metrics from the relations'* — no new model, just the "
        "heads aggregated per team."
    )
    axes = _ATTACK_AXES + _DEFEND_AXES
    default = table.head(2)["team"].tolist()
    teams = st.multiselect("Teams to compare", table["team"].tolist(), default=default)
    if teams:
        fig = team_radar_figure(table, teams, axes)
        st.pyplot(fig, use_container_width=False)
        close(fig)
        st.caption(
            "Axes are min-max normalised across all teams, so the shape is *relative* identity. "
            "Attacking (top): chance creation, threat afforded (Dynamic-xT), passing-option "
            "richness, counter share. Defensive (bottom): solidity, ball-recovery rate, press "
            "decisiveness, lane suppression. (Line height is in the table; lower x = higher line.)"
        )
    st.markdown("**All teams** (≥30 possessions), sortable:")
    st.dataframe(table.round(3), use_container_width=True, hide_index=True)
    st.caption(
        "An honest caveat: these are *model-derived* tendencies on StatsBomb 360 (mostly "
        "international sides + Bayer Leverkusen), averaged per possession — a relational "
        "fingerprint, not an official rating."
    )


def report_mode() -> None:
    """Mode: headline results with plain-language readings."""
    st.subheader("Model report")
    final = load_csv("final_table.csv")
    if final is not None:
        st.markdown("**Headline table** - held-out test set, higher AUC is better:")
        nice = final[["model", "success_auc", "success_brier", "xt_rmse", "run_rmse_m"]].round(3)
        st.dataframe(nice, use_container_width=True, hide_index=True)
        st.caption(
            "Both learned models beat every baseline on **success** (~0.73 AUC vs 0.699). "
            "After reparametrising the run head to predict a *displacement*, both also "
            "narrowly beat the 6.46 m no-move baseline on the run task (GAT 6.35 m). The GAT "
            "adds the receiver head (the rings); single-seed numbers shown, error bars below."
        )

    c1, c2 = st.columns(2, gap="large")
    with c1:
        rel = FIGURES_DIR / "reliability_success.png"
        if rel.exists():
            st.image(str(rel), use_container_width=True)
            st.caption(
                "**Calibration.** Points on the dashed line mean the probabilities are "
                "honest: what the model calls 70% happens about 70% of the time."
            )
    with c2:
        hit = FIGURES_DIR / "run_hitrate_curve.png"
        if hit.exists():
            st.image(str(hit), use_container_width=True)
            st.caption(
                "**The run head, honestly.** It originally lost to 'predict no movement' by "
                "~2× (an absolute-position head collapses to the squad centroid). "
                "Reparametrised to predict a *displacement* from the runner's own position, "
                "both models now narrowly beat the ~6.46 m no-move baseline — but the margin "
                "is small, so the dashboard frames runs as a calibrated *instinct* direction, "
                "not a bullseye."
            )

    slices = load_csv("slices.csv")
    if slices is not None:
        st.markdown("**Where is the model strongest?**")
        nice = slices[["model", "slice", "n", "success_auc", "success_brier"]].round(3)
        st.dataframe(nice, use_container_width=True, hide_index=True)
        st.caption(
            "Set-pieces are now excluded, so open_play is the full test set. Excluding them "
            "is exactly why the headline is ~0.74, not the 0.797 an earlier draft reported on "
            "the easier pooled data. The set-piece and high-stakes slices disappear because "
            "high box-density moments were overwhelmingly set-pieces; counters (n=49) are too "
            "few to read on their own."
        )

    abl = load_csv("ablations.csv")
    if abl is not None:
        with st.expander("Ablations - which ingredients matter?"):
            st.dataframe(
                abl[["variant", "success_auc", "run_rmse_m"]].round(3),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "Graph-level context and velocity each help the GAT a little; matched width "
                "(hidden 128) recovers part of the GAT-vs-transformer gap, the rest is "
                "architectural. Note the gaps are small (1-2 AUC points) and from a single "
                "seed - directional, not yet statistically nailed down."
            )


# --- shell -----------------------------------------------------------------------
def main() -> None:
    """Dashboard entry point."""
    if not PREDICTIONS_PATH.exists():
        st.error(
            "Prediction cache not found. Build it first: `python -m app.precompute` "
            "(needs trained checkpoints in results/checkpoints/)."
        )
        st.stop()
    df = load_cache()
    n_test = int(df["in_test"].sum()) if "in_test" in df else len(df)

    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-kicker">RELATIONAL FOOTBALL MODEL</div>
          <div class="hero-title">State of Play</div>
          <div class="hero-sub">One frozen moment of an attack &mdash; two neural networks &mdash;
            three reads: <b>will it come off, how dangerous is it, and where is the runner
            going?</b></div>
          <div class="chips">
            <span class="chip"><b>426</b>&nbsp;matches</span>
            <span class="chip"><b>11,841</b>&nbsp;possessions</span>
            <span class="chip"><b>{n_test:,}</b>&nbsp;unseen here</span>
            <span class="chip"><b>0.746</b>&nbsp;success AUC</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### View")
        mode = st.radio(
            "Mode",
            ["Start here", "Browse", "Case studies", "Compare", "Team identity", "Model report"],
            label_visibility="collapsed",
        )
        st.markdown("### Overlays")
        show_control = st.toggle(
            "Pitch control surface",
            value=True,
            help="Brighter green = ground the attacking team would win a race to "
            "(Spearman 2018 pitch control).",
        )
        which = st.multiselect(
            "Run predictions",
            ["Transformer", "GAT"],
            default=["Transformer"],
            help="Which model's expected run to draw for the white-circled attacker.",
        )
        models = tuple(m for m, label in (("tfm", "Transformer"), ("gat", "GAT")) if label in which)
        st.divider()
        st.caption(
            "All possessions are real moments from World Cups, Euros and the Bundesliga, "
            "men's and women's. The models see anonymous player geometry plus a little "
            "context (restart type and the clock) — no names or identities."
        )

    if mode == "Start here":
        start_here(df, show_control, models)
    elif mode == "Browse":
        browse_mode(df, show_control, models)
    elif mode == "Case studies":
        case_study_mode(df, show_control, models)
    elif mode == "Compare":
        compare_mode(df, show_control, models)
    elif mode == "Team identity":
        team_identity_mode()
    else:
        report_mode()


main()
