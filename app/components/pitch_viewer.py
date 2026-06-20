"""Dark-themed freeze-frame rendering for the dashboard.

Draws a cached possession row (from ``app/assets/predictions.parquet``): the BEFORE
frame (the frozen moment the model sees, with the prediction overlay) and a real
BEFORE/AFTER pair (trigger frame next to the genuine player layout 1.5 s later).
Pure numpy + matplotlib + mplsoccer — no torch.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mplsoccer import Pitch

from app.components.prediction_overlay import (
    ATTACKER,
    CARRIER,
    DEFENDER,
    MODEL_COLORS,
    overlay_actual_movement,
    overlay_control_surface,
    overlay_predictions,
)

BG = "#fbf7ee"  # warm card/paper: the figure surround blends into the dashboard cards
CARD = "#fbf7ee"
PITCH_COLOR = "#33503c"  # rich forest green: the focal panel on the warm-light UI
LINE_COLOR = "#cfe0c4"  # soft cream-green pitch markings
TEXT = "#3a352c"  # warm charcoal (titles/legend sit on the cream surround)
MUTED = "#8c8472"


def _pitch() -> Pitch:
    return Pitch(
        pitch_type="statsbomb",
        pitch_color=PITCH_COLOR,
        line_color=LINE_COLOR,
        linewidth=1.25,
        line_zorder=2,
    )


def _glow(pitch, ax, px, py, color, size, alpha, zorder=3):
    """Soft halo behind a set of markers, so players 'pop' off the pitch."""
    if len(px):
        pitch.scatter(px, py, ax=ax, c=color, s=size, alpha=alpha, edgecolors="none", zorder=zorder)


def _draw_players(pitch, ax, px, py, teammate, actor) -> None:
    """Players with soft glows: emerald attackers, slate defenders, glowing white carrier."""
    px, py = np.asarray(px, float), np.asarray(py, float)
    teammate, actor = np.asarray(teammate, bool), np.asarray(actor, bool)
    atk, dfn = px[teammate & ~actor], py[teammate & ~actor]
    dx, dy = px[~teammate], py[~teammate]
    # Halos first (depth), then crisp markers on top.
    _glow(pitch, ax, atk, dfn, ATTACKER, 560, 0.16)
    _glow(pitch, ax, dx, dy, DEFENDER, 480, 0.12)
    pitch.scatter(
        atk, dfn, ax=ax, c=ATTACKER, s=200, edgecolors="#5c2f24", linewidths=1.1, zorder=4
    )
    pitch.scatter(
        dx, dy, ax=ax, c=DEFENDER, s=190, edgecolors="#5a6470", linewidths=1.1, zorder=4
    )
    if actor.any():
        ax_, ay_ = px[actor], py[actor]
        _glow(pitch, ax, ax_, ay_, "#fff6e8", 900, 0.28, zorder=4)
        pitch.scatter(
            ax_, ay_, ax=ax, c=CARRIER, s=360, marker="*",
            edgecolors="#26352b", linewidths=0.8, zorder=5,
        )


def _direction_cue(ax) -> None:
    ax.annotate(
        "attacking",
        xy=(72.0, 82.2),
        xytext=(50.0, 82.2),
        color=MUTED,
        fontsize=7.5,
        va="center",
        arrowprops={"arrowstyle": "-|>", "color": MUTED, "lw": 1.0},
        annotation_clip=False,
    )


def _legend(ax, models, predictions: bool) -> None:
    h = [
        Line2D([], [], marker="o", ls="", mfc=ATTACKER, mec="none", ms=9, label="attacker"),
        Line2D([], [], marker="o", ls="", mfc=DEFENDER, mec="none", ms=9, label="defender"),
        Line2D([], [], marker="*", ls="", mfc=CARRIER, mec="none", ms=13, label="on the ball"),
    ]
    if predictions:
        for m in models:
            h.append(
                Line2D(
                    [],
                    [],
                    color=MODEL_COLORS[m],
                    lw=2.2,
                    marker="D",
                    ms=6,
                    label=f"predicted run ({m.upper()})",
                )
            )
        h.append(
            Line2D(
                [],
                [],
                color="#7a7264",
                lw=2.2,
                ls=(0, (4, 2)),
                marker="X",
                ms=8,
                label="actual run, +1.5 s",
            )
        )
    leg = ax.legend(
        handles=h,
        loc="upper left",
        ncol=len(h),
        frameon=False,
        fontsize=7.5,
        bbox_to_anchor=(0.0, -0.02),
        handletextpad=0.4,
        columnspacing=1.0,
    )
    for t in leg.get_texts():
        t.set_color(TEXT)


def frame_figure(
    row,
    show_control=False,
    show_predictions=True,
    models=("tfm",),
    title=None,
    legend=True,
    figsize=(9.6, 6.7),
):
    """The BEFORE frame: the frozen moment plus the prediction overlay."""
    pitch = _pitch()
    fig, ax = pitch.draw(figsize=figsize)
    fig.patch.set_facecolor(BG)
    if show_control:
        overlay_control_surface(ax, row)
    _draw_players(pitch, ax, row["px"], row["py"], row["teammate"], row["actor"])
    if show_predictions:
        overlay_predictions(ax, pitch, row, models=models)
    _direction_cue(ax)
    if title:
        ax.set_title(title, color=TEXT, fontsize=10.5, pad=10, loc="left")
    if legend:
        _legend(ax, models, show_predictions)
    return fig


def before_after_figure(row, models=("tfm",), show_control=True, figsize=(13.4, 5.0)):
    """Two pitches: the trigger frame (with prediction) and the real frame 1.5 s later."""
    pitch = _pitch()
    fig, (axb, axa) = plt.subplots(1, 2, figsize=figsize)
    fig.patch.set_facecolor(BG)
    pitch.draw(ax=axb)
    pitch.draw(ax=axa)

    # BEFORE
    if show_control:
        overlay_control_surface(axb, row)
    _draw_players(pitch, axb, row["px"], row["py"], row["teammate"], row["actor"])
    overlay_predictions(axb, pitch, row, models=models)
    axb.set_title("NOW  ·  the moment the model sees", color=TEXT, fontsize=10, pad=8, loc="left")

    # AFTER (real positions ~1.5 s later)
    if bool(row.get("has_after", False)) and len(row["after_px"]):
        _draw_players(
            pitch, axa, row["after_px"], row["after_py"], row["after_teammate"], row["after_actor"]
        )
        overlay_actual_movement(axa, pitch, row)
        axa.set_title(
            "1.5 s LATER  ·  what actually happened", color=TEXT, fontsize=10, pad=8, loc="left"
        )
    else:
        axa.text(60, 40, "no +1.5 s frame available", color=MUTED, ha="center", fontsize=10)
        axa.set_title("1.5 s LATER", color=TEXT, fontsize=10, pad=8, loc="left")

    for ax in (axb, axa):
        _direction_cue(ax)
    return fig


def sequence_figure(frame: dict, step: int, n_steps: int, superiority: int, figsize=(9.6, 6.7)):
    """One build-up frame for the state-of-play panel, with the numerical-superiority read.

    ``frame`` is one entry of the decoded ``frames_json`` list (px/py/teammate/actor);
    ``superiority`` is attackers-minus-defenders ahead of the ball for this frame.
    """
    pitch = _pitch()
    fig, ax = pitch.draw(figsize=figsize)
    fig.patch.set_facecolor(BG)
    _draw_players(pitch, ax, frame["px"], frame["py"], frame["teammate"], frame["actor"])
    _direction_cue(ax)
    when = "trigger" if step == n_steps - 1 else f"trigger − {n_steps - 1 - step}"
    sign = "+" if superiority > 0 else ""
    color = "#a85d2e" if superiority > 0 else ("#4f7fa3" if superiority < 0 else MUTED)
    ax.set_title(
        f"build-up frame {step + 1}/{n_steps}  ·  {when}",
        color=TEXT,
        fontsize=10.5,
        pad=10,
        loc="left",
    )
    ax.text(
        1.0,
        82.2,
        f"ahead of the ball:  {sign}{superiority}",
        color=color,
        fontsize=9.5,
        va="center",
        fontweight="bold",
    )
    return fig


def counterfactual_figure(row, cf: dict, figsize=(9.6, 6.7)):
    """Success-probability surface as the key attacker is swept, over the freeze frame.

    ``cf`` decodes ``app/assets/counterfactual.parquet``: ``surface`` (dys x dxs), the
    ``dxs``/``dys`` offset axes (m), ``key_index``, and the success-maximising ``best_dx``/
    ``best_dy``. The heatmap is anchored at the key attacker's current position.
    """
    import numpy as np

    pitch = _pitch()
    fig, ax = pitch.draw(figsize=figsize)
    fig.patch.set_facecolor(BG)

    px = np.asarray(row["px"], float)
    py = np.asarray(row["py"], float)
    k = int(cf["key_index"])
    kx, ky = float(px[k]), float(py[k])
    dxs, dys = np.asarray(cf["dxs"], float), np.asarray(cf["dys"], float)
    surface = np.asarray(cf["surface"], float)
    extent = [kx + dxs.min(), kx + dxs.max(), ky + dys.min(), ky + dys.max()]
    im = ax.imshow(
        surface,
        extent=extent,
        origin="lower",
        cmap="magma",
        alpha=0.72,
        aspect="auto",
        zorder=1,
        vmin=float(surface.min()),
        vmax=float(surface.max()),
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("P(success) if the runner moves here", color=TEXT, fontsize=8)
    cbar.ax.tick_params(colors=MUTED, labelsize=7)

    _draw_players(pitch, ax, px, py, row["teammate"], row["actor"])
    # Current position (ring) and the success-maximising move (arrow + star).
    pitch.scatter(
        [kx], [ky], ax=ax, s=320, facecolors="none", edgecolors="#f4f4f5", linewidths=1.8, zorder=6
    )
    bx, by = kx + float(cf["best_dx"]), ky + float(cf["best_dy"])
    if abs(cf["best_dx"]) > 1e-6 or abs(cf["best_dy"]) > 1e-6:
        ax.annotate(
            "",
            xy=(bx, by),
            xytext=(kx, ky),
            arrowprops={"arrowstyle": "-|>", "color": "#f4f4f5", "lw": 2.2},
            zorder=7,
        )
        pitch.scatter(
            [bx], [by], ax=ax, s=260, marker="*", c="#f4f4f5", edgecolors="#111",
            linewidths=0.6, zorder=8,
        )
    _direction_cue(ax)
    ax.set_title(
        f"Where should the most advanced attacker move?  "
        f"({cf['base']:.0%} → {cf['best']:.0%})",
        color=TEXT,
        fontsize=10.5,
        pad=10,
        loc="left",
    )
    return fig


def attention_figure(row, importance, figsize=(9.6, 6.7)):
    """Freeze frame with players sized by the GAT's attention importance (explainability)."""
    import numpy as np

    pitch = _pitch()
    fig, ax = pitch.draw(figsize=figsize)
    fig.patch.set_facecolor(BG)
    px = np.asarray(row["px"], float)
    py = np.asarray(row["py"], float)
    teammate = np.asarray(row["teammate"], bool)
    actor = np.asarray(row["actor"], bool)
    imp = np.asarray(importance, float)
    if imp.size != px.size or imp.sum() == 0:
        imp = np.ones(px.size) / max(px.size, 1)
    sizes = 90 + 1400 * (imp / imp.max())  # marker area scaled to importance
    for mask, color, edge in ((teammate, ATTACKER, "#04261d"), (~teammate, DEFENDER, "#1f2025")):
        pitch.scatter(
            px[mask], py[mask], ax=ax, s=sizes[mask], c=color, edgecolors=edge,
            linewidths=1.0, zorder=4, alpha=0.9,
        )
    if actor.any():
        pitch.scatter(px[actor], py[actor], ax=ax, c=CARRIER, s=320, marker="*", zorder=5)
    top = int(np.argmax(imp))
    pitch.scatter(
        [px[top]], [py[top]], ax=ax, s=float(sizes[top]) + 220, facecolors="none",
        edgecolors="#f4f4f5", linewidths=1.8, zorder=6,
    )
    _direction_cue(ax)
    ax.set_title(
        "What the model focused on (attention)  ·  bigger = more attended",
        color=TEXT, fontsize=10.5, pad=10, loc="left",
    )
    return fig


def team_radar_figure(table, teams, axes, figsize=(7.2, 7.2)):
    """Radar of normalised team-identity metrics (attacking + defensive) for ``teams``.

    ``table`` is ``results/team_metrics.csv``; ``axes`` is the list of metric columns. Each
    metric is min-max normalised across all teams so the shape is comparative.
    """
    import numpy as np

    norm = table.copy()
    for a in axes:
        col = norm[a].astype(float)
        lo, hi = col.min(), col.max()
        norm[a] = (col - lo) / (hi - lo) if hi > lo else 0.5
    angles = np.linspace(0, 2 * np.pi, len(axes), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"polar": True})
    fig.patch.set_facecolor(BG)
    ax.set_facecolor("#f6efe0")
    palette = ["#cf6a3c", "#5f8a4c", "#4f7fa3", "#9b5ca5"]  # distinct hues on warm light
    for i, team in enumerate(teams):
        rows = norm[norm["team"] == team]
        if rows.empty:
            continue
        vals = rows.iloc[0][axes].astype(float).tolist()
        vals += vals[:1]
        c = palette[i % len(palette)]
        ax.plot(angles, vals, color=c, lw=2.0, label=team)
        ax.fill(angles, vals, color=c, alpha=0.12)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([a.replace("_", " ") for a in axes], color=TEXT, fontsize=8)
    ax.set_yticklabels([])
    ax.set_ylim(0, 1)
    ax.tick_params(colors=MUTED)
    ax.spines["polar"].set_color("#d8cdb5")
    ax.grid(color="#d8cdb5", alpha=0.7)
    leg = ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), frameon=False, fontsize=8.5)
    for t in leg.get_texts():
        t.set_color(TEXT)
    return fig


def close(fig) -> None:
    """Close a figure (Streamlit keeps references otherwise)."""
    plt.close(fig)
