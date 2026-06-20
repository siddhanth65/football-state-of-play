"""Model-prediction overlays for the pitch viewer.

The hard part of this project to *show* is the run head: a predicted future
position vs the real one. The overlay makes the two unmistakable by drawing both
as vectors from the same origin (the runner): a solid coloured arrow for the
model's prediction, a dashed white arrow for what actually happened, with the gap
between them called out in metres. Receiver rings and the Spearman pitch-control
surface complete the picture. Everything is computed from the cached row — no torch.
"""

from __future__ import annotations

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch

# Locked palette — tuned for contrast on the rich-green pitch (warm "golden-hour" theme).
ATTACKER = "#e07a5f"  # warm coral — pops on green
DEFENDER = "#cfd6dd"  # light slate — clear, cool contrast
CARRIER = "#fff6e8"  # cream-white (the player on the ball)
ACTUAL = "#fff6e8"
RING = "#f2b134"  # amber receiver rings
MODEL_COLORS = {"gat": "#4f7fa3", "tfm": "#cf6a3c"}  # steel-blue GAT, terracotta transformer
MODEL_LABELS = {"gat": "GAT", "tfm": "Transformer"}

CONTROL_CMAP = LinearSegmentedColormap.from_list(
    "control", [(0.0, "#0b0f0d"), (0.5, "#0f2a20"), (1.0, "#0fae7c")]
)


def key_attacker(row) -> int | None:
    """Index of the run-head subject: furthest-forward teammate excl. the carrier."""
    px = np.asarray(row["px"], dtype=float)
    teammate = np.asarray(row["teammate"], dtype=bool)
    actor = np.asarray(row["actor"], dtype=bool)
    cand = teammate & ~actor
    if not cand.any():
        return None
    return int(np.where(cand)[0][int(np.argmax(px[cand]))])


def _arrow(ax, p0, p1, color: str, *, dashed: bool = False, curve: float = 0.12) -> None:
    """A gently curved arrow; dashed = 'what actually happened'."""
    ax.add_patch(
        FancyArrowPatch(
            (p0[0], p0[1]),
            (p1[0], p1[1]),
            connectionstyle=f"arc3,rad={curve}",
            arrowstyle="-|>,head_width=3.2,head_length=5.2",
            color=color,
            lw=2.2,
            ls=(0, (4, 2)) if dashed else "solid",
            zorder=7,
            shrinkA=9,
            shrinkB=3,
        )
    )


def _receiver_probs(row):
    probs = row.get("gat_receiver_probs") if hasattr(row, "get") else row["gat_receiver_probs"]
    if probs is None or len(probs) == 0:
        return None
    return np.asarray(probs, dtype=float)


def overlay_predictions(ax, pitch, row, models=("tfm",), show_actual: bool = True) -> None:
    """Run vectors (predicted solid, actual dashed) + receiver rings on the BEFORE frame."""
    px = np.asarray(row["px"], dtype=float)
    py = np.asarray(row["py"], dtype=float)
    key = key_attacker(row)

    # Receiver rings first (under the arrows).
    probs = _receiver_probs(row)
    if probs is not None:
        n = min(len(probs), len(px))
        for rank, i in enumerate(np.argsort(probs[:n])[::-1][:3]):
            if probs[i] <= 0:
                continue
            ax.scatter(
                px[i],
                py[i],
                s=540 + 2300 * float(probs[i]),
                facecolors="none",
                edgecolors=RING,
                linewidths=1.2,
                alpha=0.85 - 0.22 * rank,
                zorder=3,
            )
        top = int(np.argsort(probs[:n])[::-1][0])
        if probs[top] > 0:
            ax.annotate(
                f"{probs[top]:.0%} next pass",
                (px[top], py[top]),
                xytext=(0, -17),
                textcoords="offset points",
                ha="center",
                color=RING,
                fontsize=7.5,
                zorder=8,
            )

    if key is None:
        return
    start = (px[key], py[key])
    # Subject halo + label.
    ax.scatter(
        *start, s=720, facecolors="none", edgecolors=CARRIER, linewidths=1.1, alpha=0.7, zorder=5
    )
    ax.annotate(
        "the runner",
        start,
        xytext=(0, 13),
        textcoords="offset points",
        ha="center",
        color=CARRIER,
        fontsize=7.5,
        zorder=8,
    )

    for j, m in enumerate(models):
        pred = (float(row[f"{m}_run_x"]), float(row[f"{m}_run_y"]))
        _arrow(ax, start, pred, MODEL_COLORS[m], curve=0.10 + 0.08 * j)
        ax.scatter(
            *pred,
            s=150,
            marker="D",
            facecolors=MODEL_COLORS[m],
            edgecolors="#0b0f0d",
            linewidths=0.7,
            zorder=7,
        )

    if show_actual:
        actual = (float(row["run_true_x"]), float(row["run_true_y"]))
        _arrow(ax, start, actual, ACTUAL, dashed=True, curve=-0.10)
        ax.scatter(
            *actual, s=230, marker="X", c=ACTUAL, edgecolors="#0b0f0d", linewidths=0.8, zorder=7
        )
        # The runner's real displacement = the "stay put" (no-move) baseline error.
        moved = float(np.hypot(start[0] - actual[0], start[1] - actual[1]))
        ax.annotate(
            f"ran {moved:.0f} m",
            actual,
            xytext=(0, -15),
            textcoords="offset points",
            color="#a1a1aa",
            fontsize=6.5,
            ha="center",
            zorder=8,
        )
        # Call out the model's miss (predicted diamond -> actual X) for the lead model.
        # If "off by" exceeds "ran", the model did worse than assuming no movement.
        m0 = models[0]
        pred0 = np.array([float(row[f"{m0}_run_x"]), float(row[f"{m0}_run_y"])])
        miss = float(np.hypot(pred0[0] - actual[0], pred0[1] - actual[1]))
        mid = (pred0 + np.array(actual)) / 2
        ax.annotate(
            f"{MODEL_LABELS[m0]} off by {miss:.0f} m",
            (mid[0], mid[1]),
            color="#a1a1aa",
            fontsize=7,
            ha="center",
            zorder=8,
        )


def overlay_actual_movement(ax, pitch, row) -> None:
    """On the AFTER frame, show the runner's real movement + where the model guessed.

    Ghost = the runner's start (1.5 s earlier); dashed line to the X = the move they
    actually made; hollow diamond = the model's prediction, for comparison.
    """
    actual = (float(row["run_true_x"]), float(row["run_true_y"]))
    key = key_attacker(row)
    if key is not None:
        start = (float(np.asarray(row["px"])[key]), float(np.asarray(row["py"])[key]))
        ax.scatter(
            *start,
            s=300,
            facecolors="none",
            edgecolors="#52525b",
            linewidths=1.0,
            alpha=0.7,
            zorder=4,
        )
        ax.annotate(
            "started here",
            start,
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            color="#52525b",
            fontsize=7,
            zorder=8,
        )
        _arrow(ax, start, actual, ACTUAL, dashed=True, curve=0.12)
    ax.scatter(*actual, s=260, marker="X", c=ACTUAL, edgecolors="#0b0f0d", linewidths=0.8, zorder=7)
    ax.annotate(
        "runner, +1.5 s",
        actual,
        xytext=(0, -16),
        textcoords="offset points",
        ha="center",
        color=CARRIER,
        fontsize=7.5,
        zorder=8,
    )
    pred = (float(row["tfm_run_x"]), float(row["tfm_run_y"]))
    ax.scatter(
        *pred,
        s=150,
        marker="D",
        facecolors="none",
        edgecolors=MODEL_COLORS["tfm"],
        linewidths=1.5,
        zorder=7,
    )
    ax.annotate(
        "model guessed",
        pred,
        xytext=(0, 13),
        textcoords="offset points",
        ha="center",
        color=MODEL_COLORS["tfm"],
        fontsize=7,
        zorder=8,
    )


def overlay_control_surface(ax, row, step: float = 2.0) -> None:
    """Shade the attacking-team pitch-control probability over the whole pitch."""
    from data.possessions import PITCH_LENGTH, PITCH_WIDTH
    from features.pitch_control import control_surface_points

    pos = np.column_stack([row["px"], row["py"]]).astype(float)
    vel = np.column_stack([row["vx"], row["vy"]]).astype(float)
    # The nearest-neighbour velocity estimate (no player IDs in 360) can spike to
    # absurd values on a bad cross-frame match (the cache holds speeds > 700 m/s);
    # clamp to a plausible sprint speed so a single outlier can't distort the surface.
    speed = np.hypot(vel[:, 0], vel[:, 1])
    max_speed = 12.0
    vel = vel * np.where(speed > max_speed, max_speed / np.maximum(speed, 1e-9), 1.0)[:, None]
    teammate = np.asarray(row["teammate"], dtype=bool)
    xs = np.arange(step / 2, PITCH_LENGTH, step)
    ys = np.arange(step / 2, PITCH_WIDTH, step)
    gx, gy = np.meshgrid(xs, ys)
    control = control_surface_points(
        pos, vel, teammate, np.column_stack([gx.ravel(), gy.ravel()])
    ).reshape(gx.shape)
    ax.pcolormesh(
        xs, ys, control, cmap=CONTROL_CMAP, vmin=0, vmax=1, alpha=0.55, zorder=1, shading="gouraud"
    )
