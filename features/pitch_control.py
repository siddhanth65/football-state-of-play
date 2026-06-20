"""Pitch control for the B2 baseline (CLAUDE.md §5.6).

# Adapted from Spearman 2018 ("Beyond Expected Goals") via Laurie Shaw's Friends of
# Tracking implementation, https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking
# (MIT licence). Simplified to the closed-form "potential field" variant (Fernández &
# Bornn 2018): each player's time-to-arrive at a target point is reaction time plus
# distance at top speed (after drifting with current velocity through the reaction
# window), and control is a logistic race between the two teams' fastest arrivers.
# The full Spearman model integrates arrival-time PDFs against ball flight time; the
# closed form is the standard cheap approximation and is sufficient for a scalar
# baseline feature.

Works on the cached graph tensors from :mod:`data.graphs` (positions and velocities
are stored normalised; they are converted back to metres here).
"""

from __future__ import annotations

import numpy as np

from data.possessions import BOX_X_MIN, BOX_Y_MAX, BOX_Y_MIN, PITCH_LENGTH, PITCH_WIDTH

# Spearman 2018 / Shaw (FoT) motion-model parameters.
REACTION_TIME_S = 0.7
MAX_SPEED_MS = 5.0
CONTROL_SIGMA_S = 0.45  # logistic scale on the arrival-time difference

# Penalty-area integration grid for the B2 scalar (2 m resolution).
BOX_GRID_STEP = 2.0


def _denorm(data) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(positions_m, velocities_ms, teammate_mask)`` from a graph."""
    x = data.x.detach().cpu().numpy()
    scale = np.array([PITCH_LENGTH, PITCH_WIDTH])
    return x[:, 0:2] * scale, x[:, 2:4] * scale, x[:, 4] > 0.5


def time_to_arrive(pos: np.ndarray, vel: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Per-player time to reach each target point (Spearman motion model).

    Args:
        pos: ``(N, 2)`` player positions in metres.
        vel: ``(N, 2)`` player velocities in m/s.
        targets: ``(G, 2)`` target points in metres.

    Returns:
        ``(N, G)`` arrival times in seconds: reaction time + remaining distance at
        top speed, after drifting with the current velocity during the reaction.
    """
    drifted = pos + vel * REACTION_TIME_S
    dists = np.linalg.norm(targets[None, :, :] - drifted[:, None, :], axis=2)
    return REACTION_TIME_S + dists / MAX_SPEED_MS


def control_surface_points(
    pos: np.ndarray, vel: np.ndarray, teammate: np.ndarray, targets: np.ndarray
) -> np.ndarray:
    """Attacking-team control probability at each target point (pure numpy).

    Args:
        pos: ``(N, 2)`` player positions in metres.
        vel: ``(N, 2)`` player velocities in m/s.
        teammate: length-``N`` attacking-team mask.
        targets: ``(G, 2)`` pitch points in metres.

    Returns:
        ``(G,)`` probabilities that the attacking team controls each point —
        a logistic race between the fastest attacker and fastest defender.
    """
    tta = time_to_arrive(pos, vel, targets)
    t_att = tta[teammate].min(axis=0) if teammate.any() else np.full(len(targets), np.inf)
    t_def = tta[~teammate].min(axis=0) if (~teammate).any() else np.full(len(targets), np.inf)
    return 1.0 / (1.0 + np.exp(-(t_def - t_att) / CONTROL_SIGMA_S))


def control_surface(data, targets: np.ndarray) -> np.ndarray:
    """Attacking-team control probability at each target point, from a graph."""
    pos, vel, teammate = _denorm(data)
    return control_surface_points(pos, vel, teammate, targets)


def box_grid(step: float = BOX_GRID_STEP) -> np.ndarray:
    """Penalty-area grid points ``(G, 2)`` in metres."""
    xs = np.arange(BOX_X_MIN + step / 2, PITCH_LENGTH, step)
    ys = np.arange(BOX_Y_MIN + step / 2, BOX_Y_MAX, step)
    gx, gy = np.meshgrid(xs, ys)
    return np.column_stack([gx.ravel(), gy.ravel()])


def box_control(data, targets: np.ndarray | None = None) -> float:
    """B2 scalar: mean attacking-team control over the penalty area."""
    if targets is None:
        targets = box_grid()
    return float(control_surface(data, targets).mean())
