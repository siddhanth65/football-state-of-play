"""OBSO (Off-Ball Scoring Opportunity) for the B3 baseline (CLAUDE.md §5.6).

# Adapted from Spearman 2018 ("Beyond Expected Goals"), with the PAUSA formulation
# (Lee et al. 2026) as the reference: OBSO = Σ_p P(Transition→p) · P(Control at p) ·
# P(Score | p). Simplifications for a scalar baseline feature:
# - Transition: isotropic Gaussian around the ball (σ = 14 m), normalised over the
#   grid — Spearman's empirical next-ball-location kernel is approximately this.
# - Control: the closed-form pitch-control race from :mod:`features.pitch_control`.
# - Score: the pooled xT grid (data/processed/xt_grid.npy) as the relative scoring
#   potential, normalised to [0, 1]; falls back to a distance-to-goal logistic.

The bilinear grid lookup is duplicated from :func:`data.labels.xt_value` on purpose:
importing ``data.labels`` pulls in ``statsbombpy``, which must never be imported
after ``torch_geometric`` in the same process (Windows DLL access violation).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from data.possessions import PITCH_LENGTH, PITCH_WIDTH
from features.pitch_control import control_surface

XT_GRID_PATH = Path("data/processed/xt_grid.npy")
TRANSITION_SIGMA_M = 14.0

# Attacking-half integration grid (coarser than B2's box grid; OBSO sums pitch-wide).
OBSO_X_MIN = 60.0
OBSO_GRID_STEP = 3.0


def obso_grid(step: float = OBSO_GRID_STEP) -> np.ndarray:
    """Attacking-half grid points ``(G, 2)`` in metres."""
    xs = np.arange(OBSO_X_MIN + step / 2, PITCH_LENGTH, step)
    ys = np.arange(step / 2, PITCH_WIDTH, step)
    gx, gy = np.meshgrid(xs, ys)
    return np.column_stack([gx.ravel(), gy.ravel()])


def _bilinear(grid: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vectorised bilinear lookup on an ``(n_x, n_y)`` pitch grid (cell centres)."""
    n_x, n_y = grid.shape
    gx = np.clip(x / PITCH_LENGTH * n_x - 0.5, 0, n_x - 1)
    gy = np.clip(y / PITCH_WIDTH * n_y - 0.5, 0, n_y - 1)
    i0, j0 = np.floor(gx).astype(int), np.floor(gy).astype(int)
    i1, j1 = np.minimum(i0 + 1, n_x - 1), np.minimum(j0 + 1, n_y - 1)
    fx, fy = gx - i0, gy - j0
    top = grid[i0, j0] * (1 - fx) + grid[i1, j0] * fx
    bot = grid[i0, j1] * (1 - fx) + grid[i1, j1] * fx
    return top * (1 - fy) + bot * fy


def scoring_surface(targets: np.ndarray, xt_grid: np.ndarray | None = None) -> np.ndarray:
    """Relative scoring potential in [0, 1] at each target point.

    Uses the pooled xT grid when available (normalised by its max); otherwise a
    logistic decay in distance to the goal centre.
    """
    if xt_grid is None and XT_GRID_PATH.exists():
        xt_grid = np.load(XT_GRID_PATH)
    if xt_grid is not None and float(xt_grid.max()) > 0:
        return _bilinear(xt_grid, targets[:, 0], targets[:, 1]) / float(xt_grid.max())
    d_goal = np.linalg.norm(targets - np.array([PITCH_LENGTH, PITCH_WIDTH / 2]), axis=1)
    return 1.0 / (1.0 + np.exp((d_goal - 16.0) / 4.0))


def transition_surface(ball: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """P(next ball location = p): normalised Gaussian kernel around the ball."""
    d2 = ((targets - ball) ** 2).sum(axis=1)
    w = np.exp(-d2 / (2 * TRANSITION_SIGMA_M**2))
    total = float(w.sum())
    return w / total if total > 0 else w


def _ball_position(data) -> np.ndarray:
    """Ball position in metres: the actor node if visible, else the deepest player."""
    x = data.x.detach().cpu().numpy()
    scale = np.array([PITCH_LENGTH, PITCH_WIDTH])
    actor = x[:, 5] > 0.5
    if actor.any():
        return x[actor][0, 0:2] * scale
    return x[np.argmax(x[:, 0]), 0:2] * scale


def obso(data, targets: np.ndarray | None = None, scoring: np.ndarray | None = None) -> float:
    """B3 scalar: Σ_p Transition(p) · Control(p) · Score(p) over the attacking half.

    Args:
        data: A graph from :mod:`data.graphs`.
        targets: Integration grid (defaults to :func:`obso_grid`).
        scoring: Precomputed :func:`scoring_surface` for ``targets`` — pass it when
            scoring many graphs to avoid re-reading the xT grid per call.

    Returns:
        The OBSO integral.
    """
    if targets is None:
        targets = obso_grid()
    if scoring is None:
        scoring = scoring_surface(targets)
    t = transition_surface(_ball_position(data), targets)
    c = control_surface(data, targets)
    return float((t * c * scoring).sum())
