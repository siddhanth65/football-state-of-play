"""Bauer–Anzer-style engineered features for the B1 baseline (CLAUDE.md §5.6).

Computed from a graph's node tensor (positions + role flags from
:mod:`data.graphs`) — team shape, compactness, line height, and pressure proxies that
a logistic/linear model can use without message passing. Augmented with the
team-shape descriptors from Brandes et al. 2025 (RESEARCH_INTEGRATION §2.2).
"""

from __future__ import annotations

import numpy as np

FEATURE_NAMES = (
    "n_teammates",
    "n_defenders",
    "att_compactness",
    "def_compactness",
    "def_line_height",
    "att_width",
    "furthest_attacker_x",
    "defenders_ahead_of_ball",
    "mean_def_dist_to_ball",
)
N_ENGINEERED = len(FEATURE_NAMES)


def _spread(a: np.ndarray) -> float:
    return float(a.std()) if a.size > 1 else 0.0


def engineered_features(data) -> np.ndarray:
    """Return the ``(N_ENGINEERED,)`` feature vector for one graph."""
    x = data.x.detach().cpu().numpy()
    xc, yc = x[:, 0], x[:, 1]
    teammate = x[:, 4] > 0.5
    actor = x[:, 5] > 0.5
    defender = ~teammate
    ball_x = float(xc[actor][0]) if actor.any() else float(xc.max())
    return np.array(
        [
            float(teammate.sum()),
            float(defender.sum()),
            float(np.hypot(_spread(xc[teammate]), _spread(yc[teammate]))),
            float(np.hypot(_spread(xc[defender]), _spread(yc[defender]))),
            float(xc[defender].mean()) if defender.any() else 0.0,
            float(yc[teammate].max() - yc[teammate].min()) if teammate.sum() > 1 else 0.0,
            float(xc[teammate].max()) if teammate.any() else 0.0,
            float((xc[defender] > ball_x).sum()),
            float(x[:, 7][defender].mean()) if defender.any() else 0.0,
        ],
        dtype=float,
    )
