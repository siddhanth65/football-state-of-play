"""Baselines B0–B3 (CLAUDE.md §5.6).

- **B0** — class prior: predict mean success, mean xT, and "no movement" for the run.
- **B1** — logistic (success) + linear (xT) regression on the engineered features.
- **B2** — Spearman 2018 pitch control integrated over the penalty area → logistic.
- **B3** — OBSO (Transition × Control × Score) over the attacking half → logistic.

B2/B3 reuse :class:`EngineeredBaseline` on their scalar geometric feature; the run
head has no geometric analogue, so all baselines share the no-move run prediction.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

from features.engineered import engineered_features
from features.obso import obso, obso_grid, scoring_surface
from features.pitch_control import box_control, box_grid


class PriorBaseline:
    """B0 — predict the training-set means / no movement."""

    def fit(self, y_success: np.ndarray, y_xt: np.ndarray) -> PriorBaseline:
        """Store the training-set success rate and mean xT."""
        self.p_success = float(np.mean(y_success))
        self.mean_xt = float(np.mean(y_xt))
        return self

    def predict_success(self, n: int) -> np.ndarray:
        """Constant success-rate prediction for ``n`` rows."""
        return np.full(n, self.p_success)

    def predict_xt(self, n: int) -> np.ndarray:
        """Constant mean-xT prediction for ``n`` rows."""
        return np.full(n, self.mean_xt)


class EngineeredBaseline:
    """B1 — logistic (success) + linear (xT) on engineered features."""

    def __init__(self) -> None:
        self.clf = LogisticRegression(max_iter=1000)
        self.reg = LinearRegression()

    def fit(self, x: np.ndarray, y_success: np.ndarray, y_xt: np.ndarray) -> EngineeredBaseline:
        """Fit the logistic (success) and linear (xT) models."""
        self.clf.fit(x, y_success)
        self.reg.fit(x, y_xt)
        return self

    def predict_success(self, x: np.ndarray) -> np.ndarray:
        """Predicted success probabilities."""
        return self.clf.predict_proba(x)[:, 1]

    def predict_xt(self, x: np.ndarray) -> np.ndarray:
        """Predicted xT progression."""
        return self.reg.predict(x)


def features_matrix(graphs: list) -> np.ndarray:
    """Stack engineered features for a list of graphs into ``(N, N_ENGINEERED)``."""
    return np.vstack([engineered_features(g) for g in graphs])


def pitch_control_matrix(graphs: list) -> np.ndarray:
    """B2 feature: ``(N, 1)`` mean attacker pitch control over the penalty area."""
    targets = box_grid()
    return np.array([[box_control(g, targets)] for g in graphs], dtype=float)


def obso_matrix(graphs: list) -> np.ndarray:
    """B3 feature: ``(N, 1)`` OBSO integral over the attacking half."""
    targets = obso_grid()
    scoring = scoring_surface(targets)
    return np.array([[obso(g, targets, scoring)] for g in graphs], dtype=float)


def run_baseline_nomove(graphs: list) -> np.ndarray:
    """B0 run prediction: "stay put" = a **zero displacement** for every possession.

    ``y_run`` is now the furthest-forward attacker's displacement over ~1.5 s (anchored
    in ``data.graphs.build_data``), so the no-move prediction is simply the zero vector:
    its RMSE equals the runner's actual movement — the honest reference the learned run
    heads must beat. (Equivalent to the old "predict current position" baseline under the
    absolute parametrisation, since the anchor cancels in the displacement.)
    """
    return np.zeros((len(graphs), 2), dtype=float)
