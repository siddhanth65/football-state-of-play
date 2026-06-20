"""Evaluation metrics per head (CLAUDE.md §7.1).

Success → ROC-AUC + Brier; xT → RMSE + R²; run → RMSE in metres + hit-rate within 3 m.
Run inputs/predictions are normalised ``(x, y)`` in [0, 1]; converted to metres here.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss, mean_squared_error, r2_score, roc_auc_score

from data.graphs import PITCH_LENGTH, PITCH_WIDTH


def success_metrics(y_true: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    """ROC-AUC and Brier score (AUC is ``nan`` if only one class is present)."""
    y_true = np.asarray(y_true)
    auc = float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) > 1 else float("nan")
    return {"auc": auc, "brier": float(brier_score_loss(y_true, prob))}


def xt_metrics(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """RMSE and R² for the xT-progression head."""
    return {
        "rmse": float(mean_squared_error(y_true, pred) ** 0.5),
        "r2": float(r2_score(y_true, pred)),
    }


def run_metrics(y_true_norm: np.ndarray, pred_norm: np.ndarray) -> dict[str, float]:
    """RMSE in metres and hit-rate within 3 m for the run head."""
    scale = np.array([PITCH_LENGTH, PITCH_WIDTH])
    dist = np.linalg.norm((np.asarray(y_true_norm) - np.asarray(pred_norm)) * scale, axis=1)
    return {"rmse_m": float(np.sqrt((dist**2).mean())), "hit_rate_3m": float((dist <= 3).mean())}
