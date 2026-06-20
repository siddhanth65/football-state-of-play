"""Dynamic-xT redesign experiment: is a configuration-conditioned threat learnable? (REPORT §4.5).

The realized xT-progression target is at the noise ceiling (R² ≈ 0) — it is dominated by where
the possession *ends*, an outcome a single freeze-frame cannot see. This trains the **same** xT
head against three targets and compares, to show the head was never the problem — the target was:

- ``realized`` — ``xT(end) − xT(trigger)`` (the committed null).
- ``obso`` — **OBSO at the trigger frame** (:func:`features.obso.obso`): a freeze-frame-conditioned
  scoring value, hence a *learnable* threat. A fast neural surrogate of a physics threat surface =
  a learned **Dynamic xT**. (Transparent caveat: predicting a model-derived surface.)
- ``risk`` — realized × P(next pass completes): risk-adjusted value (Paul et al. 2025), where
  P(complete) is a logistic on the engineered features (:mod:`features.engineered`).

The xT head and training protocol are unchanged; only the regression target swaps (we clone the
graphs and reassign ``y_xt``). Trains xT-only (``weights=(0, 1, 0)``, no receiver/defense heads)
to isolate the question. Writes ``results/dxt.csv``. Run ``python -m eval.dxt``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

from data.graphs import load_graph_cache
from features.engineered import engineered_features
from features.obso import obso
from models.gnn import GAT
from train.train_gnn import train
from train.utils import build_splits

logger = logging.getLogger(__name__)
DXT_PATH = Path("results/dxt.csv")


def completion_probability(
    graphs: list, splits: dict[str, list[int]], poss: pd.DataFrame
) -> np.ndarray:
    """P(next pass completes) per graph from a logistic on the engineered features.

    Fit on the train split's possessions that have a completion label; predict for all.
    """
    comp = poss.set_index(["match_id", "possession"])["next_pass_completed"].to_dict()
    x = np.nan_to_num(np.array([engineered_features(g) for g in graphs], dtype=float))
    keys = [(int(g.match_id), int(g.possession)) for g in graphs]
    y = np.array([comp.get(k, np.nan) for k in keys], dtype=float)
    train_ids = set(splits["train"])
    fit = np.array([int(g.match_id) in train_ids for g in graphs]) & np.isfinite(y)
    clf = LogisticRegression(max_iter=1000).fit(x[fit], y[fit].astype(int))
    return clf.predict_proba(x)[:, 1]


def run_dxt(
    graphs: list, splits: dict[str, list[int]], poss: pd.DataFrame, epochs: int = 40
) -> pd.DataFrame:
    """Train the xT head against realized / obso / risk targets; return R²/RMSE per target."""
    p_complete = completion_probability(graphs, splits, poss)
    realized = np.array([float(g.y_xt) for g in graphs], dtype=float)
    targets = {
        "realized": realized,
        "obso": np.array([float(obso(g)) for g in graphs], dtype=float),
        "risk": realized * p_complete,
    }
    rows: list[dict] = []
    for name, yvals in targets.items():
        gs = [g.clone() for g in graphs]
        for g, v in zip(gs, yvals, strict=True):
            g.y_xt = torch.tensor([float(v)], dtype=torch.float)
        model = GAT(with_receiver=False, with_defense=False)  # isolate the xT head
        _, m = train(gs, splits, epochs=epochs, weights=(0.0, 1.0, 0.0), model=model)
        rows.append(
            {
                "xt_target": name,
                "xt_r2": m["xt_r2"],
                "xt_rmse": m["xt_rmse"],
                "target_std": float(np.std(yvals)),
            }
        )
        logger.info("DxT %-8s R2=%.4f RMSE=%.4f", name, m["xt_r2"], m["xt_rmse"])
    return pd.DataFrame(rows)


def main() -> None:
    """Run the DxT target ablation and write ``results/dxt.csv``."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    poss = pd.read_parquet("data/processed/possessions.parquet")
    splits = build_splits(poss)
    graphs = load_graph_cache()
    table = run_dxt(graphs, splits, poss)
    DXT_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(DXT_PATH, index=False)
    print("\n" + table.round(4).to_string(index=False))
    print(f"\nWrote {DXT_PATH}")


if __name__ == "__main__":
    main()
