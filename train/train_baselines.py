"""Train + evaluate baselines B0–B3 → ``results/baselines.csv`` (CLAUDE.md §5.6, §7).

Reads the cached graphs (``data/graphs.py``), splits by match, fits the baselines on
train, and reports success / xT / run metrics on the held-out test set.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from data.graphs import GRAPH_CACHE_PATH, build_graph_cache, load_graph_cache
from eval.metrics import run_metrics, success_metrics, xt_metrics
from models.baselines import (
    EngineeredBaseline,
    PriorBaseline,
    features_matrix,
    obso_matrix,
    pitch_control_matrix,
    run_baseline_nomove,
)
from train.utils import build_splits, set_seed, split_graphs

logger = logging.getLogger(__name__)
RESULTS_PATH = Path("results/baselines.csv")


def _targets(graphs: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_s = np.array([float(g.y_success) for g in graphs])
    y_x = np.array([float(g.y_xt) for g in graphs])
    y_r = np.array([g.y_run.numpy().reshape(2) for g in graphs])
    return y_s, y_x, y_r


def evaluate_baselines(train_graphs: list, test_graphs: list) -> pd.DataFrame:
    """Fit B0–B3 on train, return a metrics table on test."""
    ytr_s, ytr_x, _ = _targets(train_graphs)
    yte_s, yte_x, yte_r = _targets(test_graphs)
    n = len(test_graphs)
    # All baselines share the no-move run prediction (none model movement).
    run_m = {f"run_{k}": v for k, v in run_metrics(yte_r, run_baseline_nomove(test_graphs)).items()}

    def fitted_row(name: str, xtr: np.ndarray, xte: np.ndarray) -> dict:
        b = EngineeredBaseline().fit(xtr, ytr_s, ytr_x)
        return {
            "model": name,
            **{
                f"success_{k}": v for k, v in success_metrics(yte_s, b.predict_success(xte)).items()
            },
            **{f"xt_{k}": v for k, v in xt_metrics(yte_x, b.predict_xt(xte)).items()},
            **run_m,
        }

    b0 = PriorBaseline().fit(ytr_s, ytr_x)
    rows = [
        {
            "model": "B0_prior",
            **{f"success_{k}": v for k, v in success_metrics(yte_s, b0.predict_success(n)).items()},
            **{f"xt_{k}": v for k, v in xt_metrics(yte_x, b0.predict_xt(n)).items()},
            **run_m,
        },
        fitted_row("B1_engineered", features_matrix(train_graphs), features_matrix(test_graphs)),
    ]
    logger.info("baselines: computing B2 pitch control surfaces ...")
    rows.append(
        fitted_row(
            "B2_pitch_control",
            pitch_control_matrix(train_graphs),
            pitch_control_matrix(test_graphs),
        )
    )
    logger.info("baselines: computing B3 OBSO surfaces ...")
    rows.append(fitted_row("B3_obso", obso_matrix(train_graphs), obso_matrix(test_graphs)))
    return pd.DataFrame(rows)


def main() -> None:
    """Build/load graphs, split, train baselines, write the results table."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    set_seed()
    graphs = (
        load_graph_cache()
        if GRAPH_CACHE_PATH.exists()
        else build_graph_cache(
            pd.read_parquet("data/processed/possessions.parquet"), out_path=None, limit=20
        )
    )
    poss = pd.read_parquet("data/processed/possessions.parquet")
    splits = build_splits(poss)
    train_graphs = split_graphs(graphs, splits["train"])
    test_graphs = split_graphs(graphs, splits["test"])
    logger.info("baselines: %d train / %d test graphs", len(train_graphs), len(test_graphs))

    table = evaluate_baselines(train_graphs, test_graphs)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS_PATH, index=False)
    print("\n" + table.to_string(index=False))
    print(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
