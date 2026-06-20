"""Hyperparameter random search for the GAT (REPORT §6 future work).

Samples lr / hidden / layers / dropout, trains a short GAT per config, and ranks by
**validation** success AUC (never the test set — model selection on test would leak).
Writes ``results/sweep.csv`` incrementally. Run ``python -m train.sweep --trials 12
--epochs 25``; adopt the top config into ``train/configs/gnn_default.yaml``.
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from data.graphs import load_graph_cache
from models.gnn import GAT
from train.train_gnn import evaluate, train
from train.utils import build_splits, set_seed, split_graphs

logger = logging.getLogger(__name__)
SWEEP_PATH = Path("results/sweep.csv")
GRID = {
    "lr": [3e-4, 1e-4, 5e-5],
    "hidden": [64, 96, 128],
    "layers": [2, 3, 4],
    "dropout": [0.0, 0.1, 0.2],
}


def main() -> None:
    """Run the random search and write ``results/sweep.csv`` (ranked by val AUC)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=25)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    poss = pd.read_parquet("data/processed/possessions.parquet")
    splits = build_splits(poss)
    graphs = load_graph_cache()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_graphs = split_graphs(graphs, splits["val"])

    rng = random.Random(42)
    seen: set[tuple] = set()
    rows: list[dict] = []
    for t in range(args.trials):
        cfg = {k: rng.choice(v) for k, v in GRID.items()}
        key = tuple(cfg.values())
        if key in seen:
            continue
        seen.add(key)
        set_seed(42)
        model = GAT(hidden=cfg["hidden"], layers=cfg["layers"], dropout=cfg["dropout"])
        # two_stage=False: train the receiver head with CE only (see receiver diagnosis).
        model, _ = train(
            graphs,
            splits,
            epochs=args.epochs,
            patience=5,
            lr=cfg["lr"],
            model=model,
            two_stage=False,
        )
        vm = evaluate(model, DataLoader(val_graphs, batch_size=64), device)
        row = {
            **cfg,
            "val_success_auc": vm.get("success_auc"),
            "val_success_brier": vm.get("success_brier"),
            "val_receiver_top1": vm.get("receiver_top1"),
        }
        rows.append(row)
        logger.info("trial %d %s -> val AUC %.4f", t, cfg, vm.get("success_auc", 0.0))
        pd.DataFrame(rows).sort_values("val_success_auc", ascending=False).to_csv(
            SWEEP_PATH, index=False
        )

    table = pd.DataFrame(rows).sort_values("val_success_auc", ascending=False)
    print("\n" + table.round(4).to_string(index=False))
    print(f"\nBest: {table.iloc[0].to_dict()}\nWrote {SWEEP_PATH}")


if __name__ == "__main__":
    main()
