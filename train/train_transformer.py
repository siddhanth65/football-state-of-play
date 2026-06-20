"""Train the transformer secondary model (CLAUDE.md §5.3, §10 Week 5).

Identical protocol to ``train_gnn`` (same splits, optimiser, schedule, metrics) so the
GAT-vs-transformer comparison is apples-to-apples; only the encoder differs. Appends a
``Transformer`` row to ``results/final_table.csv``.

Run ``python -m train.train_transformer`` (full) or ``--smoke`` for a quick check.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import torch

from data.graphs import GRAPH_CACHE_PATH, build_graph_cache, load_graph_cache
from models.transformer import PlayerTransformer
from train.train_gnn import CKPT_DIR, train
from train.utils import build_splits, load_config

logger = logging.getLogger(__name__)


def main() -> None:
    """Entry point: load graphs + splits, train, save checkpoint + update the table."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="train/configs/transformer_default.yaml")
    cfg = load_config(pre.parse_known_args()[0].config)
    tr_cfg = cfg.get("train", {})

    parser = argparse.ArgumentParser(parents=[pre])
    parser.add_argument("--epochs", type=int, default=tr_cfg.get("max_epochs", 50))
    parser.add_argument("--batch-size", type=int, default=tr_cfg.get("batch_size", 32))
    parser.add_argument("--smoke", action="store_true", help="2 epochs on a small subset")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    poss = pd.read_parquet("data/processed/possessions.parquet")
    if GRAPH_CACHE_PATH.exists():
        graphs = load_graph_cache()
    else:
        logger.warning("graph cache missing — building a small subset")
        graphs = build_graph_cache(poss, out_path=None, limit=10)
    splits = build_splits(poss)

    epochs = 2 if args.smoke else args.epochs
    if args.smoke:
        keep = set(splits["train"][:6] + splits["val"][:3] + splits["test"][:3])
        graphs = [g for g in graphs if int(g.match_id) in keep]

    model, metrics = train(
        graphs,
        splits,
        epochs=epochs,
        batch_size=args.batch_size,
        lr=float(tr_cfg.get("lr", 1e-4)),
        weight_decay=float(tr_cfg.get("weight_decay", 1e-4)),
        patience=int(tr_cfg.get("early_stopping_patience", 7)),
        model=PlayerTransformer(),
    )
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = CKPT_DIR / ("transformer_smoke.pt" if args.smoke else "transformer.pt")
    torch.save(model.state_dict(), ckpt)
    print("\nTest metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"\nSaved checkpoint -> {ckpt}")

    if not args.smoke:
        table_path = Path("results/final_table.csv")
        row = pd.DataFrame([{"model": "Transformer", **metrics}])
        if table_path.exists():
            table = pd.read_csv(table_path)
            table = table[table["model"] != "Transformer"]
            row = pd.concat([table, row], ignore_index=True)
        row.to_csv(table_path, index=False)
        print(f"Wrote {table_path}")


if __name__ == "__main__":
    main()
