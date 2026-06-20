"""Ablation suite (CLAUDE.md §7.3) → ``results/ablations.csv``.

Trains every variant with the identical protocol (same match-level splits, seed 42,
shorter schedule: 30 epochs / patience 5) so rows are directly comparable. The suite
is also designed to *explain the GAT-vs-transformer gap* seen in the final table:
``gat_hidden64`` tests width sensitivity (the base is now the sweep-selected 128) and
``gat_no_receiver`` the multi-task tax; ``gat_reflect`` ablates lateral-flip augmentation.

The CSV is written incrementally after each variant, so a crash never loses
finished runs. Run ``python -m eval.ablations`` (or ``make ablations``); use
``--only <substring>`` to run a subset and ``--epochs N`` for a quick pass.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import torch

from data.graphs import load_graph_cache
from models.gnn import GAT
from models.transformer import PlayerTransformer
from train.train_gnn import train, use_alt_run_label
from train.utils import build_splits

logger = logging.getLogger(__name__)
ABLATIONS_PATH = Path("results/ablations.csv")
EPOCHS = 30
PATIENCE = 5


def zero_velocity(graphs: list) -> list:
    """Clone graphs with the ``vx, vy`` node columns zeroed (never mutates inputs)."""
    out = []
    for g in graphs:
        g2 = g.clone()
        g2.x = g2.x.clone()
        g2.x[:, 2:4] = 0.0
        out.append(g2)
    return out


def _variants() -> dict[str, dict]:
    """Variant name → kwargs for :func:`run_variant`.

    ``model_fn`` is a factory (fresh weights per run); ``graphs_fn`` optionally
    transforms the shared graph list; remaining keys forward to ``train()``.
    """
    return {
        "gat_base": {"model_fn": GAT},
        "gat_hidden64": {"model_fn": lambda: GAT(hidden=64)},  # width-down vs the 128 base
        "gat_reflect": {"model_fn": GAT, "augment_reflect": True},  # lateral pitch flip aug
        "gat_no_receiver": {"model_fn": lambda: GAT(with_receiver=False), "w_receiver": 0.0},
        "gat_success_only": {
            "model_fn": lambda: GAT(with_receiver=False),
            "w_receiver": 0.0,
            "weights": (1.0, 0.0, 0.0),
        },
        "gat_no_edge_features": {"model_fn": lambda: GAT(use_edge_attr=False)},
        "gat_no_globals": {"model_fn": lambda: GAT(use_globals=False)},
        "gat_no_velocity": {"model_fn": GAT, "graphs_fn": zero_velocity},
        "transformer_base": {"model_fn": PlayerTransformer},
        "transformer_no_velocity": {"model_fn": PlayerTransformer, "graphs_fn": zero_velocity},
        # §3 #3 run-label ablation: run metrics are against the next-pass-receiver
        # target on the ~93% of graphs that have one — not comparable to the rows
        # above. Cloned first: use_alt_run_label overwrites y_run in place.
        "gat_run_alt": {
            "model_fn": GAT,
            "graphs_fn": lambda gs: use_alt_run_label([g.clone() for g in gs]),
        },
    }


def run_variant(
    name: str,
    graphs: list,
    splits: dict[str, list[int]],
    epochs: int,
    model_fn: Callable[[], torch.nn.Module],
    graphs_fn: Callable[[list], list] | None = None,
    **train_kwargs,
) -> dict[str, float]:
    """Train one variant and return its test metrics."""
    if graphs_fn is not None:
        graphs = graphs_fn(graphs)
    logger.info("=== ablation %s (%d graphs) ===", name, len(graphs))
    _, metrics = train(
        graphs, splits, epochs=epochs, patience=PATIENCE, model=model_fn(), **train_kwargs
    )
    return metrics


def main() -> None:
    """Run the ablation suite and write ``results/ablations.csv`` incrementally."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=None, help="substring filter on variant names")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    poss = pd.read_parquet("data/processed/possessions.parquet")
    graphs = load_graph_cache()
    splits = build_splits(poss)

    rows: list[dict] = []
    for name, spec in _variants().items():
        if args.only and args.only not in name:
            continue
        metrics = run_variant(name, graphs, splits, args.epochs, **spec)
        rows.append({"variant": name, **metrics})
        ABLATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(ABLATIONS_PATH, index=False)
        logger.info("ablations: wrote %d rows -> %s", len(rows), ABLATIONS_PATH)

    print("\n" + pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
