"""Multi-seed training for error bars (addresses the single-seed limitation, REPORT §6).

Trains the GAT and transformer over several seeds on the **fixed** match-level split
(only model initialisation + batch order vary), and reports mean ± std per metric so
model-vs-model and model-vs-baseline claims carry uncertainty rather than resting on one
lucky run.

Run ``python -m train.multiseed`` (``--seeds 42 1 2 3 4`` to override; ``--no-two-stage``
to train the GAT receiver head without the two-stage success term).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from data.graphs import load_graph_cache
from models.transformer import PlayerTransformer
from train.train_gnn import train
from train.utils import build_splits, set_seed

logger = logging.getLogger(__name__)
SEEDS_PATH = Path("results/final_table_seeds.csv")
METRICS = (
    "success_auc",
    "success_brier",
    "xt_rmse",
    "run_rmse_m",
    "run_hit_rate_3m",
    "receiver_top1",
    "receiver_top3",
    "presser_top1",
    "presser_top3",
    "defsuccess_auc",
    "dxt_r2",
    "defline_rmse_m",
    "xpass_auc",
)


def run_multiseed(
    graphs: list, splits: dict[str, list[int]], seeds: list[int], two_stage: bool = True
) -> pd.DataFrame:
    """Train GAT + transformer over ``seeds`` and aggregate metrics to mean ± std.

    Writes ``results/final_table_seeds.csv`` after each model so a long run that is
    interrupted still keeps the finished models.
    """
    # Per-model learning rate: the GAT adopts the sweep-selected lr (results/sweep.csv);
    # the transformer keeps the lr it was tuned at (1e-4) — the sweep covered the GAT only.
    lr_for = {"GAT": 3e-4, "Transformer": 1e-4}
    rows: list[dict] = []
    for name in ("GAT", "Transformer"):
        per: dict[str, list[float]] = {}
        for s in seeds:
            set_seed(s)  # seed the model init too (esp. the transformer, built here)
            model = None if name == "GAT" else PlayerTransformer()
            _, m = train(graphs, splits, model=model, seed=s, two_stage=two_stage, lr=lr_for[name])
            for k in METRICS:
                if m.get(k) is not None:
                    per.setdefault(k, []).append(float(m[k]))
            logger.info("%s seed %d: success_auc=%.4f", name, s, m.get("success_auc", float("nan")))
        row: dict[str, float | str | int] = {"model": name, "n_seeds": len(seeds)}
        for k, vals in per.items():
            row[f"{k}_mean"] = float(np.mean(vals))
            row[f"{k}_std"] = float(np.std(vals))
        rows.append(row)
        SEEDS_PATH.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(SEEDS_PATH, index=False)  # incremental save
    return pd.DataFrame(rows)


def main() -> None:
    """Train the multi-seed suite and write ``results/final_table_seeds.csv``."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2])
    parser.add_argument("--no-two-stage", action="store_true", help="GAT receiver CE only")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    poss = pd.read_parquet("data/processed/possessions.parquet")
    splits = build_splits(poss)
    graphs = load_graph_cache()
    table = run_multiseed(graphs, splits, args.seeds, two_stage=not args.no_two_stage)
    SEEDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(SEEDS_PATH, index=False)
    print("\n" + table.round(4).to_string(index=False))
    print(f"\nWrote {SEEDS_PATH}")


if __name__ == "__main__":
    main()
