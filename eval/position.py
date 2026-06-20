"""Position-aware experiment: does adding player positions help? (REPORT §4.6).

StatsBomb 360 carries no per-dot identity, so "individual positions" is realised two ways
(:func:`data.graphs.add_position_features`): a **heuristic per-node role** (team-relative third,
for every dot) and the **ball-carrier's true role** (the one position StatsBomb gives us, as a
graph-level one-hot). This trains the GAT with and without those features on the same split and
compares — a self-contained ablation that never touches the committed headline (we adopt only
if it clearly helps). Writes ``results/position.csv``. Run ``python -m eval.position``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from data.graphs import (
    N_ACTOR_ROLE_FEATURES,
    N_GLOBAL_FEATURES,
    N_NODE_FEATURES,
    N_NODE_ROLE_FEATURES,
    add_position_features,
    load_graph_cache,
)
from models.gnn import GAT
from train.train_gnn import train
from train.utils import build_splits

logger = logging.getLogger(__name__)
POSITION_PATH = Path("results/position.csv")
_KEYS = (
    "success_auc",
    "success_brier",
    "run_rmse_m",
    "receiver_top1",
    "receiver_top3",
    "presser_top1",
    "presser_top3",
)


def _augment(graphs: list, poss: pd.DataFrame) -> list:
    role = poss.set_index(["match_id", "possession"])["actor_role"].to_dict()
    return [
        add_position_features(g, role.get((int(g.match_id), int(g.possession)))) for g in graphs
    ]


def run_position(graphs: list, splits: dict[str, list[int]], poss: pd.DataFrame, epochs: int = 40):
    """Train base vs position-augmented GAT on the same split; return the comparison table."""
    rows: list[dict] = []

    _, mb = train(graphs, splits, epochs=epochs)
    rows.append({"variant": "base", **{k: mb.get(k) for k in _KEYS}})
    logger.info("position base: success_auc=%.4f", mb.get("success_auc", float("nan")))

    aug = _augment(graphs, poss)
    model = GAT(
        node_dim=N_NODE_FEATURES + N_NODE_ROLE_FEATURES,
        global_dim=N_GLOBAL_FEATURES + N_ACTOR_ROLE_FEATURES,
    )
    _, mp = train(aug, splits, epochs=epochs, model=model)
    rows.append({"variant": "position", **{k: mp.get(k) for k in _KEYS}})
    logger.info("position +pos: success_auc=%.4f", mp.get("success_auc", float("nan")))
    return pd.DataFrame(rows)


def main() -> None:
    """Run the position ablation and write ``results/position.csv``."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    poss = pd.read_parquet("data/processed/possessions.parquet")
    splits = build_splits(poss)
    graphs = load_graph_cache()
    table = run_position(graphs, splits, poss)
    POSITION_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(POSITION_PATH, index=False)
    print("\n" + table.round(4).to_string(index=False))
    print(f"\nWrote {POSITION_PATH}")


if __name__ == "__main__":
    main()
