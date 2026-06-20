"""Does adding defensive-shape features improve the (weak) defensive heads? (REPORT §4.10).

The defensive-success head is weak (0.59 AUC) partly because its target is outcome-bound, but
also because the model is given little explicit defensive context. This tests whether four
shape descriptors of the **defending team** — compactness, defensive-line height, width, and
numerical balance ahead of the ball (Brandes et al. 2025 shape graphs; PPDA intuition) — help,
as a self-contained ablation that never touches the committed headline (adopt only if it helps).
Writes ``results/defense_features.csv``. Run ``python -m eval.defense_features``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from data.graphs import N_DEF_SHAPE_FEATURES, load_graph_cache
from train.train_gnn import train
from train.utils import build_splits

logger = logging.getLogger(__name__)
DEF_FEAT_PATH = Path("results/defense_features.csv")
N_DEF_SHAPE = N_DEF_SHAPE_FEATURES
_KEYS = ("success_auc", "defsuccess_auc", "presser_top1", "defline_rmse_m", "xpass_auc")


def _zero_shape(graphs: list) -> list:
    """Clone graphs with the (now-baked) defensive-shape globals zeroed = the 'base' arm."""
    out = []
    for g in graphs:
        h = g.clone()
        h.u = h.u.clone()
        h.u[:, -N_DEF_SHAPE_FEATURES:] = 0.0
        out.append(h)
    return out


def run_defense_features(graphs: list, splits: dict[str, list[int]], epochs: int = 40):
    """On/off ablation of the now-adopted defensive-shape globals (same split)."""
    rows: list[dict] = []
    _, mb = train(_zero_shape(graphs), splits, epochs=epochs)
    rows.append({"variant": "base", **{k: mb.get(k) for k in _KEYS}})
    logger.info("defense base: defsuccess_auc=%.4f", mb.get("defsuccess_auc", float("nan")))

    _, mp = train(graphs, splits, epochs=epochs)  # shape present (headline)
    rows.append({"variant": "+def_shape", **{k: mp.get(k) for k in _KEYS}})
    logger.info("defense +shape: defsuccess_auc=%.4f", mp.get("defsuccess_auc", float("nan")))
    return pd.DataFrame(rows)


def main() -> None:
    """Run the defensive-features ablation and write ``results/defense_features.csv``."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    poss = pd.read_parquet("data/processed/possessions.parquet")
    table = run_defense_features(load_graph_cache(), build_splits(poss))
    DEF_FEAT_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(DEF_FEAT_PATH, index=False)
    print("\n" + table.round(4).to_string(index=False))
    print(f"\nWrote {DEF_FEAT_PATH}")


if __name__ == "__main__":
    main()
