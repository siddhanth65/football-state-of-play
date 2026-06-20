"""Cross-gender transfer experiment: does the model generalise men <-> women? (REPORT §6).

The dataset pools men's and women's competitions and assumes they are exchangeable without
testing it. This trains on one gender and evaluates on the other (both directions), so the
generalisation gap is measured rather than assumed. Reuses the standard training loop on a
custom by-match split. Writes ``results/transfer.csv``. Run ``python -m eval.transfer``.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import pandas as pd

from data.graphs import load_graph_cache
from train.train_gnn import train
from train.utils import SEED

logger = logging.getLogger(__name__)
TRANSFER_PATH = Path("results/transfer.csv")
VAL_FRACTION = 0.15


def _gender(name: object) -> str:
    return "women" if "women" in str(name).lower() else "men"


def _split(train_ids: list[int], test_ids: list[int], seed: int = SEED) -> dict[str, list[int]]:
    """Hold out a val fraction of the training-gender matches; test = the other gender."""
    ids = list(train_ids)
    random.Random(seed).shuffle(ids)
    n_val = max(1, int(len(ids) * VAL_FRACTION))
    return {"train": ids[n_val:], "val": ids[:n_val], "test": list(test_ids)}


def run_transfer(graphs: list, poss: pd.DataFrame, epochs: int = 40) -> pd.DataFrame:
    """Train on each gender and evaluate on the other; return the comparison table."""
    matches = poss.drop_duplicates("match_id")[["match_id", "competition_name"]].copy()
    matches["g"] = matches["competition_name"].map(_gender)
    men = matches.loc[matches["g"] == "men", "match_id"].astype(int).tolist()
    women = matches.loc[matches["g"] == "women", "match_id"].astype(int).tolist()
    logger.info("transfer: %d men's matches, %d women's matches", len(men), len(women))

    rows: list[dict] = []
    for name, tr_ids, te_ids in (("men->women", men, women), ("women->men", women, men)):
        _, m = train(graphs, _split(tr_ids, te_ids), epochs=epochs)
        rows.append(
            {
                "direction": name,
                "n_train_matches": len(tr_ids),
                "n_test_matches": len(te_ids),
                "success_auc": m.get("success_auc"),
                "success_brier": m.get("success_brier"),
                "receiver_top1": m.get("receiver_top1"),
            }
        )
        logger.info("transfer %s: success_auc=%.4f", name, m.get("success_auc", float("nan")))
    return pd.DataFrame(rows)


def main() -> None:
    """Run the cross-gender transfer experiment and write ``results/transfer.csv``."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    poss = pd.read_parquet("data/processed/possessions.parquet")
    graphs = load_graph_cache()
    table = run_transfer(graphs, poss)
    TRANSFER_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(TRANSFER_PATH, index=False)
    print("\n" + table.round(4).to_string(index=False))
    print(f"\nWrote {TRANSFER_PATH}")


if __name__ == "__main__":
    main()
