"""Slice analyses on the held-out test set (CLAUDE.md §7.2) → ``results/slices.csv``.

Slices: **all** possessions, **counter-attacks** (``play_pattern == "From Counter"``),
and **high-stakes** possessions (at least 4 attackers inside the penalty area at the
trigger). Metrics per slice for both trained models.

Run ``python -m eval.slices`` (needs ``graphs.pt`` + both checkpoints).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from data.graphs import PLAY_PATTERNS, load_graph_cache
from data.possessions import BOX_X_MIN, BOX_Y_MAX, BOX_Y_MIN, PITCH_LENGTH, PITCH_WIDTH
from eval.figures import collect, load_models
from eval.metrics import run_metrics, success_metrics, xt_metrics
from train.utils import build_splits, split_graphs

logger = logging.getLogger(__name__)
SLICES_PATH = Path("results/slices.csv")
COUNTER_IDX = PLAY_PATTERNS.index("From Counter")
HIGH_STAKES_ATTACKERS = 4


def attackers_in_box(g) -> int:
    """Number of attacking-team players (excl. nobody) inside the penalty area."""
    x = g.x.numpy()
    px, py = x[:, 0] * PITCH_LENGTH, x[:, 1] * PITCH_WIDTH
    teammate = x[:, 4] > 0.5
    in_box = (px >= BOX_X_MIN) & (py >= BOX_Y_MIN) & (py <= BOX_Y_MAX)
    return int((teammate & in_box).sum())


def slice_masks(graphs: list, poss: pd.DataFrame) -> dict[str, np.ndarray]:
    """Boolean masks for the §7.2 slices.

    Includes ``open_play`` vs ``set_piece``: the dataset pools set-piece-proximate
    triggers (``from_set_piece``, ~32% of rows) rather than dropping them, so the
    slice makes that split visible instead of mislabelling the whole set "open play".
    """
    counter = np.array([bool(g.u[0, COUNTER_IDX] > 0.5) for g in graphs])
    high = np.array([attackers_in_box(g) >= HIGH_STAKES_ATTACKERS for g in graphs])
    sp_lookup = {
        (int(r.match_id), int(r.possession)): bool(r.from_set_piece)
        for r in poss.itertuples(index=False)
    }
    set_piece = np.array(
        [sp_lookup.get((int(g.match_id), int(g.possession)), False) for g in graphs]
    )
    # Positional-zone slices: the ball-carrier's role (the one known position). Answers
    # "does the model read midfield build-up differently from final-third forwards?"
    role_lookup = {
        (int(r.match_id), int(r.possession)): r.actor_role
        for r in poss.itertuples(index=False)
    }
    roles = [role_lookup.get((int(g.match_id), int(g.possession))) for g in graphs]
    masks = {
        "all": np.ones(len(graphs), dtype=bool),
        "open_play": ~set_piece,
        "set_piece": set_piece,
        "counter": counter,
        "high_stakes": high,
    }
    for bucket in ("DEF", "MID", "FWD"):
        masks[f"carrier_{bucket}"] = np.array([r == bucket for r in roles])
    return masks


def main() -> None:
    """Evaluate both models on each slice and write the table."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    poss = pd.read_parquet("data/processed/possessions.parquet")
    test = split_graphs(load_graph_cache(), build_splits(poss)["test"])
    masks = slice_masks(test, poss)
    logger.info(
        "slices: %d test graphs (%d open-play, %d set-piece, %d counters, %d high-stakes)",
        len(test),
        int(masks["open_play"].sum()),
        int(masks["set_piece"].sum()),
        int(masks["counter"].sum()),
        int(masks["high_stakes"].sum()),
    )

    rows = []
    for model_name, model in load_models().items():
        r = collect(model, test)
        for slice_name, m in masks.items():
            if m.sum() < 20:
                continue
            row = {"model": model_name, "slice": slice_name, "n": int(m.sum())}
            row.update(
                {
                    f"success_{k}": v
                    for k, v in success_metrics(r["y_success"][m], r["p_success"][m]).items()
                }
            )
            row.update({f"xt_{k}": v for k, v in xt_metrics(r["y_xt"][m], r["xt"][m]).items()})
            row.update({f"run_{k}": v for k, v in run_metrics(r["y_run"][m], r["run"][m]).items()})
            rows.append(row)

    table = pd.DataFrame(rows)
    SLICES_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(SLICES_PATH, index=False)
    print("\n" + table.round(4).to_string(index=False))
    print(f"\nWrote {SLICES_PATH}")


if __name__ == "__main__":
    main()
