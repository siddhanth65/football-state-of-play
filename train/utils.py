"""Training utilities: seeding and the match-level data split (CLAUDE.md §6).

The split is **by match** (never by event) and **stratified by competition**, so the
test set is matches the model has never seen. Cached to ``data/processed/splits.json``.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

SEED = 42
SPLITS_PATH = Path("data/processed/splits.json")


def load_config(path: str | Path) -> dict:
    """Load a YAML run config from ``train/configs/`` ({} if the file is missing)."""
    p = Path(path)
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def set_seed(seed: int = SEED) -> None:
    """Fix all RNG seeds (Python, NumPy, Torch)."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - seed the global RNG for sklearn reproducibility
    torch.manual_seed(seed)


def build_splits(
    possessions: pd.DataFrame,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = SEED,
    out_path: Path | None = SPLITS_PATH,
) -> dict[str, list[int]]:
    """Build a match-level, competition-stratified train/val/test split.

    Args:
        possessions: The labelled dataset (needs ``match_id`` + ``competition_name``).
        ratios: ``(train, val, test)`` fractions.
        seed: RNG seed.
        out_path: Where to cache the split JSON; ``None`` to skip writing.

    Returns:
        ``{"train": [match_id], "val": [...], "test": [...]}``.
    """
    rng = np.random.default_rng(seed)
    matches = possessions.drop_duplicates("match_id")[["match_id", "competition_name"]]
    splits: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for _comp, grp in matches.groupby("competition_name"):
        ids = grp["match_id"].tolist()
        rng.shuffle(ids)
        n = len(ids)
        n_tr = int(n * ratios[0])
        n_va = int(n * ratios[1])
        splits["train"] += ids[:n_tr]
        splits["val"] += ids[n_tr : n_tr + n_va]
        splits["test"] += ids[n_tr + n_va :]
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(splits, indent=2, default=int), encoding="utf-8")
    return splits


def load_splits(path: Path = SPLITS_PATH) -> dict[str, list[int]]:
    """Load the cached split JSON."""
    return json.loads(path.read_text(encoding="utf-8"))


def split_graphs(graphs: list, match_ids: list[int]) -> list:
    """Filter a graph list to those whose ``match_id`` is in ``match_ids``."""
    keep = set(match_ids)
    return [g for g in graphs if int(g.match_id) in keep]
