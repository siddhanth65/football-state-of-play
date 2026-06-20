"""Per-possession sequences of freeze-frame graphs for the temporal model (REPORT §6).

The single-frame model is the biggest limitation: one snapshot cannot see motion or the
build-up of an attack. This module assembles, for each possession in
``possessions.parquet``, the **last K on-ball freeze-frames up to and including the
trigger** — the build-up — as a short sequence of per-frame graphs. A temporal model
(:mod:`models.temporal`) then reads the sequence to judge whether the attack reaches
threat, using the *evolution* of the state of play (momentum, a developing 3v2), not a
frozen instant.

No player IDs are needed: each frame is an independent permutation-invariant graph, and
the temporal model reasons over the sequence of graph-level embeddings (see REPORT's
player-ID discussion). Derived from existing artifacts + the local clone, so it needs no
rebuild of the core dataset. Run ``python -m data.sequences``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import torch

from data.graphs import build_data
from data.load import load_events_local, load_frames_local
from data.possessions import add_event_time

logger = logging.getLogger(__name__)
SEQUENCE_CACHE_PATH = Path("data/processed/sequences.pt")
SEQUENCE_LEN = 6  # build-up frames up to and including the trigger (newest last)


def buildup_event_ids(events: pd.DataFrame, row, k: int = SEQUENCE_LEN) -> list[str]:
    """The last ``k`` on-ball possession-team event ids (with location) up to the trigger.

    Args:
        events: One match's events with a ``time_s`` column.
        row: A possession row (needs ``possession``/``possession_team``/``trigger_time_s``).
        k: Maximum sequence length.

    Returns:
        Ordered event ids (oldest → trigger), at most ``k``.
    """
    grp = events[
        (events["possession"] == row.possession)
        & (events["team"] == row.possession_team)
        & events["location"].notna()
        & (events["time_s"] <= row.trigger_time_s)
    ]
    return grp.sort_values("index")["id"].tolist()[-k:]


def build_possession_sequences(
    possessions: pd.DataFrame,
    k: int = SEQUENCE_LEN,
    out_path: Path | None = SEQUENCE_CACHE_PATH,
    limit: int | None = None,
) -> list[dict]:
    """Assemble one build-up sequence (list of per-frame graphs) per possession.

    Velocity is left zero-filled per frame; the temporal model infers motion from the
    position sequence itself (avoids the variable inter-frame ``dt`` problem).

    Args:
        possessions: The labelled dataset (``data/processed/possessions.parquet``).
        k: Max frames per sequence.
        out_path: Where to ``torch.save`` the records; ``None`` to skip.
        limit: Only the first ``limit`` matches (smoke testing).

    Returns:
        Records ``{match_id, possession, frames:[Data], y_success, y_xt, length}``.
    """
    records: list[dict] = []
    match_ids = possessions["match_id"].unique()
    if limit is not None:
        match_ids = match_ids[:limit]
    for n, mid in enumerate(match_ids, start=1):
        try:
            events = add_event_time(load_events_local(int(mid)))
            frames = load_frames_local(int(mid))
        except Exception as exc:  # noqa: BLE001 - log + skip a match
            logger.warning("sequences: skipping match %s: %s", mid, exc)
            continue
        frames_by_event = dict(tuple(frames.groupby("id")))
        sub = possessions[possessions["match_id"] == mid]
        for row in sub.itertuples(index=False):
            seq = []
            for eid in buildup_event_ids(events, row, k):
                fr = frames_by_event.get(eid)
                if fr is None or len(fr) == 0:
                    continue
                try:
                    seq.append(build_data(fr, row))
                except ValueError:
                    continue
            if not seq:
                continue
            records.append(
                {
                    "match_id": int(row.match_id),
                    "possession": int(row.possession),
                    "frames": seq,
                    "y_success": float(row.success),
                    "y_xt": float(row.xt_progression),
                    "length": len(seq),
                }
            )
        if n % 25 == 0:
            logger.info("sequences: %d/%d matches, %d records", n, len(match_ids), len(records))

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(records, out_path)
        logger.info("Wrote %d sequences to %s", len(records), out_path)
    return records


def load_sequence_cache(path: Path = SEQUENCE_CACHE_PATH) -> list[dict]:
    """Load the cached sequence records."""
    return torch.load(path, weights_only=False)


def main() -> None:
    """Build and cache build-up sequences for the full labelled dataset."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    poss = pd.read_parquet("data/processed/possessions.parquet")
    records = build_possession_sequences(poss)
    lengths = [r["length"] for r in records]
    print(f"\nBuilt {len(records)} sequences -> {SEQUENCE_CACHE_PATH}")
    if lengths:
        import numpy as np

        print(
            f"  length: mean {np.mean(lengths):.2f}  "
            f"median {int(np.median(lengths))}  max {max(lengths)}"
        )


if __name__ == "__main__":
    main()
