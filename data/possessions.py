"""Possession-chain construction from StatsBomb events (CLAUDE.md §4.3).

Week 2 responsibility: turn the raw event stream into one row per
**attacking-third possession**, with its trigger event identified and tagged. The
output index is the substrate the label stage (:mod:`data.labels`) joins freeze
frames onto.

This module is event-only — it never loads 360 frames. It also owns the shared
geometry / horizon **constants** (the resolved §3 scope decisions, kept as named
constants so a change is a one-line edit) and time helpers, which
:mod:`data.labels` imports. Keeping the dependency one-directional
(``labels`` → ``possessions``) avoids an import cycle.

Run as ``python -m data.possessions`` (or ``make data``) to build and cache
``data/processed/possessions_index.parquet``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from data import load

logger = logging.getLogger(__name__)

# --- Paths -----------------------------------------------------------------
PROCESSED_DIR = Path("data/processed")
POSSESSIONS_INDEX_PATH = PROCESSED_DIR / "possessions_index.parquet"

# --- Pitch geometry (StatsBomb 120 x 80, attacking left -> right) ----------
PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0
ATTACKING_THIRD_X = 80.0
BOX_X_MIN = 102.0
BOX_Y_MIN = 18.0
BOX_Y_MAX = 62.0

# --- Resolved §3 scope decisions (week1.md, locked 2026-06-08) -------------
SUCCESS_SHOT_HORIZON_S = 15.0  # #1: shot within 15 s ...
SUCCESS_BOX_HORIZON_S = 10.0  # #1: ... OR penalty-area entry within 10 s
RUN_HORIZON_S = 1.5  # #2: attacker run target at t + 1.5 s
RUN_TOLERANCE_S = 2.0  # drop the row if no frame within this of t+1.5 s
# #3 ("furthest-forward attacker, excl. carrier") and #4 ("first attacking-third
# event") are implemented in the trigger / run-target logic below + in data.labels.

# Run target via the §3 #3 *alternative* ("the attacker the carrier plays the next
# pass to"): the end location of the first possession-team pass at/after the trigger,
# within this window. Cleaner than nearest-neighbour matching (360 has no player IDs)
# and the basis for the H_receiver node label (data.graphs). Power et al. 2017.
NEXT_PASS_HORIZON_S = 15.0
# Previous-frame velocity (data.graphs): use the last possession-team on-ball event
# before the trigger, but only if it is recent enough that differencing is meaningful.
PREV_FRAME_MAX_DT_S = 5.0

# --- Filtering parameters --------------------------------------------------
MIN_VISIBLE_PLAYERS = 10  # applied at the label stage (needs frames)
SETPIECE_EXCLUSION_S = 5.0  # exclude triggers within 5 s of a set-piece (open play)

COUNTER_PATTERN = "From Counter"
SET_PIECE_PATTERNS = frozenset(
    {"From Corner", "From Free Kick", "From Throw In", "From Goal Kick", "From Kick Off"}
)
SET_PIECE_PASS_TYPES = frozenset({"Corner", "Free Kick", "Throw-in", "Goal Kick", "Kick Off"})
SET_PIECE_SHOT_TYPES = frozenset({"Free Kick", "Penalty"})

# Coarse role buckets for the ball-carrier's StatsBomb position (the only player whose
# role is known — 360 frames carry no identity). Used as a graph-level feature.
ROLE_BUCKETS = ("GK", "DEF", "MID", "FWD")


def actor_role_bucket(position: object) -> str | None:
    """Map a StatsBomb position name to a coarse GK/DEF/MID/FWD bucket (or None)."""
    if not isinstance(position, str) or not position:
        return None
    p = position.lower()
    if "keeper" in p:
        return "GK"
    if "back" in p:  # Center/Right/Left Back, Wing Back
        return "DEF"
    if "midfield" in p:
        return "MID"
    if "forward" in p or "striker" in p or "wing" in p:  # wingers count as forwards
        return "FWD"
    return None


def parse_timestamp(timestamp: str) -> float:
    """Parse a StatsBomb ``"HH:MM:SS.mmm"`` timestamp to seconds.

    Args:
        timestamp: Event ``timestamp`` string (period-relative clock).

    Returns:
        Seconds since the start of the period as a float.
    """
    hours, minutes, seconds = timestamp.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def xy(cell: object) -> tuple[float, float] | None:
    """Extract ``(x, y)`` from a StatsBomb location cell.

    Args:
        cell: A ``[x, y]`` list (or ``[x, y, z]`` for some events), or NaN/None.

    Returns:
        ``(x, y)`` floats, or ``None`` if the cell has no location.
    """
    if isinstance(cell, (list, tuple)) and len(cell) >= 2:
        return float(cell[0]), float(cell[1])
    return None


def in_attacking_third(x: float) -> bool:
    """Whether a pitch x-coordinate is in the attacking third."""
    return x >= ATTACKING_THIRD_X


def in_penalty_area(x: float, y: float) -> bool:
    """Whether ``(x, y)`` is inside the attacked penalty area."""
    return x >= BOX_X_MIN and BOX_Y_MIN <= y <= BOX_Y_MAX


def add_event_time(events: pd.DataFrame) -> pd.DataFrame:
    """Return ``events`` with a ``time_s`` column (period-relative seconds).

    Possessions never cross a period boundary, so period-relative seconds are
    sufficient for the intra-possession horizon comparisons used downstream.

    Args:
        events: A StatsBomb event DataFrame with a ``timestamp`` column.

    Returns:
        A copy with an added float ``time_s`` column.
    """
    out = events.copy()
    out["time_s"] = out["timestamp"].map(parse_timestamp)
    return out


def _is_set_piece_restart(row: pd.Series) -> bool:
    """Whether a single event is a set-piece *restart action*.

    Detects the restart pass/shot itself (corner, free kick, throw-in, goal kick,
    kick-off, penalty) via ``pass_type``/``shot_type`` — deliberately NOT the
    possession-level ``play_pattern``, which tags every event of a
    set-piece-origin possession and would over-exclude long open-play phases that
    merely *started* from a restart.
    """
    if row.get("pass_type") in SET_PIECE_PASS_TYPES:
        return True
    return row.get("shot_type") in SET_PIECE_SHOT_TYPES


def build_match_possessions(events: pd.DataFrame, match_id: int) -> pd.DataFrame:
    """Build the attacking-third possession index for one match.

    For each ``possession`` index, the trigger is the first on-ball event by the
    possession team in the attacking third (§3 #4). The possession is kept only if
    such a trigger exists. Each row records the trigger, the possession end
    location, the run-target source event (≈ trigger + 1.5 s), and the
    counter / set-piece tags.

    Args:
        events: One match's event DataFrame (from :func:`data.load.load_events`).
        match_id: StatsBomb match id (events may omit it per row).

    Returns:
        One row per kept possession. Empty (typed) frame if none qualify.
    """
    events = add_event_time(events)
    rows: list[dict] = []

    for possession, grp in events.groupby("possession", sort=True):
        grp = grp.sort_values("index")
        team = grp["possession_team"].dropna()
        if team.empty:
            continue
        possession_team = team.iloc[0]

        # On-ball events by the possession team that carry a location.
        on_ball = grp[(grp["team"] == possession_team) & grp["location"].notna()]
        if on_ball.empty:
            continue

        locs = on_ball["location"].map(xy)
        on_ball = on_ball.assign(
            _x=[p[0] if p else np.nan for p in locs],
            _y=[p[1] if p else np.nan for p in locs],
        ).dropna(subset=["_x", "_y"])
        if on_ball.empty:
            continue

        # Trigger: first attacking-third on-ball event (§3 #4).
        att = on_ball[on_ball["_x"] >= ATTACKING_THIRD_X]
        if att.empty:
            continue
        trigger = att.iloc[0]
        trigger_time = float(trigger["time_s"])

        # Possession end location: last located on-ball event of the possession.
        end = on_ball.iloc[-1]

        # Set-piece tag: trigger within SETPIECE_EXCLUSION_S of an actual restart
        # action in this possession (Fernández–Bornn 2021). We deliberately do NOT
        # exclude on the possession-level play_pattern, which would drop entire
        # open-play phases that merely started from a throw-in / goal kick.
        sp_mask = grp.apply(_is_set_piece_restart, axis=1)
        sp_times = grp.loc[sp_mask, "time_s"]
        from_set_piece = bool(
            (
                (trigger_time - sp_times >= 0) & (trigger_time - sp_times <= SETPIECE_EXCLUSION_S)
            ).any()
        )

        # Run-target source: possession-team event closest to trigger + 1.5 s,
        # within tolerance (so the freeze frame's `teammate` flag means attackers).
        later = on_ball[on_ball["time_s"] > trigger_time].copy()
        run_target_event_id = None
        if not later.empty:
            target_t = trigger_time + RUN_HORIZON_S
            later["_dt"] = (later["time_s"] - target_t).abs()
            best = later.sort_values("_dt").iloc[0]
            if float(best["_dt"]) <= RUN_TOLERANCE_S:
                run_target_event_id = best["id"]

        # Next pass (§3 #3 alternative + the H_receiver label, Power et al. 2017):
        # the first *completed* possession-team pass at/after the trigger with an end
        # location. Restricting to completed passes (StatsBomb: NaN ``pass_outcome``)
        # means the end location is the receiver's actual position, so the nearest-teammate
        # receiver label in data.graphs is the real recipient, not a guess at the line of
        # a failed pass.
        next_pass_end_x, next_pass_end_y = np.nan, np.nan
        if "pass_end_location" in grp.columns:
            passes = grp[
                (grp["team"] == possession_team)
                & (grp["type"] == "Pass")
                & (grp["time_s"] >= trigger_time)
                & (grp["time_s"] <= trigger_time + NEXT_PASS_HORIZON_S)
            ]
            if "pass_outcome" in passes.columns:
                passes = passes[passes["pass_outcome"].isna()]  # completed only
            for cell in passes["pass_end_location"]:
                p = xy(cell)
                if p is not None:
                    next_pass_end_x, next_pass_end_y = p
                    break

        # Completion of the *first* pass after the trigger (any outcome) — the execution
        # signal behind the risk-adjusted xT target (xT_gain x P(complete); Paul et al. 2025).
        next_pass_completed = np.nan
        if "pass_end_location" in grp.columns:
            first = grp[
                (grp["team"] == possession_team)
                & (grp["type"] == "Pass")
                & (grp["time_s"] >= trigger_time)
                & (grp["time_s"] <= trigger_time + NEXT_PASS_HORIZON_S)
            ].sort_values("time_s")
            if not first.empty and "pass_outcome" in first.columns:
                next_pass_completed = float(pd.isna(first.iloc[0]["pass_outcome"]))

        # Ball-carrier role bucket (the only known position; 360 has no per-dot identity).
        actor_role = None
        if "position" in grp.columns:
            actor_role = actor_role_bucket(trigger.get("position"))

        # Previous on-ball event (velocity differencing in data.graphs): the last
        # possession-team located event before the trigger, if recent enough that
        # frame-to-frame differencing is meaningful.
        prev_event_id, prev_dt_s = None, np.nan
        before = on_ball[on_ball["time_s"] < trigger_time]
        if not before.empty:
            prev = before.iloc[-1]
            dt = trigger_time - float(prev["time_s"])
            if 0.0 < dt <= PREV_FRAME_MAX_DT_S:
                prev_event_id, prev_dt_s = prev["id"], dt

        rows.append(
            {
                "match_id": match_id,
                "possession": int(possession),
                "possession_team": possession_team,
                "period": int(trigger["period"]),
                "trigger_event_id": trigger["id"],
                "trigger_time_s": trigger_time,
                "trigger_x": float(trigger["_x"]),
                "trigger_y": float(trigger["_y"]),
                "end_x": float(end["_x"]),
                "end_y": float(end["_y"]),
                "play_pattern": trigger["play_pattern"],
                "from_counter": trigger["play_pattern"] == COUNTER_PATTERN,
                "from_set_piece": bool(from_set_piece),
                "run_target_event_id": run_target_event_id,
                "next_pass_end_x": next_pass_end_x,
                "next_pass_end_y": next_pass_end_y,
                "next_pass_completed": next_pass_completed,
                "actor_role": actor_role,
                "prev_event_id": prev_event_id,
                "prev_dt_s": prev_dt_s,
                "n_events": int(len(grp)),
            }
        )

    return pd.DataFrame(rows, columns=_INDEX_COLUMNS)


_INDEX_COLUMNS = [
    "match_id",
    "possession",
    "possession_team",
    "period",
    "trigger_event_id",
    "trigger_time_s",
    "trigger_x",
    "trigger_y",
    "end_x",
    "end_y",
    "play_pattern",
    "from_counter",
    "from_set_piece",
    "run_target_event_id",
    "next_pass_end_x",
    "next_pass_end_y",
    "next_pass_completed",
    "actor_role",
    "prev_event_id",
    "prev_dt_s",
    "n_events",
]


def build_possessions_index(
    match_records: list[dict] | None = None, limit: int | None = None
) -> pd.DataFrame:
    """Build the pooled attacking-third possession index across all 360 matches.

    Scope is **all** 360-available matches (the extras are pooled in per the Week-1
    decision); ``is_target`` and competition identity are carried through for slice
    analysis. Matches that fail to load are logged and skipped.

    Args:
        match_records: Cached match records; if ``None``, loads via
            :func:`data.load.load_cached_match_ids`.
        limit: If set, only process the first ``limit`` matches (smoke testing).

    Returns:
        The concatenated possession index for all processed matches.
    """
    if match_records is None:
        match_records = load.load_cached_match_ids()
    if limit is not None:
        match_records = match_records[:limit]

    frames: list[pd.DataFrame] = []
    for i, rec in enumerate(match_records, start=1):
        mid = rec["match_id"]
        try:
            events = load.load_events_local(mid)
        except Exception as exc:  # noqa: BLE001 - log and continue per match
            logger.warning("Failed to load events for match %s: %s", mid, exc)
            continue
        match_poss = build_match_possessions(events, mid)
        if match_poss.empty:
            continue
        match_poss["competition_name"] = rec.get("competition_name")
        match_poss["season_name"] = rec.get("season_name")
        match_poss["is_target"] = bool(rec.get("is_target", False))
        frames.append(match_poss)
        if i % 25 == 0:
            logger.info("Processed %d/%d matches", i, len(match_records))

    if not frames:
        return pd.DataFrame(
            columns=[*_INDEX_COLUMNS, "competition_name", "season_name", "is_target"]
        )
    return pd.concat(frames, ignore_index=True)


def cache_possessions_index(index: pd.DataFrame, out_path: Path = POSSESSIONS_INDEX_PATH) -> Path:
    """Write the possession index to parquet."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    index.to_parquet(out_path, index=False)
    logger.info("Wrote %d possessions to %s", len(index), out_path)
    return out_path


def _parse_limit(argv: list[str]) -> int | None:
    """Parse an optional ``--limit N`` from argv (for smoke runs)."""
    if "--limit" in argv:
        return int(argv[argv.index("--limit") + 1])
    return None


def main() -> None:
    """Build and cache the pooled attacking-third possession index."""
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    limit = _parse_limit(sys.argv)
    index = build_possessions_index(limit=limit)
    cache_possessions_index(index)

    n = len(index)
    print(f"\nAttacking-third possessions: {n}")
    if n:
        print(f"  counters (From Counter):   {int(index['from_counter'].sum())}")
        print(f"  from set piece:            {int(index['from_set_piece'].sum())}")
        print(f"  open play (kept for model): {int((~index['from_set_piece']).sum())}")
        print(f"  with run-target frame:     {int(index['run_target_event_id'].notna().sum())}")


if __name__ == "__main__":
    main()
