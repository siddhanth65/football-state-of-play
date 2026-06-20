"""Match listing and event/frame loading from StatsBomb open data.

Week 1 responsibility (CLAUDE.md §4.1, §10): enumerate the competitions that have
360 freeze-frame data, list their matches, and cache a clean match-ID list to
``data/processed/match_ids.json``.

Uses :mod:`statsbombpy`, which reads the free open data over the network (no
credentials needed). The heavier per-match event/frame loaders here are thin
wrappers used by later weeks.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from statsbombpy import sb

logger = logging.getLogger(__name__)

# --- Paths -----------------------------------------------------------------
PROCESSED_DIR = Path("data/processed")
MATCH_IDS_PATH = PROCESSED_DIR / "match_ids.json"

# Local open-data clone (no network): the event/frame builders read these directly,
# so `make data` is reproducible offline and does not depend on statsbombpy hitting
# GitHub for ~850 requests (slow + fragile on long unattended runs).
LOCAL_EVENTS_DIR = Path("data/raw/open-data/data/events")
LOCAL_360_DIR = Path("data/raw/open-data/data/three-sixty")

# --- Target competitions (CLAUDE.md §4.1) ----------------------------------
# These are the *intended* competitions. Exact StatsBomb names/seasons are to be
# verified in the Week 1 EDA, so we never silently drop anything: every
# 360-available match is listed, with an ``is_target`` flag computed from this
# table. Confirm the final scope in docs/meeting_notes/week1.md.
TARGET_COMPETITIONS: list[dict[str, str]] = [
    {"competition_name": "UEFA Euro", "season_name": "2020"},
    {"competition_name": "UEFA Euro", "season_name": "2024"},
    {"competition_name": "UEFA Women's Euro", "season_name": "2022"},
    {"competition_name": "UEFA Women's Euro", "season_name": "2025"},
    {"competition_name": "FIFA World Cup", "season_name": "2022"},
    {"competition_name": "Women's World Cup", "season_name": "2023"},
    {"competition_name": "1. Bundesliga", "season_name": "2023/2024"},
]

# statsbombpy marks per-match 360 availability with this status string.
MATCH_STATUS_360_AVAILABLE = "available"

# Minimal columns kept for the cached match list.
_MATCH_COLUMNS = [
    "match_id",
    "competition_id",
    "season_id",
    "competition_name",
    "season_name",
    "match_date",
    "home_team",
    "away_team",
    "match_status_360",
]


def _norm(value: object) -> str:
    """Normalise a label for case-insensitive comparison."""
    return str(value).strip().casefold()


def is_target_competition(competition_name: str, season_name: str) -> bool:
    """Return whether a competition-season is in the intended target set.

    Args:
        competition_name: StatsBomb ``competition_name``.
        season_name: StatsBomb ``season_name`` (e.g. ``"2023/2024"``).

    Returns:
        True if ``(competition_name, season_name)`` matches a
        :data:`TARGET_COMPETITIONS` entry, case-insensitively.
    """
    name, season = _norm(competition_name), _norm(season_name)
    return any(
        _norm(t["competition_name"]) == name and _norm(t["season_name"]) == season
        for t in TARGET_COMPETITIONS
    )


def filter_360_competitions(competitions: pd.DataFrame) -> pd.DataFrame:
    """Keep only competition-seasons that have 360 freeze-frame data.

    Args:
        competitions: Output of :func:`statsbombpy.sb.competitions`.

    Returns:
        The subset of rows where ``match_available_360`` is populated, with the
        index reset.
    """
    mask = competitions["match_available_360"].notna()
    return competitions.loc[mask].reset_index(drop=True)


def list_360_competitions() -> pd.DataFrame:
    """List all competition-seasons with 360 data available.

    Returns:
        A DataFrame of 360-available competition-seasons (see
        :func:`filter_360_competitions`).
    """
    return filter_360_competitions(sb.competitions())


def list_available_matches(competitions: pd.DataFrame | None = None) -> pd.DataFrame:
    """List every match that has 360 freeze-frame data available.

    Iterates the 360-available competition-seasons and keeps matches whose
    ``match_status_360`` is ``"available"``. Each row is tagged with
    ``is_target`` (see :func:`is_target_competition`). Competitions that fail to
    load are logged and skipped rather than aborting the whole listing.

    Args:
        competitions: Pre-fetched 360 competition table; if ``None``,
            :func:`list_360_competitions` is called.

    Returns:
        A DataFrame with one row per 360-available match (columns
        :data:`_MATCH_COLUMNS` plus ``is_target``), sorted by competition,
        season, then match date.
    """
    if competitions is None:
        competitions = list_360_competitions()

    frames: list[pd.DataFrame] = []
    for row in competitions.itertuples(index=False):
        try:
            matches = sb.matches(competition_id=row.competition_id, season_id=row.season_id)
        except Exception as exc:  # noqa: BLE001 - log and continue per competition
            logger.warning(
                "Failed to load matches for %s %s: %s",
                row.competition_name,
                row.season_name,
                exc,
            )
            continue

        if "match_status_360" not in matches.columns:
            continue
        available = matches.loc[matches["match_status_360"] == MATCH_STATUS_360_AVAILABLE].copy()
        if available.empty:
            continue

        # Source competition/season identity from the competition row. sb.matches
        # exposes `season` (not `season_name`) and inconsistent id columns, so we
        # set these explicitly to avoid silent name-mismatch gaps.
        available["competition_id"] = row.competition_id
        available["season_id"] = row.season_id
        available["competition_name"] = row.competition_name
        available["season_name"] = row.season_name
        for col in _MATCH_COLUMNS:
            if col not in available.columns:
                available[col] = pd.NA
        frames.append(available[_MATCH_COLUMNS])

    if not frames:
        return pd.DataFrame(columns=[*_MATCH_COLUMNS, "is_target"])

    out = pd.concat(frames, ignore_index=True)
    out["is_target"] = [
        is_target_competition(c, s)
        for c, s in zip(out["competition_name"], out["season_name"], strict=True)
    ]
    return out.sort_values(["competition_name", "season_name", "match_date"]).reset_index(drop=True)


def cache_match_ids(matches: pd.DataFrame | None = None, out_path: Path = MATCH_IDS_PATH) -> Path:
    """Write the 360 match list to JSON.

    Args:
        matches: Match table to cache; if ``None``,
            :func:`list_available_matches` is called.
        out_path: Destination JSON path.

    Returns:
        The path written.
    """
    if matches is None:
        matches = list_available_matches()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = matches.astype(object).where(matches.notna(), None).to_dict("records")
    out_path.write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote %d matches to %s", len(records), out_path)
    return out_path


def load_cached_match_ids(path: Path = MATCH_IDS_PATH) -> list[dict]:
    """Load the cached match list.

    Args:
        path: Path written by :func:`cache_match_ids`.

    Returns:
        The list of match records.

    Raises:
        FileNotFoundError: If the cache does not exist yet.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m data.load` (or `make match-ids`) first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def load_events(match_id: int) -> pd.DataFrame:
    """Load the full event stream for a match.

    Args:
        match_id: StatsBomb match id.

    Returns:
        The event DataFrame from :func:`statsbombpy.sb.events`.
    """
    return sb.events(match_id=match_id)


def load_frames(match_id: int) -> pd.DataFrame:
    """Load the 360 freeze-frame data for a match.

    Args:
        match_id: StatsBomb match id.

    Returns:
        The freeze-frame DataFrame from :func:`statsbombpy.sb.frames` (one row
        per visible player per event).
    """
    return sb.frames(match_id=match_id, fmt="dataframe")


def _name(value: object) -> object:
    """Extract a nested StatsBomb ``{"id":.., "name":..}`` name, or ``None``."""
    return value.get("name") if isinstance(value, dict) else None


def load_events_local(match_id: int) -> pd.DataFrame:
    """Load + flatten a match's events from the local clone (no network).

    Mirrors the subset of :func:`statsbombpy.sb.events` columns the possession and
    label pipelines consume, reading ``data/raw/open-data/data/events/{id}.json``
    directly. Nested ``{"id","name"}`` objects are flattened to their name and the
    ``pass``/``carry``/``shot`` sub-dicts to the ``<group>_<field>`` columns
    statsbombpy uses (a missing ``pass.outcome`` ⇒ NaN ⇒ a *completed* pass).

    Args:
        match_id: StatsBomb match id.

    Returns:
        The flattened event DataFrame.
    """
    records = json.loads((LOCAL_EVENTS_DIR / f"{int(match_id)}.json").read_text(encoding="utf-8"))
    rows = []
    for e in records:
        p = e.get("pass") or {}
        c = e.get("carry") or {}
        s = e.get("shot") or {}
        rows.append(
            {
                "id": e.get("id"),
                "index": e.get("index"),
                "period": e.get("period"),
                "timestamp": e.get("timestamp"),
                "possession": e.get("possession"),
                "type": _name(e.get("type")),
                "possession_team": _name(e.get("possession_team")),
                "team": _name(e.get("team")),
                "play_pattern": _name(e.get("play_pattern")),
                "position": _name(e.get("position")),  # the actor's role on this event
                "location": e.get("location"),
                "pass_end_location": p.get("end_location"),
                "pass_outcome": _name(p.get("outcome")),
                "pass_type": _name(p.get("type")),
                "carry_end_location": c.get("end_location"),
                "shot_outcome": _name(s.get("outcome")),
                "shot_type": _name(s.get("type")),
            }
        )
    return pd.DataFrame(rows)


def load_frames_local(match_id: int) -> pd.DataFrame:
    """Load a match's 360 freeze frames from the local clone (no network).

    Returns one row per visible player (``id``/``teammate``/``actor``/``keeper``/
    ``location``), matching the shape the graph/label builders expect. Avoids
    ``statsbombpy`` (which also triggers a torch_geometric DLL crash on Windows when
    imported after it).

    Args:
        match_id: StatsBomb match id.

    Returns:
        The freeze-frame DataFrame.
    """
    records = json.loads((LOCAL_360_DIR / f"{int(match_id)}.json").read_text(encoding="utf-8"))
    rows = [
        {
            "id": rec.get("event_uuid") or rec.get("id"),
            "teammate": player.get("teammate"),
            "actor": player.get("actor"),
            "keeper": player.get("keeper"),
            "location": player.get("location"),
        }
        for rec in records
        for player in rec.get("freeze_frame", [])
    ]
    return pd.DataFrame(rows, columns=["id", "teammate", "actor", "keeper", "location"])


def main() -> None:
    """Build and cache the 360 match list, printing a summary."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    matches = list_available_matches()
    cache_match_ids(matches)

    total = len(matches)
    n_target = int(matches["is_target"].sum()) if total else 0
    print(f"\n360-available matches: {total}  (target-scope: {n_target})\n")
    if total:
        summary = (
            matches.groupby(["competition_name", "season_name", "is_target"])
            .size()
            .reset_index(name="matches")
            .sort_values("matches", ascending=False)
        )
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
