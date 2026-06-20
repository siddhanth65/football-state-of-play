"""Unit tests for data.load.

Offline by default: statsbombpy is monkeypatched. One real-network integration
test is marked ``network`` and deselected unless you run ``pytest -m network``.
"""

from types import SimpleNamespace

import pandas as pd
import pytest

from data import load


# --- is_target_competition -------------------------------------------------
def test_is_target_competition_matches_case_insensitively():
    assert load.is_target_competition("FIFA World Cup", "2022")
    assert load.is_target_competition("fifa world cup", " 2022 ")
    assert load.is_target_competition("1. Bundesliga", "2023/2024")


def test_is_target_competition_rejects_non_targets():
    assert not load.is_target_competition("Premier League", "2015/2016")
    # Right competition, wrong season -> not a target.
    assert not load.is_target_competition("FIFA World Cup", "2018")


# --- filter_360_competitions -----------------------------------------------
def test_filter_360_competitions_keeps_only_360_rows():
    comps = pd.DataFrame(
        {
            "competition_id": [1, 2, 3],
            "season_id": [10, 20, 30],
            "competition_name": ["A", "B", "C"],
            "season_name": ["2020", "2021", "2022"],
            "match_available_360": ["2021-01-01T00:00:00", None, "2023-01-01T00:00:00"],
        }
    )
    out = load.filter_360_competitions(comps)
    assert list(out["competition_id"]) == [1, 3]
    assert out.index.tolist() == [0, 1]  # index reset


# --- list_available_matches (monkeypatched statsbombpy) --------------------
def _fake_sb(monkeypatch):
    comps = pd.DataFrame(
        {
            "competition_id": [55, 99],
            "season_id": [43, 1],
            "competition_name": ["FIFA World Cup", "Random Cup"],
            "season_name": ["2022", "1999"],
            "match_available_360": ["2022-12-01T00:00:00", None],
        }
    )

    def fake_matches(competition_id, season_id, **kwargs):
        if (competition_id, season_id) == (55, 43):
            return pd.DataFrame(
                {
                    "match_id": [3001, 3002],
                    "competition_id": [55, 55],
                    "season_id": [43, 43],
                    "competition_name": ["FIFA World Cup", "FIFA World Cup"],
                    "season_name": ["2022", "2022"],
                    "match_date": ["2022-12-18", "2022-12-14"],
                    "home_team": ["Argentina", "France"],
                    "away_team": ["France", "Morocco"],
                    "match_status_360": ["available", "unscheduled"],
                }
            )
        raise AssertionError("only the 360 competition should be queried")

    fake = SimpleNamespace(
        competitions=lambda **kw: comps,
        matches=fake_matches,
    )
    monkeypatch.setattr(load, "sb", fake)


def test_list_available_matches_filters_and_tags(monkeypatch):
    _fake_sb(monkeypatch)
    out = load.list_available_matches()

    # Only the 360-available competition is queried, only "available" matches kept.
    assert list(out["match_id"]) == [3001]
    assert bool(out.loc[0, "is_target"]) is True
    assert set(_required_cols()).issubset(out.columns)


def _required_cols():
    return [*load._MATCH_COLUMNS, "is_target"]


# --- cache round-trip ------------------------------------------------------
def test_cache_and_load_round_trip(tmp_path):
    matches = pd.DataFrame(
        {
            "match_id": [7, 8],
            "competition_id": [55, 55],
            "season_id": [43, 43],
            "competition_name": ["FIFA World Cup", "FIFA World Cup"],
            "season_name": ["2022", "2022"],
            "match_date": ["2022-12-18", "2022-12-14"],
            "home_team": ["Argentina", "France"],
            "away_team": ["France", "Morocco"],
            "match_status_360": ["available", "available"],
            "is_target": [True, True],
        }
    )
    out_path = tmp_path / "match_ids.json"
    load.cache_match_ids(matches, out_path=out_path)

    records = load.load_cached_match_ids(out_path)
    assert len(records) == 2
    assert records[0]["match_id"] == 7
    assert records[0]["competition_name"] == "FIFA World Cup"


def test_load_cached_match_ids_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load.load_cached_match_ids(tmp_path / "nope.json")


# --- real network (deselected by default) ----------------------------------
@pytest.mark.network
def test_list_360_competitions_live():
    comps = load.list_360_competitions()
    assert not comps.empty
    assert comps["match_available_360"].notna().all()
