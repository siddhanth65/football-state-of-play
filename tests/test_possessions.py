"""Unit tests for data.possessions (offline, synthetic events)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data import possessions as P

_DEFAULTS = {
    "possession": 1,
    "possession_team": "A",
    "team": "A",
    "location": None,
    "timestamp": "00:00:00.000",
    "period": 1,
    "id": "e",
    "play_pattern": "Regular Play",
    "pass_type": np.nan,
    "shot_type": np.nan,
    "type": "Pass",
}


def test_actor_role_bucket_maps_positions():
    assert P.actor_role_bucket("Goalkeeper") == "GK"
    assert P.actor_role_bucket("Right Center Back") == "DEF"
    assert P.actor_role_bucket("Left Wing Back") == "DEF"  # "back" wins
    assert P.actor_role_bucket("Center Defensive Midfield") == "MID"
    assert P.actor_role_bucket("Center Forward") == "FWD"
    assert P.actor_role_bucket("Right Wing") == "FWD"  # winger -> forward
    assert P.actor_role_bucket(None) is None
    assert P.actor_role_bucket("") is None


def make_events(records: list[dict]) -> pd.DataFrame:
    """Build a StatsBomb-like event frame, filling unspecified columns."""
    rows = []
    for i, rec in enumerate(records):
        row = {**_DEFAULTS, **rec}
        row.setdefault("index", i)
        rows.append(row)
    return pd.DataFrame(rows)


# --- helpers ---------------------------------------------------------------
def test_parse_timestamp():
    assert P.parse_timestamp("00:00:08.123") == 8.123
    assert P.parse_timestamp("00:01:30.000") == 90.0
    assert P.parse_timestamp("01:00:00") == 3600.0


def test_xy_handles_lists_and_nan():
    assert P.xy([85.0, 40.0]) == (85.0, 40.0)
    assert P.xy([85.0, 40.0, 0.5]) == (85.0, 40.0)  # 3D goalkeeper location
    assert P.xy(np.nan) is None
    assert P.xy(None) is None


# --- build_match_possessions ----------------------------------------------
def test_trigger_is_first_attacking_third_event():
    events = make_events(
        [
            {"index": 0, "location": [50.0, 40.0], "timestamp": "00:00:00.000", "id": "e0"},
            {"index": 1, "location": [85.0, 40.0], "timestamp": "00:00:01.000", "id": "e1"},
            {"index": 2, "location": [95.0, 45.0], "timestamp": "00:00:02.500", "id": "e2"},
            {"index": 3, "location": [100.0, 50.0], "timestamp": "00:00:04.000", "id": "e3"},
        ]
    )
    out = P.build_match_possessions(events, match_id=7)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["trigger_event_id"] == "e1"  # first x >= 80, not e2
    assert row["trigger_x"] == 85.0
    assert row["end_x"] == 100.0  # last located event
    assert row["run_target_event_id"] == "e2"  # closest to trigger + 1.5 s
    assert not row["from_counter"]
    assert not row["from_set_piece"]
    assert row["match_id"] == 7


def test_possession_never_reaching_attacking_third_is_dropped():
    events = make_events(
        [
            {"index": 0, "location": [30.0, 40.0], "id": "e0"},
            {"index": 1, "location": [55.0, 40.0], "id": "e1"},
        ]
    )
    assert P.build_match_possessions(events, match_id=1).empty


def test_from_counter_tag():
    events = make_events(
        [{"index": 0, "location": [90.0, 40.0], "play_pattern": "From Counter", "id": "e0"}]
    )
    row = P.build_match_possessions(events, match_id=1).iloc[0]
    assert row["from_counter"]
    assert not row["from_set_piece"]  # From Counter is open play


def test_from_set_piece_via_corner_restart():
    # The corner pass itself is the trigger -> set-piece restart at dt = 0.
    events = make_events(
        [{"index": 0, "location": [118.0, 1.0], "pass_type": "Corner", "id": "e0"}]
    )
    assert P.build_match_possessions(events, match_id=1).iloc[0]["from_set_piece"]


def test_set_piece_origin_open_play_not_excluded():
    # Started from a throw-in deep, builds to the attacking third 25 s later.
    # play_pattern stays "From Throw In" but it is open play by the trigger, so it
    # must NOT be tagged as a set piece (the over-exclusion the proximity rule fixes).
    events = make_events(
        [
            {
                "index": 0,
                "location": [30.0, 1.0],
                "pass_type": "Throw-in",
                "play_pattern": "From Throw In",
                "timestamp": "00:00:00.000",
                "id": "e0",
            },
            {
                "index": 1,
                "location": [88.0, 40.0],
                "play_pattern": "From Throw In",
                "timestamp": "00:00:25.000",
                "id": "e1",
            },
        ]
    )
    row = P.build_match_possessions(events, match_id=1).iloc[0]
    assert row["trigger_event_id"] == "e1"
    assert not row["from_set_piece"]


def test_from_set_piece_via_proximity_window():
    # A free-kick restart at t=0, then an open-play trigger 3 s later (< 5 s).
    events = make_events(
        [
            {
                "index": 0,
                "location": [50.0, 40.0],
                "pass_type": "Free Kick",
                "timestamp": "00:00:00.000",
                "id": "e0",
            },
            {"index": 1, "location": [88.0, 40.0], "timestamp": "00:00:03.000", "id": "e1"},
        ]
    )
    assert P.build_match_possessions(events, match_id=1).iloc[0]["from_set_piece"]


def test_next_pass_end_is_first_pass_at_or_after_trigger():
    events = make_events(
        [
            {
                "index": 0,
                "location": [50.0, 40.0],
                "timestamp": "00:00:00.000",
                "id": "e0",
                "pass_end_location": [60.0, 40.0],
            },
            {
                "index": 1,
                "location": [85.0, 40.0],
                "timestamp": "00:00:01.000",
                "id": "e1",
                "pass_end_location": [100.0, 50.0],
            },
        ]
    )
    row = P.build_match_possessions(events, match_id=1).iloc[0]
    # The trigger pass itself counts; the pre-trigger pass does not.
    assert row["next_pass_end_x"] == 100.0
    assert row["next_pass_end_y"] == 50.0
    # And the located event before the trigger becomes the velocity source.
    assert row["prev_event_id"] == "e0"
    assert abs(row["prev_dt_s"] - 1.0) < 1e-9


def test_prev_event_dropped_when_too_old():
    events = make_events(
        [
            {"index": 0, "location": [50.0, 40.0], "timestamp": "00:00:00.000", "id": "e0"},
            {"index": 1, "location": [85.0, 40.0], "timestamp": "00:00:10.000", "id": "e1"},
        ]
    )
    row = P.build_match_possessions(events, match_id=1).iloc[0]
    assert row["prev_event_id"] is None
    assert np.isnan(row["prev_dt_s"])


def test_run_target_dropped_when_no_event_within_tolerance():
    # Only event after the trigger is 5 s later -> outside the 2 s tolerance.
    events = make_events(
        [
            {"index": 0, "location": [85.0, 40.0], "timestamp": "00:00:01.000", "id": "e0"},
            {"index": 1, "location": [95.0, 40.0], "timestamp": "00:00:06.000", "id": "e1"},
        ]
    )
    row = P.build_match_possessions(events, match_id=1).iloc[0]
    assert row["run_target_event_id"] is None
