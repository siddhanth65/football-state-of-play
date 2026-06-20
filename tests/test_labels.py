"""Unit tests for data.labels (offline, synthetic events + frames)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data import labels as L


# --- defensive labels ------------------------------------------------------
def test_compute_offside_line_second_deepest_defender():
    frame = pd.DataFrame(
        [
            {"location": [100.0, 30.0], "teammate": False, "keeper": False, "actor": False},
            {"location": [106.0, 50.0], "teammate": False, "keeper": False, "actor": False},
            {"location": [118.0, 40.0], "teammate": False, "keeper": True, "actor": False},
            {"location": [90.0, 40.0], "teammate": True, "keeper": False, "actor": True},
        ]
    )
    # Outfield defenders at x=100, 106 (keeper excluded); offside line = 2nd-deepest = 100.
    assert L.compute_offside_line(frame) == 100.0


def test_compute_def_stop_detects_recovery_in_window():
    events = pd.DataFrame(
        {
            "team": ["A", "B", "B"],
            "type": ["Pass", "Interception", "Ball Recovery"],
            "time_s": [10.0, 12.0, 30.0],
        }
    )
    # Defender (team B) interception 2 s after the trigger -> defensive stop within 8 s.
    assert L.compute_def_stop(events, possession_team="A", trigger_time=10.0) is True
    # The 30 s recovery is outside the 8 s window from t=21.
    assert L.compute_def_stop(events, possession_team="A", trigger_time=21.0) is False


# --- xT grid ---------------------------------------------------------------
def test_zone_index_bounds():
    assert L.zone_index(np.array([0.0]), np.array([0.0]))[0] == 0
    # Bottom-right corner clamps to the last zone.
    last = L.zone_index(np.array([120.0]), np.array([80.0]))[0]
    assert last == L.XT_NX * L.XT_NY - 1


def test_xt_grid_builder_rewards_zones_near_goal():
    # Shots that score from near the goal should yield positive xT there.
    events = pd.DataFrame(
        {
            "team": ["A"] * 4,
            "possession_team": ["A"] * 4,
            "type": ["Shot", "Shot", "Pass", "Carry"],
            "shot_outcome": ["Goal", "Off T", np.nan, np.nan],
            "location": [[114.0, 40.0], [114.0, 40.0], [60.0, 40.0], [80.0, 40.0]],
            "pass_end_location": [np.nan, np.nan, [110.0, 40.0], np.nan],
            "carry_end_location": [np.nan, np.nan, np.nan, [90.0, 40.0]],
            "pass_outcome": [np.nan, np.nan, np.nan, np.nan],
        }
    )
    builder = L.XTGridBuilder()
    builder.add(events)
    grid = builder.finalize()
    assert grid.shape == (L.XT_NX, L.XT_NY)
    assert np.isfinite(grid).all()
    # Goalmouth zone (x≈114, y≈40) should carry positive threat.
    assert L.xt_value(114.0, 40.0, grid) > 0.0


def test_xt_value_bilinear_on_known_grid():
    grid = np.array([[0.0, 0.0], [0.0, 4.0]])  # (n_x=2, n_y=2)
    # Far corner cell centre -> its own value.
    corner = L.xt_value(120.0, 80.0, grid)
    assert abs(corner - 4.0) < 1e-9
    # Exact pitch centre sits at the meeting point of all four cells -> mean.
    centre = L.xt_value(60.0, 40.0, grid)
    assert abs(centre - 1.0) < 1e-9


def test_compute_xt_progression_uses_grid():
    grid = np.zeros((L.XT_NX, L.XT_NY))
    grid[15, 6] = 1.0  # a high-threat zone near the opponent goal
    prog = L.compute_xt_progression(60.0, 40.0, 117.0, 43.0, grid)
    assert prog > 0.0


# --- success ---------------------------------------------------------------
def _poss_events(records: list[dict]) -> pd.DataFrame:
    base = {
        "team": "A",
        "type": "Pass",
        "location": None,
        "pass_end_location": np.nan,
        "carry_end_location": np.nan,
    }
    return pd.DataFrame([{**base, **r} for r in records])


def test_success_true_on_shot_within_horizon():
    ev = _poss_events([{"type": "Shot", "time_s": 10.0, "location": [110.0, 40.0]}])
    assert L.compute_success(ev, "A", trigger_time=0.0) is True


def test_success_true_on_box_entry_within_horizon():
    ev = _poss_events([{"type": "Pass", "time_s": 5.0, "location": [108.0, 40.0]}])
    assert L.compute_success(ev, "A", trigger_time=0.0) is True


def test_success_false_when_shot_too_late():
    ev = _poss_events([{"type": "Shot", "time_s": 20.0, "location": [110.0, 40.0]}])
    assert L.compute_success(ev, "A", trigger_time=0.0) is False


def test_success_ignores_other_team():
    ev = _poss_events([{"team": "B", "type": "Shot", "time_s": 5.0, "location": [110.0, 40.0]}])
    assert L.compute_success(ev, "A", trigger_time=0.0) is False


# --- run target (nearest-neighbour) ----------------------------------------
def _frame(rows: list[dict]) -> pd.DataFrame:
    base = {"teammate": True, "actor": False, "keeper": False}
    return pd.DataFrame([{**base, **r} for r in rows])


def test_compute_run_target_matches_furthest_forward_attacker():
    trigger = _frame(
        [
            {"location": [90.0, 40.0], "actor": True},  # ball carrier (excluded)
            {"location": [100.0, 30.0]},  # furthest-forward attacker -> p0
            {"location": [70.0, 50.0]},
            {"location": [60.0, 20.0], "teammate": False},  # defender
        ]
    )
    target = _frame(
        [
            {"location": [104.0, 31.0]},  # the attacker, advanced
            {"location": [72.0, 49.0]},
            {"location": [99.0, 28.0], "teammate": False},  # defender (ignored)
        ]
    )
    run = L.compute_run_target(trigger, target)
    assert run == (104.0, 31.0)


def test_compute_run_target_none_without_teammates_in_target():
    trigger = _frame([{"location": [100.0, 30.0]}])
    target = _frame([{"location": [50.0, 40.0], "teammate": False}])
    assert L.compute_run_target(trigger, target) is None


def test_compute_run_target_none_when_nearest_too_far():
    # Nearest teammate is 40 m away -> beyond MAX_RUN_MATCH_DIST -> spurious -> None.
    trigger = _frame([{"location": [100.0, 30.0]}])
    target = _frame([{"location": [60.0, 30.0]}])
    assert L.compute_run_target(trigger, target) is None


# --- orchestration: ≥10-visible filter -------------------------------------
def test_build_labeled_dataset_applies_visible_filter():
    index = pd.DataFrame(
        [
            {  # enough visible players -> kept
                "match_id": 1,
                "possession": 1,
                "possession_team": "A",
                "period": 1,
                "trigger_event_id": "t1",
                "trigger_time_s": 0.0,
                "trigger_x": 85.0,
                "trigger_y": 40.0,
                "end_x": 110.0,
                "end_y": 40.0,
                "play_pattern": "Regular Play",
                "from_counter": False,
                "from_set_piece": False,
                "run_target_event_id": "r1",
                "n_events": 5,
            },
            {  # only 9 visible -> dropped
                "match_id": 1,
                "possession": 2,
                "possession_team": "A",
                "period": 1,
                "trigger_event_id": "t2",
                "trigger_time_s": 30.0,
                "trigger_x": 88.0,
                "trigger_y": 40.0,
                "end_x": 95.0,
                "end_y": 40.0,
                "play_pattern": "Regular Play",
                "from_counter": False,
                "from_set_piece": False,
                "run_target_event_id": "r2",
                "n_events": 4,
            },
        ]
    )
    events = pd.DataFrame(
        {
            "possession": [1, 1, 2],
            "team": ["A", "A", "A"],
            "type": ["Pass", "Shot", "Pass"],
            "timestamp": ["00:00:00.000", "00:00:05.000", "00:00:30.000"],
            "location": [[85.0, 40.0], [110.0, 40.0], [88.0, 40.0]],
            "pass_end_location": [np.nan, np.nan, np.nan],
            "carry_end_location": [np.nan, np.nan, np.nan],
        }
    )
    # Trigger t1 has 10 visible; t2 has 9. Both run-target frames have teammates.
    frames = pd.DataFrame(
        {
            "id": ["t1"] * 10 + ["t2"] * 9 + ["r1", "r2"],
            "teammate": [True] * 21,
            "actor": [False] * 21,
            "location": [[85.0 + i, 40.0] for i in range(10)]
            + [[88.0 + i, 40.0] for i in range(9)]
            + [[100.0, 41.0], [96.0, 41.0]],
        }
    )
    out = L.build_labeled_dataset(
        index,
        grid=np.zeros((L.XT_NX, L.XT_NY)),
        load_events_fn=lambda mid: events,
        load_frames_fn=lambda mid: frames,
    )
    assert list(out["possession"]) == [1]  # possession 2 dropped (< 10 visible)
    assert out.iloc[0]["n_visible"] == 10
    assert bool(out.iloc[0]["success"]) is True  # shot at t=5 s
    assert out.iloc[0]["run_target_x"] == 100.0
