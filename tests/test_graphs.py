"""Unit tests for data.graphs (offline, synthetic frame + row)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from data import graphs as G


def _frame(rows: list[dict]) -> pd.DataFrame:
    base = {"teammate": True, "actor": False, "keeper": False}
    return pd.DataFrame([{**base, **r} for r in rows])


def _row(**kw) -> SimpleNamespace:
    base = {
        "trigger_x": 90.0,
        "trigger_y": 40.0,
        "play_pattern": "Regular Play",
        "from_counter": False,
        "trigger_time_s": 600.0,
    }
    return SimpleNamespace(**{**base, **kw})


def _basic_frame() -> pd.DataFrame:
    return _frame(
        [
            {"location": [90.0, 40.0], "actor": True},  # ball carrier (attacker)
            {"location": [100.0, 30.0]},  # attacker
            {"location": [105.0, 50.0], "teammate": False},  # defender
            {"location": [118.0, 40.0], "teammate": False, "keeper": True},  # keeper
        ]
    )


# --- extraction ------------------------------------------------------------
def test_frame_points_extracts_positions_and_flags():
    pts, tm, ac, kp = G.frame_points(_basic_frame())
    assert pts.shape == (4, 2)
    assert tm.tolist() == [True, True, False, False]
    assert ac.tolist() == [True, False, False, False]
    assert kp.tolist() == [False, False, False, True]


# --- node features ---------------------------------------------------------
def test_node_features_shape_and_normalisation():
    pts, tm, ac, kp = G.frame_points(_basic_frame())
    ball = pts[ac][0]
    feats = G.node_features(pts, tm, ac, kp, ball, np.zeros_like(pts))
    assert feats.shape == (4, G.N_NODE_FEATURES)
    # x, y normalised into [0, 1]; angle columns into [-1, 1].
    assert (feats[:, 0] >= 0).all() and (feats[:, 0] <= 1).all()
    assert (np.abs(feats[:, 10]) <= 1).all()
    # The actor row carries the is_actor flag.
    assert feats[0, 5] == 1.0


# --- edge features ---------------------------------------------------------
def test_edge_features_all_pairs_directed():
    pts, tm, _, _ = G.frame_points(_basic_frame())
    edge_index, edge_attr = G.edge_features(pts, tm, np.zeros_like(pts))
    n = len(pts)
    assert edge_index.shape == (2, n * (n - 1))
    assert edge_attr.shape == (n * (n - 1), G.N_EDGE_FEATURES)
    assert (edge_index[0] != edge_index[1]).all()  # no self-loops


# --- globals ---------------------------------------------------------------
def test_global_features_one_hot_and_length():
    u = G.global_features("From Counter", from_counter=True, trigger_time_s=0.0)
    assert u.shape == (G.N_BASE_GLOBAL_FEATURES,)  # base globals (shape appended in build_data)
    assert u[G.PLAY_PATTERNS.index("From Counter")] == 1.0
    assert u[-2] == 1.0  # from_counter flag
    assert u[-1] == 1.0  # time remaining at t=0 -> full half left


def test_build_data_appends_defensive_shape_globals():
    data = G.build_data(_basic_frame(), _row())
    assert data.u.shape == (1, G.N_GLOBAL_FEATURES)  # base + 4 defensive-shape features
    assert G.N_GLOBAL_FEATURES == G.N_BASE_GLOBAL_FEATURES + G.N_DEF_SHAPE_FEATURES


# --- velocity --------------------------------------------------------------
def test_nn_velocity_zero_without_prev_frame():
    pts = np.array([[90.0, 40.0], [100.0, 30.0]])
    assert np.allclose(G.nearest_neighbour_velocity(pts, np.empty((0, 2)), dt=1.0), 0.0)


def test_nn_velocity_matches_shifted_points():
    prev = np.array([[10.0, 10.0], [80.0, 70.0]])
    pts = prev + np.array([2.0, 0.0])  # everyone moved +2 in x over dt=1
    vel = G.nearest_neighbour_velocity(pts, prev, dt=1.0)
    assert np.allclose(vel, np.array([[2.0, 0.0], [2.0, 0.0]]))


# --- build_data ------------------------------------------------------------
def test_build_data_shapes_and_targets():
    frame = _basic_frame()
    row = _row(success=1.0, xt_progression=0.05, run_target_x=108.0, run_target_y=33.0)
    data = G.build_data(frame, row)
    n = 4
    assert data.x.shape == (n, G.N_NODE_FEATURES)
    assert data.edge_index.shape == (2, n * (n - 1))
    assert data.edge_attr.shape == (n * (n - 1), G.N_EDGE_FEATURES)
    assert data.u.shape == (1, G.N_GLOBAL_FEATURES)
    assert data.num_nodes == n
    # Targets present. The run target is a DISPLACEMENT from the run anchor (the
    # furthest-forward attacker excl. carrier = node 1 at [100, 30]), normalised.
    assert float(data.y_success) == 1.0
    assert abs(float(data.run_anchor[0, 0]) - 100.0 / 120) < 1e-6
    assert abs(float(data.run_anchor[0, 1]) - 30.0 / 80) < 1e-6
    assert abs(float(data.y_run[0, 0]) - (108.0 - 100.0) / 120) < 1e-6
    assert abs(float(data.y_run[0, 1]) - (33.0 - 30.0) / 80) < 1e-6


def test_build_data_without_labels_omits_targets():
    data = G.build_data(_basic_frame(), _row())
    assert not hasattr(data, "y_success")


def test_build_data_sets_aux_targets():
    row = _row(success=1.0, xt_progression=0.05, run_target_x=108.0, run_target_y=33.0)
    data = G.build_data(_basic_frame(), row)
    # Defensive-success (no def_stop on the row -> 0), Dynamic-xT (OBSO, finite), xPass mask.
    assert float(data.y_defsuccess) == 0.0
    assert data.y_dxt.shape == (1,) and np.isfinite(float(data.y_dxt))
    assert hasattr(data, "y_defline") and hasattr(data, "has_defline")
    assert hasattr(data, "y_xpass") and float(data.has_xpass) == 0.0  # no next pass on the row


def test_add_position_features_dims_and_onehots():
    row = _row(success=1.0, xt_progression=0.05, run_target_x=108.0, run_target_y=33.0)
    g = G.build_data(_basic_frame(), row)
    h = G.add_position_features(g, actor_role="MID")
    # Node features grow by the role one-hot; globals grow by the carrier role one-hot.
    assert h.x.shape[1] == g.x.shape[1] + G.N_NODE_ROLE_FEATURES
    assert h.u.shape[1] == g.u.shape[1] + G.N_ACTOR_ROLE_FEATURES
    # Per-node role is a one-hot over the 3 appended columns.
    role_block = h.x[:, -G.N_NODE_ROLE_FEATURES :]
    assert role_block.sum().item() == g.x.shape[0]  # exactly one role per node
    # Carrier role one-hot = MID (index 2 of GK/DEF/MID/FWD).
    actor_block = h.u[0, -G.N_ACTOR_ROLE_FEATURES :]
    assert actor_block.tolist() == [0.0, 0.0, 1.0, 0.0]
    assert float(h.y_success) == float(g.y_success)  # labels preserved


def test_reflect_graph_mirrors_laterally_and_preserves_structure():
    row = _row(success=1.0, xt_progression=0.05, run_target_x=108.0, run_target_y=33.0)
    g = G.build_data(_basic_frame(), row)
    h = G.reflect_graph(g)
    # Same topology + scalar labels; y mirrored about the midline (0.5 in normalised units).
    assert h.x.shape == g.x.shape
    assert h.edge_index.shape == g.edge_index.shape
    assert float(h.y_success) == float(g.y_success)
    assert np.allclose(h.x[:, 1].numpy(), 1.0 - g.x[:, 1].numpy())  # y flips
    assert np.allclose(h.x[:, 0].numpy(), g.x[:, 0].numpy())  # x unchanged
    assert np.allclose(h.x[:, 3].numpy(), -g.x[:, 3].numpy())  # vy sign flips
    assert np.allclose(h.y_run[:, 1].numpy(), -g.y_run[:, 1].numpy())  # run dy flips
    assert np.allclose(h.run_anchor[:, 1].numpy(), 1.0 - g.run_anchor[:, 1].numpy())
    # A double reflection is the identity.
    assert np.allclose(G.reflect_graph(h).x.numpy(), g.x.numpy(), atol=1e-6)


def test_build_data_receiver_label_nearest_teammate():
    frame = _basic_frame()
    row = _row(
        success=1.0,
        xt_progression=0.05,
        run_target_x=108.0,
        run_target_y=33.0,
        next_pass_end_x=101.0,
        next_pass_end_y=31.0,
    )
    data = G.build_data(frame, row)
    # Receiver = teammate (excl. actor) nearest [101, 31] -> node 1 at [100, 30].
    assert float(data.has_receiver) == 1.0
    assert data.y_receiver.tolist() == [0.0, 1.0, 0.0, 0.0]
    assert float(data.has_run_alt) == 1.0
    # y_run_alt is a displacement from the trigger ball location (90, 40).
    assert abs(float(data.y_run_alt[0, 0]) - (101.0 - 90.0) / 120) < 1e-6
    assert abs(float(data.y_run_alt[0, 1]) - (31.0 - 40.0) / 80) < 1e-6
    # Presser = defender (incl. keeper) nearest [101, 31]: node 3 keeper [118,40] (19.2 m)
    # edges node 2 defender [105,50] (19.4 m).
    assert float(data.has_presser) == 1.0
    assert data.y_presser.tolist() == [0.0, 0.0, 0.0, 1.0]


def test_build_data_receiver_mask_zero_without_next_pass():
    data = G.build_data(
        _basic_frame(), _row(success=0.0, xt_progression=0.0, run_target_x=90.0, run_target_y=40.0)
    )
    assert float(data.has_receiver) == 0.0
    assert float(data.y_receiver.sum()) == 0.0
    assert float(data.has_run_alt) == 0.0


def test_build_data_velocity_from_prev_frame():
    frame = _basic_frame()
    prev = _frame(
        [
            {"location": [88.0, 40.0], "actor": True},
            {"location": [98.0, 30.0]},
            {"location": [103.0, 50.0], "teammate": False},
            {"location": [116.0, 40.0], "teammate": False, "keeper": True},
        ]
    )
    data = G.build_data(frame, _row(), prev_frame=prev, dt=1.0)
    # Every player advanced +2 in x -> positive vx column (index 2), non-zero.
    assert (data.x[:, 2] > 0).any()
