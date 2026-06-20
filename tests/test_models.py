"""Unit tests for the model + training layer (offline, synthetic graphs)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader

from data import graphs as G


def make_graph(success: float = 1.0, seed: int = 0, with_receiver: bool = True):
    """Build a synthetic 12-player graph with labels (+ next-pass receiver)."""
    rng = np.random.default_rng(seed)
    rows = [
        {
            "location": [float(rng.uniform(60, 118)), float(rng.uniform(5, 75))],
            "teammate": i % 2 == 0,
            "actor": i == 0,
            "keeper": i == 11,
        }
        for i in range(12)
    ]
    row = SimpleNamespace(
        trigger_x=90.0,
        trigger_y=40.0,
        play_pattern="Regular Play",
        from_counter=False,
        trigger_time_s=600.0,
        match_id=1,
        possession=1,
        success=success,
        xt_progression=0.05,
        run_target_x=100.0,
        run_target_y=40.0,
        next_pass_end_x=100.0 if with_receiver else float("nan"),
        next_pass_end_y=40.0 if with_receiver else float("nan"),
    )
    return G.build_data(pd.DataFrame(rows), row)


def _batch(n: int = 2):
    graphs = [make_graph(float(i % 2), seed=i) for i in range(n)]
    return next(iter(DataLoader(graphs, batch_size=n)))


# --- GAT forward -----------------------------------------------------------
def test_gat_forward_shapes():
    from models.gnn import GAT

    out = GAT()(_batch(2))
    assert out["success"].shape == (2,)
    assert out["xt"].shape == (2,)
    assert out["run"].shape == (2, 2)


# --- multi-task loss -------------------------------------------------------
def test_multitask_loss_and_rebalance():
    from models.gnn import GAT
    from train.losses import multitask_loss, rebalance

    batch = _batch(2)
    total, comp = multitask_loss(GAT()(batch), batch)
    assert total.requires_grad
    assert {"success", "xt", "run"} <= set(comp)
    w = rebalance(comp)
    assert len(w) == 3 and all(x > 0 for x in w)


# --- engineered features + baselines --------------------------------------
def test_engineered_features_length():
    from features.engineered import N_ENGINEERED, engineered_features

    assert engineered_features(make_graph()).shape == (N_ENGINEERED,)


def test_baselines_fit_predict():
    from models.baselines import (
        EngineeredBaseline,
        PriorBaseline,
        features_matrix,
        run_baseline_nomove,
    )

    graphs = [make_graph(float(i % 2), seed=i) for i in range(6)]
    x = features_matrix(graphs)
    y_s = np.array([float(i % 2) for i in range(6)])
    y_x = np.random.default_rng(0).random(6)
    assert PriorBaseline().fit(y_s, y_x).predict_success(3).shape == (3,)
    assert EngineeredBaseline().fit(x, y_s, y_x).predict_success(x).shape == (6,)
    assert run_baseline_nomove(graphs).shape == (6, 2)


def test_gat_ablation_flags_forward():
    from models.gnn import GAT

    batch = _batch(2)
    assert GAT(use_edge_attr=False)(batch)["success"].shape == (2,)
    assert GAT(use_globals=False)(batch)["success"].shape == (2,)
    assert GAT(hidden=128)(batch)["success"].shape == (2,)


def test_zero_velocity_clones():
    import torch

    from eval.ablations import zero_velocity

    graphs = [make_graph(seed=1)]
    out = zero_velocity(graphs)
    assert torch.all(out[0].x[:, 2:4] == 0)
    assert out[0] is not graphs[0]  # original untouched


# --- receiver head + two-stage success --------------------------------------
def test_gat_receiver_outputs():
    import torch

    from models.gnn import GAT

    batch = _batch(2)
    out = GAT(with_receiver=True)(batch)
    assert out["success_two"].shape == (2,)
    assert torch.all((out["success_two"] > 0) & (out["success_two"] < 1))
    # Receiver probabilities sum to ~1 per graph (over feasible teammates).
    sums = out["receiver_probs"].sum(dim=1)
    assert torch.allclose(sums, torch.ones(2), atol=1e-5)
    # Infeasible nodes (defenders, the actor) carry zero probability.
    probs_flat = out["receiver_probs"][out["node_mask"]]
    feasible = (batch.x[:, 4] > 0.5) & (batch.x[:, 5] < 0.5)
    assert float(probs_flat[~feasible].sum()) < 1e-6


def test_multitask_loss_includes_receiver_terms():
    from models.gnn import GAT
    from train.losses import multitask_loss

    batch = _batch(2)
    total, comp = multitask_loss(GAT(with_receiver=True)(batch), batch, w_receiver=1.0)
    assert total.requires_grad
    assert {"receiver", "success_two"} <= set(comp)


# --- defense (presser) head -------------------------------------------------
def test_gat_defense_head_outputs_and_loss():
    import torch

    from models.gnn import GAT
    from train.losses import multitask_loss

    batch = _batch(2)
    out = GAT(with_defense=True)(batch)
    # Presser probabilities sum to ~1 per graph, over defenders only.
    sums = out["presser_probs"].sum(dim=1)
    assert torch.allclose(sums, torch.ones(2), atol=1e-5)
    probs_flat = out["presser_probs"][out["presser_mask"]]
    defenders = batch.x[:, 4] < 0.5
    assert float(probs_flat[~defenders].sum()) < 1e-6  # attackers carry no mass
    total, comp = multitask_loss(out, batch, w_defense=1.0)
    assert "presser" in comp and total.requires_grad


def test_gat_aux_heads_and_attention():

    from models.gnn import GAT
    from train.losses import multitask_loss

    batch = _batch(3)
    model = GAT()
    out = model(batch)
    for k in ("defsuccess", "defline", "dxt"):
        assert out[k].shape == (3,), (k, out[k].shape)
    assert "xpass_dense" in out and out["xpass_dense"].shape[0] == 3
    _, comp = multitask_loss(out, batch)
    assert {"defsuccess", "dxt"} <= set(comp)  # aux loss terms present
    # Attention importance is a normalised per-node vector for a single graph.
    g = make_graph(seed=3)
    imp = model.node_attention(g)
    assert imp.shape[0] == g.x.shape[0]
    assert abs(float(imp.sum()) - 1.0) < 1e-4


def test_counterfactual_success_surface_shape():
    from eval.counterfactual import GRID_DX, GRID_DY, success_surface
    from models.gnn import GAT

    cf = success_surface(GAT(), make_graph(seed=2))
    assert cf is not None
    assert cf["surface"].shape == (len(GRID_DY), len(GRID_DX))
    assert 0.0 <= cf["base"] <= 1.0 and 0.0 <= cf["best"] <= 1.0
    assert cf["best"] >= cf["base"] - 1e-6  # best is the argmax over the grid


def test_defensive_shape_descriptors():
    from data.graphs import N_DEF_SHAPE_FEATURES, defensive_shape_globals

    pts = np.array([[100.0, 30.0], [106.0, 50.0], [90.0, 40.0], [95.0, 20.0]])
    teammate = np.array([False, False, True, True])
    keeper = np.array([False, False, False, False])
    s = defensive_shape_globals(pts, teammate, keeper, ball=np.array([90.0, 40.0]))
    assert s.shape == (N_DEF_SHAPE_FEATURES,)
    assert np.all(np.isfinite(s))
    assert 0.0 <= s[0] <= 1.0  # line height is a normalised x in [0, 1]


def test_cv_bridge_frame_conversion():
    from eval.cv_bridge import frame_to_statsbomb

    rng = np.random.default_rng(0)
    rows = [
        {
            "role": "player",
            "team": i % 2,
            "pitch_x": float(rng.uniform(0, 105)),
            "pitch_y": float(rng.uniform(0, 68)),
            "is_actor": i == 0,
        }
        for i in range(12)
    ]
    rows.append({"role": "ball", "team": -1, "pitch_x": 52.0, "pitch_y": 34.0, "is_actor": False})
    fr = frame_to_statsbomb(pd.DataFrame(rows))
    assert fr is not None and len(fr) == 12  # ball dropped, 12 players kept
    assert set(["location", "teammate", "actor", "keeper"]) <= set(fr.columns)
    assert int(fr["actor"].sum()) == 1  # exactly one ball-carrier
    assert fr["keeper"].sum() >= 1  # at least one keeper tagged
    # Too-sparse frames are rejected (mirrors the >=10-visible filter).
    assert frame_to_statsbomb(pd.DataFrame(rows[:6])) is None


# --- transformer -------------------------------------------------------------
def test_transformer_forward_shapes():
    from models.transformer import PlayerTransformer

    out = PlayerTransformer()(_batch(2))
    assert out["success"].shape == (2,)
    assert out["xt"].shape == (2,)
    assert out["run"].shape == (2, 2)


# --- temporal models ---------------------------------------------------------
def test_temporal_models_forward_shapes():
    import torch
    from torch_geometric.data import Batch

    from models.temporal import TemporalGAT, TemporalTransformer

    # Two build-up sequences of 3 and 2 frames; flattened to one Batch + lengths.
    seqs = [[make_graph(seed=i) for i in range(3)], [make_graph(seed=i + 10) for i in range(2)]]
    frames = Batch.from_data_list([f for s in seqs for f in s])
    lengths = torch.tensor([3, 2])
    for model_cls in (TemporalGAT, TemporalTransformer):
        out = model_cls()(frames, lengths)
        assert out["success"].shape == (2,)
        assert out["xt"].shape == (2,)


# --- geometric baselines (B2 / B3) -------------------------------------------
def test_pitch_control_in_unit_interval_and_monotone():
    from features.pitch_control import box_control, box_grid, control_surface

    g = make_graph()
    c = control_surface(g, box_grid())
    assert ((c >= 0) & (c <= 1)).all()
    assert 0.0 <= box_control(g) <= 1.0


def test_obso_scalar_finite_nonnegative():
    from features.obso import obso

    val = obso(make_graph())
    assert np.isfinite(val) and val >= 0.0


def test_geometric_feature_matrices():
    from models.baselines import obso_matrix, pitch_control_matrix

    graphs = [make_graph(seed=i) for i in range(3)]
    assert pitch_control_matrix(graphs).shape == (3, 1)
    assert obso_matrix(graphs).shape == (3, 1)


# --- splits ----------------------------------------------------------------
def test_build_splits_disjoint_and_complete():
    from train.utils import build_splits

    poss = pd.DataFrame({"match_id": list(range(20)), "competition_name": ["A"] * 10 + ["B"] * 10})
    s = build_splits(poss, out_path=None)
    all_ids = s["train"] + s["val"] + s["test"]
    assert len(set(all_ids)) == 20
    assert not (set(s["train"]) & set(s["test"]))
    assert not (set(s["train"]) & set(s["val"]))
