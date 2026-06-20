"""Pre-compute the dashboard's prediction cache (CLAUDE.md §10 Week 6).

Runs both trained checkpoints over every **held-out test** possession and writes a
self-contained ``app/assets/predictions.parquet``: model outputs, targets, match
metadata, and the freeze-frame geometry (positions, velocities, role flags) needed
to draw each possession. The Streamlit app then needs **no torch at runtime** —
which also sidesteps the Windows statsbombpy/torch_geometric DLL-order crash.

Run ``python -m app.precompute`` after (re)training.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from data.graphs import PITCH_LENGTH, PITCH_WIDTH, frame_points, load_frames_local, load_graph_cache
from data.load import load_events_local
from data.possessions import add_event_time
from data.sequences import SEQUENCE_LEN, buildup_event_ids
from eval.counterfactual import success_surface
from models.gnn import GAT
from models.transformer import PlayerTransformer
from train.utils import build_splits

logger = logging.getLogger(__name__)
ASSETS_DIR = Path("app/assets")
PREDICTIONS_PATH = ASSETS_DIR / "predictions.parquet"
SEQUENCES_PATH = ASSETS_DIR / "sequences.parquet"
COUNTERFACTUAL_PATH = ASSETS_DIR / "counterfactual.parquet"
SCALE = np.array([PITCH_LENGTH, PITCH_WIDTH])


@torch.no_grad()
def model_outputs(model: torch.nn.Module, graphs: list, prefix: str) -> list[dict]:
    """Per-graph prediction dicts (success prob, xT, run target, receiver probs)."""
    model.eval()
    rows: list[dict] = []
    for batch in DataLoader(graphs, batch_size=128):
        out = model(batch)
        p = torch.sigmoid(out["success"]).numpy()
        xt = out["xt"].numpy()
        # The run head emits a DISPLACEMENT from the run anchor (graphs.py); add the
        # anchor back to recover the absolute predicted position the dashboard draws.
        anchor = batch.run_anchor.numpy() * SCALE
        run = anchor + out["run"].numpy() * SCALE
        recv = out.get("receiver_probs")
        press = out.get("presser_probs")
        node_counts = np.bincount(batch.batch.numpy())
        for i in range(len(p)):
            row = {
                f"{prefix}_p_success": float(p[i]),
                f"{prefix}_xt": float(xt[i]),
                f"{prefix}_run_x": float(run[i, 0]),
                f"{prefix}_run_y": float(run[i, 1]),
            }
            if "defsuccess" in out:  # GAT-only aux heads
                row[f"{prefix}_p_defstop"] = float(torch.sigmoid(out["defsuccess"])[i])
                row[f"{prefix}_dxt"] = float(out["dxt"][i])
            if recv is not None:
                row[f"{prefix}_receiver_probs"] = recv[i, : node_counts[i]].numpy().tolist()
            if press is not None:
                row[f"{prefix}_presser_probs"] = press[i, : node_counts[i]].numpy().tolist()
            rows.append(row)
    return rows


def geometry_rows(graphs: list) -> list[dict]:
    """Freeze-frame geometry + targets, one dict per graph (drawable without torch)."""
    rows = []
    for g in graphs:
        x = g.x.numpy()
        rows.append(
            {
                "match_id": int(g.match_id),
                "possession": int(g.possession),
                "px": (x[:, 0] * PITCH_LENGTH).tolist(),
                "py": (x[:, 1] * PITCH_WIDTH).tolist(),
                "vx": (x[:, 2] * PITCH_LENGTH).tolist(),
                "vy": (x[:, 3] * PITCH_WIDTH).tolist(),
                "teammate": (x[:, 4] > 0.5).tolist(),
                "actor": (x[:, 5] > 0.5).tolist(),
                "y_success": float(g.y_success),
                "y_xt": float(g.y_xt),
                # y_run is a displacement; add the anchor to recover the true position.
                "run_true_x": float((g.run_anchor[0, 0] + g.y_run[0, 0]) * PITCH_LENGTH),
                "run_true_y": float((g.run_anchor[0, 1] + g.y_run[0, 1]) * PITCH_WIDTH),
            }
        )
    return rows


def after_frame_rows(poss: pd.DataFrame, keep: set[int]) -> pd.DataFrame:
    """The real freeze frame ~1.5 s after each trigger (the run-target event).

    StatsBomb 360 has no broadcast stills; this is the genuine recorded player
    layout at the next event, so the dashboard can show a real before/after rather
    than only a single predicted point.
    """
    rows: list[dict] = []
    for mid, grp in poss[poss["match_id"].isin(keep)].groupby("match_id"):
        try:
            by_event = dict(tuple(load_frames_local(int(mid)).groupby("id")))
        except Exception as exc:  # noqa: BLE001 - log + skip a match
            logger.warning("after-frame: skip match %s: %s", mid, exc)
            by_event = {}
        for r in grp.itertuples(index=False):
            ev = getattr(r, "run_target_event_id", None)
            fr = by_event.get(ev) if isinstance(ev, str) else None
            rec = {
                "match_id": int(mid),
                "possession": int(r.possession),
                "has_after": False,
                "after_px": [],
                "after_py": [],
                "after_teammate": [],
                "after_actor": [],
            }
            if fr is not None:
                pts, tm, ac, _ = frame_points(fr)
                if len(pts):
                    rec.update(
                        has_after=True,
                        after_px=pts[:, 0].tolist(),
                        after_py=pts[:, 1].tolist(),
                        after_teammate=tm.tolist(),
                        after_actor=ac.tolist(),
                    )
            rows.append(rec)
    return pd.DataFrame(rows)


def sequence_rows(poss: pd.DataFrame, keep: set[int], k: int = SEQUENCE_LEN) -> pd.DataFrame:
    """Build-up freeze-frame geometry + a numerical-superiority read per kept possession.

    Powers the dashboard "state of play" panel: the last ``k`` on-ball frames before the
    trigger, each as drawable player coordinates, plus **attackers minus defenders ahead
    of the ball** per frame (the 3v2 read the supervisor cares about). Drawn without torch.
    Nested lists are JSON-encoded so the parquet schema stays flat and portable.
    """
    rows: list[dict] = []
    sub = poss[poss["match_id"].isin(keep)]
    for mid, grp in sub.groupby("match_id"):
        try:
            events = add_event_time(load_events_local(int(mid)))
            by_event = dict(tuple(load_frames_local(int(mid)).groupby("id")))
        except Exception as exc:  # noqa: BLE001 - log + skip a match
            logger.warning("sequence-geom: skip match %s: %s", mid, exc)
            continue
        for r in grp.itertuples(index=False):
            frames: list[dict] = []
            superiority: list[int] = []
            for eid in buildup_event_ids(events, r, k):
                fr = by_event.get(eid)
                if fr is None or len(fr) == 0:
                    continue
                pts, tm, ac, _ = frame_points(fr)
                if not len(pts):
                    continue
                ball_x = float(pts[ac][0, 0]) if ac.any() else float(r.trigger_x)
                ahead = pts[:, 0] > ball_x
                sup = int((ahead & tm & ~ac).sum()) - int((ahead & ~tm).sum())
                frames.append(
                    {
                        "px": pts[:, 0].tolist(),
                        "py": pts[:, 1].tolist(),
                        "teammate": tm.tolist(),
                        "actor": ac.tolist(),
                    }
                )
                superiority.append(sup)
            if not frames:
                continue
            rows.append(
                {
                    "match_id": int(mid),
                    "possession": int(r.possession),
                    "frames_json": json.dumps(frames),
                    "superiority_json": json.dumps(superiority),
                    "seq_len": len(frames),
                }
            )
    return pd.DataFrame(rows)


def counterfactual_rows(model: torch.nn.Module, graphs: list) -> pd.DataFrame:
    """Per-possession success-probability surface as the key attacker is swept (instinct).

    Stores the surface + the success-maximising move so the dashboard can show *where the
    most advanced attacker should go* using the strong success head (`eval.counterfactual`).
    Surface/axes are JSON-encoded to keep the parquet schema flat.
    """
    rows: list[dict] = []
    for g in graphs:
        cf = success_surface(model, g)
        if cf is None:
            continue
        rows.append(
            {
                "match_id": int(g.match_id),
                "possession": int(g.possession),
                "surface_json": json.dumps(np.round(cf["surface"], 4).tolist()),
                "dxs_json": json.dumps(cf["dxs"].tolist()),
                "dys_json": json.dumps(cf["dys"].tolist()),
                "base": round(cf["base"], 4),
                "best": round(cf["best"], 4),
                "best_dx": round(cf["best_dx"], 2),
                "best_dy": round(cf["best_dy"], 2),
                "key_index": int(cf["key_index"]),
            }
        )
    return pd.DataFrame(rows)


def match_metadata() -> pd.DataFrame:
    """Match identity columns from the cached match list."""
    records = json.loads(Path("data/processed/match_ids.json").read_text(encoding="utf-8"))
    cols = ["match_id", "competition_name", "season_name", "match_date", "home_team", "away_team"]
    return pd.DataFrame(records)[cols]


def _case_study_match_ids() -> set[int]:
    """Match ids referenced by the curated case studies.

    Kept in the cache even when they fall in the train/val splits — flagged via
    the ``in_test`` column.
    """
    path = Path("results/case_studies/case_studies.json")
    if not path.exists():
        return set()
    return {int(c["match_id"]) for c in json.loads(path.read_text(encoding="utf-8"))}


def build_cache(out_path: Path = PREDICTIONS_PATH) -> pd.DataFrame:
    """Assemble and write the dashboard cache: held-out test set + case-study matches."""
    poss = pd.read_parquet("data/processed/possessions.parquet")
    graphs = load_graph_cache()
    test_ids = set(build_splits(poss)["test"])
    keep = test_ids | _case_study_match_ids()
    test = [g for g in graphs if int(g.match_id) in keep]
    logger.info("precompute: %d graphs (%d matches)", len(test), len(keep))

    gat = GAT()
    gat.load_state_dict(torch.load("results/checkpoints/gnn.pt", weights_only=True))
    tfm = PlayerTransformer()
    tfm.load_state_dict(torch.load("results/checkpoints/transformer.pt", weights_only=True))

    geo = pd.DataFrame(geometry_rows(test))
    gat_rows = pd.DataFrame(model_outputs(gat, test, "gat"))
    tfm_rows = pd.DataFrame(model_outputs(tfm, test, "tfm"))
    table = pd.concat([geo, gat_rows, tfm_rows], axis=1)
    # Explainability: per-player attention importance (GATv2 weights). Same order as `test`,
    # assigned before the left-merges (which preserve order).
    table["gat_node_importance"] = [
        gat.node_attention(g).cpu().numpy().round(4).tolist() for g in test
    ]

    meta_poss = poss[
        ["match_id", "possession", "play_pattern", "from_counter", "trigger_time_s", "period"]
    ]
    table = table.merge(meta_poss, on=["match_id", "possession"], how="left")
    table = table.merge(match_metadata(), on="match_id", how="left")
    table = table.merge(after_frame_rows(poss, keep), on=["match_id", "possession"], how="left")
    table["in_test"] = table["match_id"].isin(test_ids)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out_path, index=False)
    logger.info("Wrote %d rows -> %s", len(table), out_path)

    seqs = sequence_rows(poss, keep)
    seqs.to_parquet(SEQUENCES_PATH, index=False)
    logger.info("Wrote %d sequence rows -> %s", len(seqs), SEQUENCES_PATH)

    cf = counterfactual_rows(gat, test)
    cf.to_parquet(COUNTERFACTUAL_PATH, index=False)
    logger.info("Wrote %d counterfactual rows -> %s", len(cf), COUNTERFACTUAL_PATH)
    return table


def main() -> None:
    """Entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    table = build_cache()
    print(f"\n{len(table)} possessions cached -> {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
