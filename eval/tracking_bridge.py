"""Tracking bridge: free tracking data (kloppy) -> state-of-play freeze-frames -> GAT.

The dense-data path (Section D). Free tracking providers give per-frame player+ball
coordinates for *all 22 players* — denser than StatsBomb 360 and the route to club-scale data
without the CV pipeline (the provider already did the computer vision). kloppy normalises
Metrica/SkillCorner into one schema; we convert it to the **same 105x68 positions table the
CV bridge consumes** (``frame, track_id, role, team, pitch_x, pitch_y, is_actor``), so both
the video path (``eval/cv_bridge``) and the tracking path flow through the identical
freeze-frame -> model code. Run ``python -m eval.tracking_bridge`` (needs network: kloppy
fetches the open data from GitHub once).

Metrica sample data: 3 full-tracking matches (anonymised). SkillCorner open data: broadcast
tracking (its repo's current match ids; pass ``--provider skillcorner --match-id <id>``).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from data.possessions import PITCH_LENGTH, PITCH_WIDTH

logger = logging.getLogger(__name__)
OUT_DIR = Path("data/processed")
SRC_LEN, SRC_WID = 105.0, 68.0  # cv_bridge's source convention (kloppy pitch is 105x68)
ACTOR_MAX_DIST_M = 3.0


def _load(provider: str, match_id: int, limit: int):
    """Load a kloppy open tracking dataset for the given provider."""
    if provider == "metrica":
        from kloppy import metrica

        return metrica.load_open_data(match_id=match_id, limit=limit)
    if provider == "skillcorner":
        from kloppy import skillcorner

        return skillcorner.load_open_data(match_id=match_id, limit=limit)
    raise ValueError(f"unknown provider: {provider}")


def tracking_to_positions(dataset, sample_every: int = 25) -> pd.DataFrame:
    """Convert a kloppy tracking dataset to the cv-football positions schema (105x68)."""
    home = dataset.metadata.teams[0]
    rows: list[dict] = []
    for i, fr in enumerate(dataset.records):
        if i % sample_every or not fr.players_coordinates:
            continue
        ball = fr.ball_coordinates
        bx = float(ball.x) * SRC_LEN if ball is not None else None
        by = float(ball.y) * SRC_WID if ball is not None else None
        frame_rows = []
        for player, pt in fr.players_coordinates.items():
            if pt is None:
                continue
            frame_rows.append(
                {
                    "frame": i,
                    "track_id": int(abs(hash(player.player_id)) % 100000),
                    "role": "player",
                    "team": 0 if player.team == home else 1,
                    "pitch_x": float(pt.x) * SRC_LEN,
                    "pitch_y": float(pt.y) * SRC_WID,
                    "is_actor": False,
                    "conf": 1.0,
                }
            )
        if not frame_rows:
            continue
        # Actor = the player nearest the ball this frame (within ACTOR_MAX_DIST_M).
        if bx is not None:
            d = [np.hypot(r["pitch_x"] - bx, r["pitch_y"] - by) for r in frame_rows]
            j = int(np.argmin(d))
            if d[j] <= ACTOR_MAX_DIST_M:
                frame_rows[j]["is_actor"] = True
            frame_rows.append(
                {"frame": i, "track_id": -1, "role": "ball", "team": -1,
                 "pitch_x": bx, "pitch_y": by, "is_actor": False, "conf": 1.0}
            )
        rows.extend(frame_rows)
    return pd.DataFrame(rows)


def team_fingerprints(pos: pd.DataFrame, model) -> pd.DataFrame:
    """Per-team attacking + defensive fingerprints from a tracking match.

    The team-metrics schema (model-derived subset); per frame the attacking side is the
    ball-carrier's team, the other side defends.
    """
    from types import SimpleNamespace

    import torch

    from data.graphs import build_data
    from eval.cv_bridge import frame_to_statsbomb

    recs: list[dict] = []
    model.eval()
    with torch.no_grad():
        for _fr, grp in pos.groupby("frame"):
            frame = frame_to_statsbomb(grp)
            if frame is None:
                continue
            atk = int(frame.attrs["atk_team"])
            here = [int(t) for t in grp.loc[grp["role"] != "ball", "team"].unique()]
            deff = next((t for t in here if t != atk), None)
            ball = grp[grp["role"] == "ball"]
            bx = float(ball.iloc[0]["pitch_x"] * PITCH_LENGTH / SRC_LEN) if len(ball) else 60.0
            by = float(ball.iloc[0]["pitch_y"] * PITCH_WIDTH / SRC_WID) if len(ball) else 40.0
            row = SimpleNamespace(
                trigger_x=bx, trigger_y=by, play_pattern="Regular Play",
                from_counter=False, trigger_time_s=600.0,
            )
            out = model(build_data(frame, row))
            succ = float(torch.sigmoid(out["success"]))
            rp = out["receiver_probs"].clamp_min(1e-9)
            xp = out["xpass_dense"]
            loc = np.array(frame["location"].tolist())
            dmask = (~frame["teammate"].to_numpy()) & (~frame["keeper"].to_numpy())
            line = float(np.sort(loc[dmask, 0])[-2]) if int(dmask.sum()) >= 2 else np.nan
            recs.append({"team": atk, "side": "atk", "success": succ,
                         "dxt": float(out["dxt"]), "ent": float(-(rp.log() * rp).sum())})
            if deff is not None:
                recs.append({"team": deff, "side": "def", "success": succ,
                             "presser": float(out["presser_probs"].max()),
                             "xpass": float(xp.sum() / (xp > 0).sum().clamp_min(1)), "line": line})
    d = pd.DataFrame(recs)
    a, f = d[d.side == "atk"].groupby("team"), d[d.side == "def"].groupby("team")
    att = pd.DataFrame({"n_attack": a.size(), "attack_success": a["success"].mean(),
                        "attack_dxt": a["dxt"].mean(), "option_richness": a["ent"].mean()})
    deff_t = pd.DataFrame({
        "n_defend": f.size(), "solidity": 1 - f["success"].mean(),
        "press_decisiveness": f["presser"].mean(), "lane_suppression": 1 - f["xpass"].mean(),
        "line_height": f["line"].mean(),
    })
    return att.join(deff_t, how="outer")


TEAM_METRICS_PATH = Path("results/tracking_team_metrics.csv")


def main() -> None:
    """Ingest one or more tracking matches -> positions -> GAT (clip readout or team metrics)."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="metrica", choices=["metrica", "skillcorner"])
    ap.add_argument("--match-ids", type=int, nargs="+", default=[1], help="match id(s) to ingest")
    ap.add_argument("--limit", type=int, default=6000, help="frames to fetch per match (speed)")
    ap.add_argument("--sample-every", type=int, default=25, help="downsample (25 fps -> 1/s)")
    ap.add_argument("--team-metrics", action="store_true", help="write per-team fingerprints")
    args = ap.parse_args()

    model = None
    if args.team_metrics:
        import torch

        from models.gnn import GAT

        model = GAT()
        model.load_state_dict(torch.load("results/checkpoints/gnn.pt", weights_only=True))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fingerprints: list[pd.DataFrame] = []
    for mid in args.match_ids:
        logger.info("loading %s match %s (limit=%d)...", args.provider, mid, args.limit)
        try:
            ds = _load(args.provider, mid, args.limit)
        except Exception as exc:  # noqa: BLE001 - skip a match that the open repo lacks
            logger.warning("skip %s match %s: %s", args.provider, mid, str(exc)[:160])
            continue
        pos = tracking_to_positions(ds, args.sample_every)
        names = [t.name for t in ds.metadata.teams]  # real club names (SkillCorner) or Home/Away
        out = OUT_DIR / f"{args.provider}_{mid}_positions.parquet"
        pos.to_parquet(out, index=False)
        nf = int(pos["frame"].nunique())
        ppf = int((pos["role"] == "player").sum()) / max(nf, 1)
        print(f"{args.provider} match {mid}: {nf} frames, {ppf:.1f} players/frame -> {out}")

        if args.team_metrics:
            fp = team_fingerprints(pos, model).reset_index(names="team_id")
            labels = [
                names[int(s)] if names[int(s)] not in ("Home", "Away")
                else f"{args.provider}{mid}_{names[int(s)]}"
                for s in fp["team_id"]
            ]
            fp.insert(0, "team", labels)
            fingerprints.append(fp.drop(columns="team_id"))
        else:
            from eval.cv_bridge import run_clip

            df = run_clip(str(out))
            if not df.empty:
                print(f"  GAT on {len(df)} frames | P(success) {df['success'].mean():.3f} "
                      f"| DxT {df['dynamic_xt'].mean():.4f} | P(def) {df['p_defstop'].mean():.3f}")

    if args.team_metrics and fingerprints:
        table = pd.concat(fingerprints, ignore_index=True).round(4)
        out_csv = TEAM_METRICS_PATH.with_name(f"tracking_team_metrics_{args.provider}.csv")
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out_csv, index=False)
        print("\nPer-team tracking fingerprints:")
        print(table.to_string(index=False))
        print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
