"""CV bridge: cv-football positions parquet -> state-of-play freeze-frames -> GAT predictions.

Closes the loop from the Phase-1 CV pipeline (``extract_positions.py``: video -> YOLO +
ByteTrack + homography -> positions parquet) to the trained relational model, so the model
runs on *any* match video, not just StatsBomb 360. Per the density finding (broadcast TV only
shows ~5-6 players), it keeps **only frames with >= MIN_PLAYERS** visible (mirroring 360's
>=10-visible filter) and treats those as freeze-frames.

Per qualifying frame: rescale 105x68 -> 120x80, set the **attacking team** = the ball-carrier's
team (else the more-advanced team), **orient** them left->right, derive **actor** (the carrier)
and **keeper** (each team's deepest player), then run the GAT. Aggregates a clip-level readout.

Run ``python -m eval.cv_bridge --positions <path/to/positions.parquet>``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from data.graphs import build_data
from data.possessions import PITCH_LENGTH, PITCH_WIDTH
from models.gnn import GAT

logger = logging.getLogger(__name__)
CKPT = Path("results/checkpoints/gnn.pt")
SRC_LEN, SRC_WID = 105.0, 68.0  # cv-football pitch convention
MIN_PLAYERS = 10  # mirror StatsBomb 360's >=10-visible filter (wide-shot frames only)


def _orient(px: np.ndarray, py: np.ndarray, attacking: np.ndarray):
    """Flip so the attacking team plays left->right (attacked goal at x=120)."""
    if attacking.any() and float(px[attacking].mean()) < PITCH_LENGTH / 2:
        return PITCH_LENGTH - px, PITCH_WIDTH - py
    return px, py


def frame_to_statsbomb(grp: pd.DataFrame) -> pd.DataFrame | None:
    """Convert one frame's CV rows to a StatsBomb-style freeze frame, or None if too sparse."""
    players = grp[grp["role"] != "ball"].dropna(subset=["pitch_x", "pitch_y"])
    if len(players) < MIN_PLAYERS:
        return None
    px = players["pitch_x"].to_numpy() * PITCH_LENGTH / SRC_LEN
    py = players["pitch_y"].to_numpy() * PITCH_WIDTH / SRC_WID
    team = players["team"].to_numpy()
    actor = (
        players["is_actor"].to_numpy().astype(bool)
        if "is_actor" in players.columns
        else np.zeros(len(players), bool)
    )

    # Attacking team = the ball-carrier's team if tagged, else the more-advanced team.
    if actor.any():
        atk_team = int(team[actor][0])
    else:
        means = {int(t): float(px[team == t].mean()) for t in np.unique(team)}
        atk_team = max(means, key=means.get)
    teammate = team == atk_team
    px, py = _orient(px, py, teammate)

    # Keeper heuristic: each team's most extreme player along the attack axis.
    keeper = np.zeros(len(players), bool)
    if teammate.any():
        keeper[np.where(teammate)[0][int(np.argmin(px[teammate]))]] = True  # attacking GK (deep)
    if (~teammate).any():
        keeper[np.where(~teammate)[0][int(np.argmax(px[~teammate]))]] = True  # defending GK
    out = pd.DataFrame(
        {
            "location": [[float(x), float(y)] for x, y in zip(px, py, strict=True)],
            "teammate": teammate,
            "actor": actor,
            "keeper": keeper,
        }
    )
    out.attrs["atk_team"] = int(atk_team)  # which side is attacking (for per-team aggregation)
    return out


@torch.no_grad()
def run_clip(positions_path: str, ckpt: Path = CKPT) -> pd.DataFrame:
    """Run the GAT over every >= MIN_PLAYERS frame of a CV positions parquet."""
    pos = pd.read_parquet(positions_path)
    model = GAT()
    model.load_state_dict(torch.load(ckpt, weights_only=True))
    model.eval()
    rows: list[dict] = []
    for fr, grp in pos.groupby("frame"):
        frame = frame_to_statsbomb(grp)
        if frame is None:
            continue
        ball = grp[grp["role"] == "ball"].dropna(subset=["pitch_x", "pitch_y"])
        bx = float(ball.iloc[0]["pitch_x"] * PITCH_LENGTH / SRC_LEN) if len(ball) else 60.0
        by = float(ball.iloc[0]["pitch_y"] * PITCH_WIDTH / SRC_WID) if len(ball) else 40.0
        row = SimpleNamespace(
            trigger_x=bx,
            trigger_y=by,
            play_pattern="Regular Play",
            from_counter=False,
            trigger_time_s=600.0,
        )
        try:
            out = model(build_data(frame, row))
        except Exception as exc:  # noqa: BLE001
            logger.warning("frame %s: %s", fr, exc)
            continue
        rows.append(
            {
                "frame": int(fr),
                "n_players": int(frame["teammate"].size),
                "success": float(torch.sigmoid(out["success"])),
                "dynamic_xt": float(out["dxt"]),
                "p_defstop": float(torch.sigmoid(out["defsuccess"])),
                "top_receiver": (
                    float(out["receiver_probs"].max()) if "receiver_probs" in out else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Run the bridge on a positions parquet and print a clip-level readout."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", required=True, help="cv-football positions parquet")
    args = ap.parse_args()
    df = run_clip(args.positions)
    if df.empty:
        print(f"No frames with >= {MIN_PLAYERS} visible players (broadcast too zoomed).")
        return
    print(f"\nRan the GAT on {len(df)} qualifying frames ({MIN_PLAYERS}+ players).")
    print("Clip-level relational readout (means):")
    print(f"  P(success)     : {df['success'].mean():.3f}")
    print(f"  Dynamic-xT     : {df['dynamic_xt'].mean():.4f}")
    print(f"  P(def recovers): {df['p_defstop'].mean():.3f}")
    print("\nVideo -> relational-metrics path; aggregate per team for a CV team identity.")


if __name__ == "__main__":
    main()
