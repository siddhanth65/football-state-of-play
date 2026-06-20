"""Per-player run prediction on tracking data — the capability sparse 360 could not support.

StatsBomb 360 has no player identities and no motion, so "where will this player run in 1.5 s"
was ill-posed (the project's run head barely beat "they don't move"). Dense tracking
(persistent IDs, ~25 fps) fixes both: we can read each player's **true** position 1.5 s later
and their **actual velocity**. This trains a small per-player run head on that and compares it
to two baselines, to show motion data turns run prediction from a near-null into a real signal.

- **no-move**: predict zero displacement (the 360-era floor).
- **constant-velocity**: extrapolate the player's current velocity for 1.5 s (uses motion —
  impossible on 360).
- **learned**: a Ridge regressor on (position, velocity, distance/angle to goal) -> displacement.

Run ``python -m eval.tracking_runs`` (needs network: kloppy fetches Metrica open data).
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

logger = logging.getLogger(__name__)
SRC_LEN, SRC_WID = 105.0, 68.0
HORIZON_S = 1.5
VEL_WINDOW_S = 0.5
OPP_GOAL = np.array([SRC_LEN, SRC_WID / 2])


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(a, b)))


def build_run_examples(dataset, sample_every: int = 10) -> pd.DataFrame:
    """One row per (player, frame): current state, velocity, and the true +1.5 s displacement."""
    fps = dataset.metadata.frame_rate or 25
    look, vw = int(HORIZON_S * fps), max(1, int(VEL_WINDOW_S * fps))
    recs = dataset.records
    rows: list[dict] = []
    for i in range(vw, len(recs) - look, sample_every):
        cur, past, fut = (
            recs[i].players_coordinates,
            recs[i - vw].players_coordinates,
            recs[i + look].players_coordinates,
        )
        for p, pt in cur.items():
            if pt is None or past.get(p) is None or fut.get(p) is None:
                continue
            cx, cy = pt.x * SRC_LEN, pt.y * SRC_WID
            fx, fy = fut[p].x * SRC_LEN, fut[p].y * SRC_WID
            ox, oy = past[p].x * SRC_LEN, past[p].y * SRC_WID
            vx, vy = (cx - ox) / VEL_WINDOW_S, (cy - oy) / VEL_WINDOW_S  # m/s
            to_goal = OPP_GOAL - np.array([cx, cy])
            rows.append(
                {
                    "frame": i,
                    "x": cx,
                    "y": cy,
                    "vx": vx,
                    "vy": vy,
                    "dist_goal": float(np.hypot(*to_goal)),
                    "angle_goal": float(np.arctan2(*to_goal[::-1])),
                    "dx": fx - cx,
                    "dy": fy - cy,
                }
            )
    return pd.DataFrame(rows)


def evaluate_runs(df: pd.DataFrame) -> dict[str, float]:
    """Time-split train/test; compare no-move, constant-velocity, and a learned predictor (m)."""
    cut = df["frame"].quantile(0.7)
    tr, te = df[df["frame"] <= cut], df[df["frame"] > cut]
    feats = ["x", "y", "vx", "vy", "dist_goal", "angle_goal"]
    y_te = te[["dx", "dy"]].to_numpy()
    nomove = _rmse(y_te, np.zeros_like(y_te))
    constvel = _rmse(y_te, te[["vx", "vy"]].to_numpy() * HORIZON_S)
    model = Ridge(alpha=1.0).fit(tr[feats], tr[["dx", "dy"]])
    learned = _rmse(y_te, model.predict(te[feats]))
    return {
        "n_train": len(tr),
        "n_test": len(te),
        "nomove_rmse_m": round(nomove, 3),
        "constvel_rmse_m": round(constvel, 3),
        "learned_rmse_m": round(learned, 3),
    }


def main() -> None:
    """Build run examples from Metrica tracking and report the three predictors."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="metrica", choices=["metrica", "skillcorner"])
    ap.add_argument("--match-ids", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--limit", type=int, default=30000, help="frames per match")
    args = ap.parse_args()

    from eval.tracking_bridge import _load

    parts = []
    for mid in args.match_ids:
        try:
            ds = _load(args.provider, mid, args.limit)
        except Exception as exc:  # noqa: BLE001
            logger.warning("skip %s %s: %s", args.provider, mid, str(exc)[:140])
            continue
        ex = build_run_examples(ds)
        ex["match"] = mid
        parts.append(ex)
        logger.info("%s match %s: %d run examples", args.provider, mid, len(ex))
    if not parts:
        print("no data loaded")
        return
    df = pd.concat(parts, ignore_index=True)
    df["frame"] = df["frame"] + df["match"] * 10_000_000  # keep the time-split per match-ordered
    res = evaluate_runs(df)
    print("\nPer-player run prediction (1.5 s ahead) on tracking, RMSE in metres:")
    print(f"  no-move (360-era floor) : {res['nomove_rmse_m']} m")
    print(f"  constant-velocity       : {res['constvel_rmse_m']} m   (uses motion, no 360)")
    print(f"  learned (Ridge)         : {res['learned_rmse_m']} m")
    print(f"  [{res['n_train']} train / {res['n_test']} test examples]")
    gain = 100 * (res["nomove_rmse_m"] - res["learned_rmse_m"]) / res["nomove_rmse_m"]
    print(
        f"\nLearned beats no-move by {gain:.0f}% -- run prediction is well-posed on tracking, "
        "unlike sparse 360 where it barely cleared no-move."
    )


if __name__ == "__main__":
    main()
