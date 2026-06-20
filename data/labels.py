"""Target-label computation for attacking-third possessions (CLAUDE.md §4.4).

Consumes the possession index from :mod:`data.possessions` and produces the three
training targets:

- ``success`` — shot within 15 s OR penalty-area entry within 10 s (event-based).
- ``xt_progression`` — ``xT(end) - xT(trigger)`` on Karun Singh's **16×12** grid
  (Singh 2019), built from the pooled events and cached to ``xt_grid.npy``.
- ``run_target_(x, y)`` — the furthest-forward attacker's position ~1.5 s later.
  StatsBomb 360 frames carry no player IDs, so the attacker is matched to the
  +1.5 s frame by **nearest-neighbour** (approximate; flagged in week1.md).

The ≥10-visible-player filter (CLAUDE.md §4.3) is applied here because it needs the
freeze frame. Run as ``python -m data.labels`` (after ``data.possessions``, or via
``make data``) to build ``data/processed/possessions.parquet``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from data import load
from data.possessions import (
    BOX_X_MIN,
    BOX_Y_MAX,
    BOX_Y_MIN,
    MIN_VISIBLE_PLAYERS,
    PITCH_LENGTH,
    PITCH_WIDTH,
    POSSESSIONS_INDEX_PATH,
    SUCCESS_BOX_HORIZON_S,
    SUCCESS_SHOT_HORIZON_S,
    add_event_time,
    xy,
)

logger = logging.getLogger(__name__)

# --- Paths -----------------------------------------------------------------
PROCESSED_DIR = Path("data/processed")
POSSESSIONS_PATH = PROCESSED_DIR / "possessions.parquet"
XT_GRID_PATH = PROCESSED_DIR / "xt_grid.npy"

# --- xT grid resolution (Karun Singh 2019: 16 x 12 = 192 zones) ------------
XT_NX = 16
XT_NY = 12
XT_ITERS = 5

# Drop a run-target match if the nearest teammate in the +1.5 s frame is implausibly
# far from the trigger attacker (≈ max sprint distance in 1.5 s): the player likely
# left the visible area, so the nearest-neighbour match would be spurious.
MAX_RUN_MATCH_DIST = 15.0

# Defensive-success label: the defending team makes a ball-winning action within this
# window of the trigger (the mirror of the attacking success head).
DEF_STOP_HORIZON_S = 8.0
DEF_WIN_TYPES = frozenset({"Ball Recovery", "Interception", "Block", "Clearance"})


# === xT grid ===============================================================
def zone_index(x: np.ndarray, y: np.ndarray, n_x: int = XT_NX, n_y: int = XT_NY) -> np.ndarray:
    """Map pitch coordinates to flat grid-zone indices ``ix * n_y + iy``.

    Args:
        x: x-coordinates in [0, 120].
        y: y-coordinates in [0, 80].
        n_x: zones along the pitch length.
        n_y: zones across the pitch width.

    Returns:
        Integer array of flat zone indices in ``[0, n_x * n_y)``.
    """
    ix = np.clip((np.asarray(x) / PITCH_LENGTH * n_x).astype(int), 0, n_x - 1)
    iy = np.clip((np.asarray(y) / PITCH_WIDTH * n_y).astype(int), 0, n_y - 1)
    return ix * n_y + iy


class XTGridBuilder:
    """Accumulates move/shot/goal/transition counts to build Karun Singh's xT grid.

    Fed one match's events at a time (so the full event corpus need not live in
    memory), then :meth:`finalize` solves the iterative xT recursion.
    """

    def __init__(self, n_x: int = XT_NX, n_y: int = XT_NY) -> None:
        self.n_x, self.n_y = n_x, n_y
        n = n_x * n_y
        self.shots = np.zeros(n)
        self.goals = np.zeros(n)
        self.moves = np.zeros(n)
        self.trans = np.zeros((n, n))

    def add(self, events: pd.DataFrame) -> None:
        """Accumulate one match's attacking actions into the running counts."""
        ev = events[events["team"] == events["possession_team"]]

        shots = ev[(ev["type"] == "Shot") & ev["location"].notna()]
        if not shots.empty:
            sx = np.array([xy(c)[0] for c in shots["location"]])
            sy = np.array([xy(c)[1] for c in shots["location"]])
            z = zone_index(sx, sy, self.n_x, self.n_y)
            np.add.at(self.shots, z, 1)
            goal = shots["shot_outcome"] == "Goal"
            np.add.at(self.goals, z[goal.to_numpy()], 1)

        self._add_moves(
            ev, "location", "pass_end_location", type_value="Pass", success_col="pass_outcome"
        )
        self._add_moves(ev, "location", "carry_end_location", type_value="Carry")

    def _add_moves(
        self,
        ev: pd.DataFrame,
        start_col: str,
        end_col: str,
        type_value: str,
        success_col: str | None = None,
    ) -> None:
        m = ev[(ev["type"] == type_value) & ev[start_col].notna() & ev[end_col].notna()]
        if success_col is not None:  # StatsBomb: NaN pass_outcome == completed
            m = m[m[success_col].isna()]
        if m.empty:
            return
        sx = np.array([xy(c)[0] for c in m[start_col]])
        sy = np.array([xy(c)[1] for c in m[start_col]])
        ex = np.array([xy(c)[0] for c in m[end_col]])
        ey = np.array([xy(c)[1] for c in m[end_col]])
        z_from = zone_index(sx, sy, self.n_x, self.n_y)
        z_to = zone_index(ex, ey, self.n_x, self.n_y)
        np.add.at(self.moves, z_from, 1)
        np.add.at(self.trans, (z_from, z_to), 1)

    def finalize(self) -> np.ndarray:
        """Solve the xT recursion and return the ``(n_x, n_y)`` grid."""
        total = self.shots + self.moves
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.where(total > 0, self.shots / total, 0.0)
            m = np.where(total > 0, self.moves / total, 0.0)
            g = np.where(self.shots > 0, self.goals / self.shots, 0.0)
            row = self.trans.sum(axis=1, keepdims=True)
            t_norm = np.where(row > 0, self.trans / row, 0.0)

        xt = np.zeros(self.n_x * self.n_y)
        shoot_payoff = s * g
        for _ in range(XT_ITERS):
            xt = shoot_payoff + m * (t_norm @ xt)
        return xt.reshape(self.n_x, self.n_y)


def xt_value(x: float, y: float, grid: np.ndarray) -> float:
    """Bilinearly interpolate the xT grid at ``(x, y)`` (Singh 2019).

    Args:
        x: x-coordinate in [0, 120].
        y: y-coordinate in [0, 80].
        grid: An ``(n_x, n_y)`` xT grid.

    Returns:
        The interpolated xT value at the location.
    """
    n_x, n_y = grid.shape
    # Continuous grid coordinates relative to cell centres.
    gx = np.clip(x / PITCH_LENGTH * n_x - 0.5, 0, n_x - 1)
    gy = np.clip(y / PITCH_WIDTH * n_y - 0.5, 0, n_y - 1)
    i0, j0 = int(np.floor(gx)), int(np.floor(gy))
    i1, j1 = min(i0 + 1, n_x - 1), min(j0 + 1, n_y - 1)
    fx, fy = gx - i0, gy - j0
    top = grid[i0, j0] * (1 - fx) + grid[i1, j0] * fx
    bot = grid[i0, j1] * (1 - fx) + grid[i1, j1] * fx
    return float(top * (1 - fy) + bot * fy)


def load_or_build_xt_grid(
    match_records: list[dict] | None = None,
    limit: int | None = None,
    path: Path = XT_GRID_PATH,
) -> np.ndarray:
    """Load the cached xT grid, or build it from the pooled events and cache it.

    Args:
        match_records: Match records to build from; defaults to the cached list.
        limit: Only use the first ``limit`` matches when building (smoke testing).
        path: Cache location for ``xt_grid.npy``.

    Returns:
        The ``(XT_NX, XT_NY)`` xT grid.
    """
    if path.exists():
        return np.load(path)
    if match_records is None:
        match_records = load.load_cached_match_ids()
    if limit is not None:
        match_records = match_records[:limit]

    builder = XTGridBuilder()
    for rec in match_records:
        try:
            builder.add(load.load_events(rec["match_id"]))
        except Exception as exc:  # noqa: BLE001 - log and continue per match
            logger.warning("xT grid: skipping match %s: %s", rec["match_id"], exc)
    grid = builder.finalize()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, grid)
    logger.info("Built + cached xT grid %s -> %s", grid.shape, path)
    return grid


# === Event-based labels ====================================================
def compute_success(
    possession_events: pd.DataFrame, possession_team: object, trigger_time: float
) -> bool:
    """Whether the possession succeeds within the horizons after the trigger.

    Args:
        possession_events: Events of the trigger's possession, with ``time_s``.
        possession_team: The attacking team (filters out defensive events).
        trigger_time: Trigger ``time_s`` (period-relative seconds).

    Returns:
        TRUE if a shot occurs within 15 s OR the ball reaches the penalty area
        within 10 s of the trigger, same possession.
    """
    ev = possession_events[
        (possession_events["team"] == possession_team)
        # >= (not >): a shot or penalty-area entry *at* the trigger instant counts as
        # success — the possession has already reached threat (was previously dropped).
        & (possession_events["time_s"] >= trigger_time)
    ]
    if ev.empty:
        return False
    dt = ev["time_s"] - trigger_time

    shot = (ev["type"] == "Shot") & (dt <= SUCCESS_SHOT_HORIZON_S)
    if bool(shot.any()):
        return True

    in_box = dt <= SUCCESS_BOX_HORIZON_S
    for col in ("location", "pass_end_location", "carry_end_location"):
        if col not in ev.columns:
            continue
        pts = ev.loc[in_box, col].map(xy)
        for p in pts:
            if p and p[0] >= BOX_X_MIN and BOX_Y_MIN <= p[1] <= BOX_Y_MAX:
                return True
    return False


def compute_xt_progression(
    trigger_x: float, trigger_y: float, end_x: float, end_y: float, grid: np.ndarray
) -> float:
    """``xT(end) - xT(trigger)`` via bilinear interpolation on the grid."""
    return xt_value(end_x, end_y, grid) - xt_value(trigger_x, trigger_y, grid)


def compute_def_stop(events: pd.DataFrame, possession_team: object, trigger_time: float) -> bool:
    """Whether the defending team makes a ball-winning action within the horizon.

    The defensive mirror of :func:`compute_success`: TRUE if any opponent
    Ball Recovery / Interception / Block / Clearance occurs within
    :data:`DEF_STOP_HORIZON_S` of the trigger (the defence actively stopped the attack).

    Args:
        events: The **full match** events with ``time_s`` (not just the possession).
        possession_team: The attacking team (the defenders are everyone else).
        trigger_time: Trigger ``time_s`` (period-relative seconds).
    """
    if not {"time_s", "team", "type"}.issubset(events.columns):
        return False
    w = events[
        (events["time_s"] > trigger_time)
        & (events["time_s"] <= trigger_time + DEF_STOP_HORIZON_S)
        & (events["team"] != possession_team)
        & (events["type"].isin(DEF_WIN_TYPES))
    ]
    return bool(len(w) > 0)


def compute_offside_line(frame: pd.DataFrame) -> float | None:
    """The offside line x: the second-deepest outfield defender (nearest the defended goal).

    Defenders are the non-teammates excluding the keeper; the offside line is set by the
    second-last defender (standard rule). Returns ``None`` if fewer than two are visible.
    """
    defenders = _frame_points(frame, teammate=False, exclude_actor=False)
    # Exclude the keeper (the deepest player) so the line is the outfield back line.
    if "keeper" not in frame.columns:
        return float(np.sort(defenders[:, 0])[-2]) if len(defenders) >= 2 else None
    keeper = frame[(~frame["teammate"].astype(bool)) & (frame["keeper"].astype(bool))]
    if len(keeper):
        kp = np.array([xy(c) for c in keeper["location"] if xy(c) is not None]).reshape(-1, 2)
        if kp.size:
            # Drop defender points coincident with a keeper point.
            keep = [not np.any(np.all(np.isclose(kp, d), axis=1)) for d in defenders]
            defenders = defenders[keep]
    if len(defenders) < 2:
        return None
    xs = np.sort(defenders[:, 0])  # ascending; deepest defender = largest x
    return float(xs[-2])  # second-deepest = the offside line


# === Frame-based label (run target) ========================================
def _frame_points(frame: pd.DataFrame, teammate: bool, exclude_actor: bool = False) -> np.ndarray:
    """Return an ``(n, 2)`` array of player positions matching the flags."""
    sel = frame[frame["teammate"] == teammate]
    if exclude_actor:
        sel = sel[~sel["actor"].astype(bool)]
    pts = [xy(c) for c in sel["location"]]
    return np.array([p for p in pts if p is not None], dtype=float).reshape(-1, 2)


def compute_run_target(
    trigger_frame: pd.DataFrame, target_frame: pd.DataFrame
) -> tuple[float, float] | None:
    """Nearest-neighbour run target for the furthest-forward attacker.

    Args:
        trigger_frame: Freeze-frame rows for the trigger event.
        target_frame: Freeze-frame rows for the ≈ trigger + 1.5 s event.

    Returns:
        ``(x, y)`` of the matched attacker in the target frame, or ``None`` if it
        cannot be resolved (no attacker / no teammates in the target frame).
    """
    attackers = _frame_points(trigger_frame, teammate=True, exclude_actor=True)
    if attackers.size == 0:
        return None
    p0 = attackers[np.argmax(attackers[:, 0])]  # furthest forward (max x), §3 #3

    # Exclude the carrier in the target frame too: the +1.5 s actor is whoever has the
    # ball then (usually a *different* player), so allowing it as a match candidate would
    # let the run target jump to the ball-carrier rather than track the original runner.
    teammates = _frame_points(target_frame, teammate=True, exclude_actor=True)
    if teammates.size == 0:
        return None
    dists = np.linalg.norm(teammates - p0, axis=1)
    j = int(np.argmin(dists))
    if dists[j] > MAX_RUN_MATCH_DIST:
        return None  # attacker not plausibly visible 1.5 s later -> spurious match
    return float(teammates[j][0]), float(teammates[j][1])


# === Orchestration =========================================================
def build_labeled_dataset(
    index: pd.DataFrame,
    grid: np.ndarray,
    load_events_fn=load.load_events_local,
    load_frames_fn=load.load_frames_local,
) -> pd.DataFrame:
    """Attach all three labels to the possession index and apply the ≥10 filter.

    Args:
        index: Possession index from :func:`data.possessions.build_possessions_index`.
        grid: The xT grid (for ``xt_progression``).
        load_events_fn: ``match_id -> events DataFrame`` (injectable for tests).
        load_frames_fn: ``match_id -> frames DataFrame`` (injectable for tests).

    Returns:
        The labelled dataset: index columns + ``success``, ``xt_progression``,
        ``n_visible``, ``run_target_x/y``. Rows that are set-piece-proximate
        (``from_set_piece``), have < 10 visible players at the trigger, or have no
        resolvable run target, are dropped (open-play modelling set).
    """
    rows: list[dict] = []
    for mid, idx_grp in index.groupby("match_id", sort=False):
        try:
            events = add_event_time(load_events_fn(mid))
            frames = load_frames_fn(mid)
        except Exception as exc:  # noqa: BLE001 - log and continue per match
            logger.warning("labels: skipping match %s: %s", mid, exc)
            continue
        events_by_poss = dict(tuple(events.groupby("possession")))
        frames_by_event = dict(tuple(frames.groupby("id")))

        for row in idx_grp.itertuples(index=False):
            # Open-play only (CLAUDE.md §4.3 step 7): drop triggers within 5 s of a
            # set-piece restart. They are pooled-but-tagged in the index; the modelling
            # dataset excludes them (set-piece dynamics differ and the subset is far
            # easier, which otherwise inflates the headline — see REPORT §4).
            if bool(getattr(row, "from_set_piece", False)):
                continue

            poss_events = events_by_poss.get(row.possession)
            if poss_events is None:
                continue

            trigger_frame = frames_by_event.get(row.trigger_event_id)
            if trigger_frame is None or len(trigger_frame) < MIN_VISIBLE_PLAYERS:
                continue  # ≥10 visible filter (CLAUDE.md §4.3)

            target_frame = frames_by_event.get(row.run_target_event_id)
            if target_frame is None:
                continue  # no +1.5 s freeze frame within tolerance
            run = compute_run_target(trigger_frame, target_frame)
            if run is None:
                continue

            rec = row._asdict()
            rec["n_visible"] = int(len(trigger_frame))
            rec["success"] = compute_success(poss_events, row.possession_team, row.trigger_time_s)
            rec["xt_progression"] = compute_xt_progression(
                row.trigger_x, row.trigger_y, row.end_x, row.end_y, grid
            )
            rec["run_target_x"], rec["run_target_y"] = run
            # Defensive heads: a defensive-recovery outcome + the offside-line shift.
            rec["def_stop"] = compute_def_stop(events, row.possession_team, row.trigger_time_s)
            off_trig = compute_offside_line(trigger_frame)
            off_tgt = compute_offside_line(target_frame)
            rec["offside_trigger_x"] = off_trig if off_trig is not None else np.nan
            rec["defline_delta_x"] = (
                (off_tgt - off_trig) if (off_trig is not None and off_tgt is not None) else np.nan
            )
            rows.append(rec)

    return pd.DataFrame(rows)


def cache_dataset(dataset: pd.DataFrame, out_path: Path = POSSESSIONS_PATH) -> Path:
    """Write the labelled possessions dataset to parquet."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(out_path, index=False)
    logger.info("Wrote %d labelled possessions to %s", len(dataset), out_path)
    return out_path


def main() -> None:
    """Build labels for the cached possession index and write the final dataset."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if not POSSESSIONS_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"{POSSESSIONS_INDEX_PATH} not found. Run `python -m data.possessions` first."
        )
    index = pd.read_parquet(POSSESSIONS_INDEX_PATH)
    grid = load_or_build_xt_grid()
    dataset = build_labeled_dataset(index, grid)
    cache_dataset(dataset)

    n = len(dataset)
    print(f"\nLabelled possessions: {n}")
    if n:
        print(f"  success rate:        {dataset['success'].mean():.3f}")
        print(f"  xt_progression mean: {dataset['xt_progression'].mean():+.4f}")
        print(f"  counters:            {int(dataset['from_counter'].sum())}")
        print(f"  open play:           {int((~dataset['from_set_piece']).sum())}")


if __name__ == "__main__":
    main()
