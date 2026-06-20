"""Freeze frame → ``torch_geometric.data.Data`` graph (CLAUDE.md §5.1).

Turns one trigger freeze frame (the visible players) plus its possession row into a
graph the GAT / transformer consume:

- **Nodes** = visible players; node features = position, velocity, role flags,
  distances/angles to ball and goals.
- **Edges** = fully connected (directed both ways); edge features = distance, relative
  velocity along the line, and the **pass-line** features (nearest-defender distance and
  angle to the line) that Power et al. (2017) found most predictive, replacing the crude
  ``crosses_other_player``.
- **Globals** (``u``) = play-pattern one-hot + counter flag + time-in-half.
- **Targets** = ``y_success`` / ``y_xt`` / ``y_run`` from :mod:`data.labels`.

Velocity needs the previous freeze frame in the possession; since 360 frames have no
player IDs it is matched by nearest-neighbour (approximate), and zero-filled when no
prior frame is supplied. Attacking direction is left → right (so the attacked goal is at
``x = 120``).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from data.load import load_frames_local  # re-exported: app/eval import it from here
from data.possessions import PITCH_LENGTH, PITCH_WIDTH, ROLE_BUCKETS, xy
from features.obso import obso  # Dynamic-xT target (freeze-frame scoring opportunity)

logger = logging.getLogger(__name__)
GRAPH_CACHE_PATH = Path("data/processed/graphs.pt")


# Goal mouths (attacking left → right): own goal behind, opponent goal ahead.
OWN_GOAL = np.array([0.0, PITCH_WIDTH / 2])
OPP_GOAL = np.array([PITCH_LENGTH, PITCH_WIDTH / 2])
PITCH_DIAG = float(np.hypot(PITCH_LENGTH, PITCH_WIDTH))

# Cap for nearest-neighbour velocity estimates (m/s): above an elite sprint (~11 m/s),
# a value is almost certainly a bad cross-frame match, not real motion.
MAX_PLAYER_SPEED = 12.0

# Global feature layout: play-pattern one-hot + counter + time-in-half.
PLAY_PATTERNS = (
    "Regular Play",
    "From Counter",
    "From Free Kick",
    "From Throw In",
    "From Corner",
    "From Goal Kick",
    "From Kick Off",
    "From Keeper",
    "Other",
)
HALF_SECONDS = 45 * 60

NODE_FEATURE_NAMES = (
    "x",
    "y",
    "vx",
    "vy",
    "is_teammate",
    "is_actor",
    "is_keeper",
    "dist_ball",
    "dist_own_goal",
    "dist_opp_goal",
    "angle_opp_goal",
    "angle_ball",
)
EDGE_FEATURE_NAMES = ("dist", "rel_vel_along", "def_dist_to_line", "def_angle_to_line")
N_NODE_FEATURES = len(NODE_FEATURE_NAMES)
N_EDGE_FEATURES = len(EDGE_FEATURE_NAMES)
# Globals = play-pattern one-hot + counter + time (base) ++ 4 defending-team shape
# descriptors (line height, compactness, width, numerical balance) adopted into the headline
# after the §4.10 experiment showed they help every head.
N_BASE_GLOBAL_FEATURES = len(PLAY_PATTERNS) + 2
N_DEF_SHAPE_FEATURES = 4
N_GLOBAL_FEATURES = N_BASE_GLOBAL_FEATURES + N_DEF_SHAPE_FEATURES


def frame_points(frame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract positions and role flags from a freeze-frame DataFrame.

    Args:
        frame: Freeze-frame rows for one event (``location``, ``teammate``,
            ``actor``, ``keeper``).

    Returns:
        ``(pts, teammate, actor, keeper)`` where ``pts`` is ``(N, 2)`` and the flags
        are length-``N`` boolean arrays, for the rows with a valid location.
    """
    pts, tm, ac, kp = [], [], [], []
    for row in frame.itertuples(index=False):
        p = xy(row.location)
        if p is None:
            continue
        pts.append(p)
        tm.append(bool(row.teammate))
        ac.append(bool(getattr(row, "actor", False)))
        kp.append(bool(getattr(row, "keeper", False)))
    return (
        np.asarray(pts, dtype=float).reshape(-1, 2),
        np.asarray(tm, dtype=bool),
        np.asarray(ac, dtype=bool),
        np.asarray(kp, dtype=bool),
    )


def nearest_neighbour_velocity(pts: np.ndarray, prev_pts: np.ndarray, dt: float) -> np.ndarray:
    """Estimate per-player velocity by matching to the nearest previous-frame point.

    Approximate because 360 frames lack player IDs. Returns zeros when there is no
    previous frame or ``dt`` is not positive.

    Args:
        pts: ``(N, 2)`` current positions.
        prev_pts: ``(M, 2)`` previous-frame positions (may be empty).
        dt: Seconds between the frames.

    Returns:
        ``(N, 2)`` velocity vectors in pitch units per second.
    """
    if prev_pts.size == 0 or dt <= 0:
        return np.zeros_like(pts)
    vel = np.zeros_like(pts)
    for i, p in enumerate(pts):
        j = int(np.argmin(np.linalg.norm(prev_pts - p, axis=1)))
        vel[i] = (p - prev_pts[j]) / dt
    # Cap to a plausible sprint speed. Nearest-neighbour matching (no player IDs) over a
    # small dt can pair the wrong players and produce absurd speeds (>700 m/s in the raw
    # cache), which then dominate the normalised node/edge features; clamp the magnitude.
    speed = np.linalg.norm(vel, axis=1)
    over = speed > MAX_PLAYER_SPEED
    if over.any():
        vel[over] *= (MAX_PLAYER_SPEED / speed[over])[:, None]
    return vel


def _angle(vectors: np.ndarray) -> np.ndarray:
    """Angle (radians) of each row vector via ``atan2(dy, dx)``."""
    return np.arctan2(vectors[:, 1], vectors[:, 0])


def node_features(
    pts: np.ndarray,
    teammate: np.ndarray,
    actor: np.ndarray,
    keeper: np.ndarray,
    ball: np.ndarray,
    velocity: np.ndarray,
) -> np.ndarray:
    """Build the ``(N, N_NODE_FEATURES)`` node feature matrix (normalised)."""
    n = len(pts)
    norm = np.array([PITCH_LENGTH, PITCH_WIDTH])
    to_ball = ball - pts
    to_opp = OPP_GOAL - pts
    feats = np.column_stack(
        [
            pts / norm,  # x, y
            velocity / norm,  # vx, vy (per second, same scale)
            teammate.astype(float),
            actor.astype(float),
            keeper.astype(float),
            np.linalg.norm(to_ball, axis=1) / PITCH_DIAG,
            np.linalg.norm(pts - OWN_GOAL, axis=1) / PITCH_DIAG,
            np.linalg.norm(to_opp, axis=1) / PITCH_DIAG,
            _angle(to_opp) / np.pi,
            _angle(to_ball) / np.pi,
        ]
    )
    return feats.reshape(n, N_NODE_FEATURES)


def _point_segment_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from each point to the segment ``a→b`` (vectorised)."""
    ab = b - a
    denom = float(ab @ ab)
    if denom == 0.0:
        return np.linalg.norm(points - a, axis=1)
    t = np.clip((points - a) @ ab / denom, 0.0, 1.0)
    proj = a + np.outer(t, ab)
    return np.linalg.norm(points - proj, axis=1)


def edge_features(
    pts: np.ndarray, teammate: np.ndarray, velocity: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Build directed all-pairs edges and their features.

    Args:
        pts: ``(N, 2)`` positions.
        teammate: length-``N`` attacking-team mask (defenders are ``~teammate``).
        velocity: ``(N, 2)`` velocities.

    Returns:
        ``(edge_index (2, E), edge_attr (E, N_EDGE_FEATURES))``. ``E = N*(N-1)``.
    """
    n = len(pts)
    defenders = pts[~teammate]
    src, dst, attr = [], [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            line = pts[j] - pts[i]
            dist = float(np.linalg.norm(line))
            unit = line / dist if dist > 0 else np.zeros(2)
            rel_vel_along = float((velocity[i] - velocity[j]) @ unit)
            if len(defenders):
                d2line = _point_segment_distance(defenders, pts[i], pts[j])
                k = int(np.argmin(d2line))
                def_dist = float(d2line[k]) / PITCH_DIAG
                to_def = defenders[k] - pts[i]
                # Angle between the pass line and the line to the nearest defender.
                def_angle = (
                    float(
                        np.arctan2(
                            line[0] * to_def[1] - line[1] * to_def[0],
                            line[0] * to_def[0] + line[1] * to_def[1],
                        )
                    )
                    / np.pi
                )
            else:
                def_dist, def_angle = 1.0, 0.0
            src.append(i)
            dst.append(j)
            attr.append([dist / PITCH_DIAG, rel_vel_along / PITCH_DIAG, def_dist, def_angle])
    edge_index = np.array([src, dst], dtype=np.int64) if src else np.zeros((2, 0), dtype=np.int64)
    edge_attr = np.array(attr, dtype=float).reshape(-1, N_EDGE_FEATURES)
    return edge_index, edge_attr


def global_features(play_pattern: object, from_counter: bool, trigger_time_s: float) -> np.ndarray:
    """Build the ``(N_BASE_GLOBAL_FEATURES,)`` base graph-level feature vector."""
    one_hot = np.zeros(len(PLAY_PATTERNS))
    if play_pattern in PLAY_PATTERNS:
        one_hot[PLAY_PATTERNS.index(play_pattern)] = 1.0
    time_remaining = float(np.clip((HALF_SECONDS - trigger_time_s) / HALF_SECONDS, 0.0, 1.0))
    return np.concatenate([one_hot, [float(from_counter), time_remaining]])


def defensive_shape_globals(
    pts: np.ndarray, teammate: np.ndarray, keeper: np.ndarray, ball: np.ndarray
) -> np.ndarray:
    """Four defending-team shape descriptors (normalised) appended to the globals.

    Line height (offside line = 2nd-deepest outfield defender), compactness (positional
    spread), width, and numerical balance ahead of the ball (defenders − attackers). Adopted
    after the §4.10 experiment showed they help every head. Zeros if too few defenders.
    """
    defenders = (~teammate) & (~keeper)
    if int(defenders.sum()) < 2:
        return np.zeros(N_DEF_SHAPE_FEATURES)
    dx = pts[defenders, 0] / PITCH_LENGTH
    dy = pts[defenders, 1] / PITCH_WIDTH
    line_height = float(np.sort(dx)[-2])
    compactness = float(np.std(dx) + np.std(dy))
    width = float(dy.max() - dy.min())
    bx = float(ball[0])
    atk_ahead = int((pts[teammate, 0] > bx).sum()) if teammate.any() else 0
    def_ahead = int((pts[defenders, 0] > bx).sum())
    balance = (def_ahead - atk_ahead) / 11.0
    return np.array([line_height, compactness, width, balance], dtype=float)


def build_data(
    frame,
    row,
    prev_frame=None,
    dt: float | None = None,
) -> Data:
    """Assemble a ``torch_geometric.data.Data`` for one trigger freeze frame.

    Args:
        frame: Trigger freeze-frame rows (the visible players).
        row: The possession row (a namedtuple/Series) with ``trigger_x/y``,
            ``play_pattern``, ``from_counter``, ``trigger_time_s`` and — if present —
            the labels ``success`` / ``xt_progression`` / ``run_target_x/y``.
        prev_frame: The previous freeze frame in the possession (for velocity), or None.
        dt: Seconds between ``prev_frame`` and ``frame`` (for velocity).

    Returns:
        A ``Data`` with ``x``, ``edge_index``, ``edge_attr``, ``u`` (globals) and, when
        the labels are present on ``row``, ``y_success`` / ``y_xt`` / ``y_run``.
    """
    pts, teammate, actor, keeper = frame_points(frame)
    if len(pts) == 0:
        raise ValueError("freeze frame has no players with valid locations")

    # Ball position: the actor's location if visible, else the trigger location.
    ball = pts[actor][0] if actor.any() else np.array([float(row.trigger_x), float(row.trigger_y)])

    velocity = np.zeros_like(pts)
    if prev_frame is not None and dt:
        prev_pts, *_ = frame_points(prev_frame)
        velocity = nearest_neighbour_velocity(pts, prev_pts, dt)

    x = node_features(pts, teammate, actor, keeper, ball, velocity)
    edge_index, edge_attr = edge_features(pts, teammate, velocity)
    u = global_features(row.play_pattern, bool(row.from_counter), float(row.trigger_time_s))
    u = np.concatenate([u, defensive_shape_globals(pts, teammate, keeper, ball)])

    data = Data(
        x=torch.tensor(x, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float),
        u=torch.tensor(u, dtype=torch.float).unsqueeze(0),
    )
    data.num_nodes = len(pts)
    if hasattr(row, "match_id"):
        data.match_id = int(row.match_id)
    if hasattr(row, "possession"):
        data.possession = int(row.possession)

    if hasattr(row, "success"):
        data.y_success = torch.tensor([float(row.success)], dtype=torch.float)
        data.y_xt = torch.tensor([float(row.xt_progression)], dtype=torch.float)
        # Run head predicts the furthest-forward attacker's DISPLACEMENT over ~1.5 s, not
        # its absolute future position. Regressing the absolute target off the mean+max
        # pooled graph embedding collapses to the squad centroid (RMSE ~13 m — *worse*
        # than the 6.5 m no-move baseline); predicting the delta makes "no movement" the
        # model's floor and lets it add learned drift. The anchor cancels in the RMSE, so
        # metres stay directly comparable to the no-move baseline and the old numbers.
        atk = teammate & ~actor
        if bool(atk.any()):
            anchor = pts[atk][int(np.argmax(pts[atk][:, 0]))]
        else:
            anchor = np.array([float(row.trigger_x), float(row.trigger_y)])
        data.run_anchor = torch.tensor(
            [[anchor[0] / PITCH_LENGTH, anchor[1] / PITCH_WIDTH]], dtype=torch.float
        )
        data.y_run = torch.tensor(
            [
                [
                    (float(row.run_target_x) - anchor[0]) / PITCH_LENGTH,
                    (float(row.run_target_y) - anchor[1]) / PITCH_WIDTH,
                ]
            ],
            dtype=torch.float,
        )

        # Defensive-success head: did the defending team win the ball back (BCE target).
        data.y_defsuccess = torch.tensor(
            [float(bool(getattr(row, "def_stop", False)))], dtype=torch.float
        )
        # Defensive-line head: the offside line's DISPLACEMENT over ~1.5 s (same delta
        # convention as the run head; the back line moves coherently so it is better-posed).
        dl = float(getattr(row, "defline_delta_x", float("nan")))
        data.y_defline = torch.tensor(
            [dl / PITCH_LENGTH if np.isfinite(dl) else 0.0], dtype=torch.float
        )
        data.has_defline = torch.tensor([float(np.isfinite(dl))], dtype=torch.float)
        # Dynamic-xT head: OBSO at the trigger (a learnable, configuration-conditioned threat).
        try:
            dxt = float(obso(data))
        except Exception:  # noqa: BLE001 - never let a threat-surface edge case break the cache
            dxt = 0.0
        data.y_dxt = torch.tensor([dxt], dtype=torch.float)

    # Node labels keyed off the next completed pass's end location:
    #  - H_receiver: the teammate (excl. carrier) nearest it = predicted next receiver.
    #  - H_defense (presser): the defender nearest it = predicted next defensive contester
    #    (the mirror of the receiver head — "which defender measures up to the next ball").
    # Plus the §3 #3 alternative run label. Always set every attribute (zeros + mask 0 when
    # absent) so PyG batching stays consistent.
    y_receiver = np.zeros(len(pts))
    has_receiver = 0.0
    y_presser = np.zeros(len(pts))
    has_presser = 0.0
    y_run_alt = np.zeros((1, 2))
    has_run_alt = 0.0
    raw_x = getattr(row, "next_pass_end_x", None)
    raw_y = getattr(row, "next_pass_end_y", None)
    npx = float(raw_x) if raw_x is not None else float("nan")
    npy = float(raw_y) if raw_y is not None else float("nan")
    if np.isfinite(npx) and np.isfinite(npy):
        # Displacement from the trigger ball location (same delta convention as y_run),
        # so the §3 #3 next-pass-receiver ablation stays comparable to the no-move floor.
        dx_alt = (npx - float(row.trigger_x)) / PITCH_LENGTH
        dy_alt = (npy - float(row.trigger_y)) / PITCH_WIDTH
        y_run_alt = np.array([[dx_alt, dy_alt]])
        has_run_alt = 1.0
        dist_to_next = np.linalg.norm(pts - np.array([npx, npy]), axis=1)
        cand = teammate & ~actor
        if cand.any():
            d = dist_to_next.copy()
            d[~cand] = np.inf
            y_receiver[int(np.argmin(d))] = 1.0
            has_receiver = 1.0
        defenders = ~teammate  # opponents (the defending team)
        if defenders.any():
            dd = dist_to_next.copy()
            dd[~defenders] = np.inf
            y_presser[int(np.argmin(dd))] = 1.0
            has_presser = 1.0
    data.y_receiver = torch.tensor(y_receiver, dtype=torch.float)
    data.has_receiver = torch.tensor([has_receiver], dtype=torch.float)
    data.y_presser = torch.tensor(y_presser, dtype=torch.float)
    data.has_presser = torch.tensor([has_presser], dtype=torch.float)
    # xPass / disruption: the actual next pass's completion, attached to the receiver node.
    # The head scores every attacker→teammate lane; 1 − xPass = how "closed" the lane is.
    npc = float(getattr(row, "next_pass_completed", float("nan")))
    data.y_xpass = torch.tensor([npc if np.isfinite(npc) else 0.0], dtype=torch.float)
    data.has_xpass = torch.tensor(
        [float(bool(has_receiver) and np.isfinite(npc))], dtype=torch.float
    )
    data.y_run_alt = torch.tensor(y_run_alt, dtype=torch.float)
    data.has_run_alt = torch.tensor([has_run_alt], dtype=torch.float)
    return data


def reflect_graph(g: Data) -> Data:
    """Return a laterally-mirrored copy of a graph (pitch symmetry ``y -> width - y``).

    Attacking direction is fixed left → right, so only the lateral (width) reflection is a
    label-preserving symmetry — a length flip would reverse the attack. Mirroring doubles
    the effective training data while respecting pitch symmetry: TacticAI's reflection
    augmentation, restricted to the one axis our fixed orientation allows
    (RESEARCH_INTEGRATION §3.2). Column indices follow ``NODE_FEATURE_NAMES`` /
    ``EDGE_FEATURE_NAMES``.
    """
    h = g.clone()
    h.x = h.x.clone()
    h.x[:, 1] = 1.0 - h.x[:, 1]  # y (normalised) mirrored about the midline
    h.x[:, 3] = -h.x[:, 3]  # vy sign flips
    h.x[:, 10] = -h.x[:, 10]  # angle_opp_goal sign flips (atan2 about flipped Δy)
    h.x[:, 11] = -h.x[:, 11]  # angle_ball sign flips
    if h.edge_attr is not None and h.edge_attr.numel():
        h.edge_attr = h.edge_attr.clone()
        h.edge_attr[:, 3] = -h.edge_attr[:, 3]  # signed def_angle_to_line flips
    for attr in ("y_run", "y_run_alt"):  # displacements: flip the y component
        if hasattr(h, attr):
            t = getattr(h, attr).clone()
            t[:, 1] = -t[:, 1]
            setattr(h, attr, t)
    if hasattr(h, "run_anchor"):  # absolute position: mirror about the midline
        a = h.run_anchor.clone()
        a[:, 1] = 1.0 - a[:, 1]
        h.run_anchor = a
    return h


N_NODE_ROLE_FEATURES = 3  # team-relative third: defensive / middle / forward
N_ACTOR_ROLE_FEATURES = len(ROLE_BUCKETS)  # GK / DEF / MID / FWD one-hot on the carrier


def add_position_features(g: Data, actor_role: str | None = None) -> Data:
    """Return a clone of ``g`` augmented with positional features (the position facet).

    Two additions, both honest about 360's missing identity:
      - **Per-node heuristic role** (3 dims appended to ``x``): each player's team-relative
        third (defensive / middle / forward), from that team's own x-terciles. A heuristic —
        360 has no per-dot identity — but a cheap structural prior available for every dot.
      - **Ball-carrier true role** (``len(ROLE_BUCKETS)`` dims appended to ``u``): the one
        position StatsBomb does give us (the actor's), as a graph-level one-hot.

    Kept as a standalone augmentation (not folded into the base features) so the position
    experiment (``eval/position.py``) can compare against the committed headline without
    re-baselining it.
    """
    h = g.clone()
    h.x = h.x.clone()
    px = h.x[:, 0].numpy()  # normalised x in [0, 1]
    teammate = h.x[:, 4].numpy() > 0.5
    roles = np.zeros((len(px), N_NODE_ROLE_FEATURES))
    for mask in (teammate, ~teammate):  # terciles within each team
        if not mask.any():
            continue
        sub = px[mask]
        q1, q2 = np.quantile(sub, [1 / 3, 2 / 3])
        third = np.where(sub <= q1, 0, np.where(sub <= q2, 1, 2))
        roles[np.where(mask)[0], third] = 1.0
    h.x = torch.cat([h.x, torch.tensor(roles, dtype=torch.float)], dim=1)

    onehot = np.zeros(N_ACTOR_ROLE_FEATURES)
    if actor_role in ROLE_BUCKETS:
        onehot[ROLE_BUCKETS.index(actor_role)] = 1.0
    h.u = torch.cat([h.u, torch.tensor(onehot, dtype=torch.float).unsqueeze(0)], dim=1)
    return h


# === Graph cache (materialise all possessions to disk for training) =========
def build_graph_cache(
    possessions,
    out_path: Path | None = GRAPH_CACHE_PATH,
    load_frames_fn=load_frames_local,
    limit: int | None = None,
) -> list[Data]:
    """Build one graph per labelled possession and ``torch.save`` the list.

    Loads freeze frames once per match. Velocity is differenced from the possession's
    previous freeze frame (``prev_event_id``/``prev_dt_s`` on the row) when one exists
    within 5 s; zero-filled otherwise. Possessions whose trigger frame is missing or
    empty are skipped.

    Args:
        possessions: The labelled dataset (``data/processed/possessions.parquet``).
        out_path: Where to save the graph list; ``None`` to skip saving.
        load_frames_fn: ``match_id -> frames DataFrame`` (injectable for tests).
        limit: Only process the first ``limit`` matches (smoke testing).

    Returns:
        The list of ``Data`` graphs.
    """
    match_ids = possessions["match_id"].unique()
    if limit is not None:
        match_ids = match_ids[:limit]

    graphs: list[Data] = []
    for n, mid in enumerate(match_ids, start=1):
        try:
            frames = load_frames_fn(int(mid))
        except Exception as exc:  # noqa: BLE001 - log and continue per match
            logger.warning("graphs: skipping match %s: %s", mid, exc)
            continue
        frames_by_event = dict(tuple(frames.groupby("id")))
        sub = possessions[possessions["match_id"] == mid]
        for row in sub.itertuples(index=False):
            fr = frames_by_event.get(row.trigger_event_id)
            if fr is None:
                continue
            # Previous freeze frame in the possession → real (NN-matched) velocities.
            prev_fr, dt = None, None
            pid = getattr(row, "prev_event_id", None)
            if isinstance(pid, str):
                prev_fr = frames_by_event.get(pid)
                dt = float(row.prev_dt_s)
            try:
                graphs.append(build_data(fr, row, prev_frame=prev_fr, dt=dt))
            except ValueError:
                continue
        if n % 25 == 0:
            logger.info("graphs: %d/%d matches, %d graphs", n, len(match_ids), len(graphs))

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(graphs, out_path)
        logger.info("Wrote %d graphs to %s", len(graphs), out_path)
    return graphs


def load_graph_cache(path: Path = GRAPH_CACHE_PATH) -> list[Data]:
    """Load the cached graph list (``weights_only=False`` for ``Data`` objects)."""
    return torch.load(path, weights_only=False)


def main() -> None:
    """Build and cache graphs for the full labelled dataset."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    poss = pd.read_parquet("data/processed/possessions.parquet")
    graphs = build_graph_cache(poss)
    print(f"\nBuilt {len(graphs)} graphs -> {GRAPH_CACHE_PATH}")


if __name__ == "__main__":
    main()
