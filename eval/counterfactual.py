"""Counterfactual "instinct" analysis: where *should* the key attacker move?

Predicting where the runner *will* go (the run head) is weak. So we ask the strong,
calibrated **success** head the inverse question: sweep the most-relevant attacker over a
grid of offsets, rebuild the graph at each, and read P(success). The result is a
success-probability surface and the offset that maximises it — the TacticAI-style
generative/counterfactual refinement (RESEARCH_INTEGRATION §3.8), grounded in a head that
works rather than one that does not. No retraining: it reuses a trained model at inference.

Edge/node features are recomputed from the moved positions with the same builders the
training graphs use (`data.graphs`), so the perturbed graph is exactly on-distribution; the
graph-level globals ``u`` are reused unchanged (they do not depend on the moved player).
"""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from data.graphs import PITCH_LENGTH, PITCH_WIDTH, edge_features, node_features

# Offset grid (metres) swept around the key attacker's current position.
GRID_DX = np.linspace(-16.0, 16.0, 9)
GRID_DY = np.linspace(-14.0, 14.0, 7)


def key_attacker_index(g: Data) -> int | None:
    """Index of the run subject: furthest-forward teammate excluding the carrier."""
    x = g.x.detach().cpu().numpy()
    cand = (x[:, 4] > 0.5) & (x[:, 5] < 0.5)
    if not cand.any():
        return None
    idxs = np.where(cand)[0]
    return int(idxs[int(np.argmax(x[idxs, 0]))])


def _decode(g: Data):
    """Recover (pts, velocity, teammate, actor, keeper) in pitch units from a graph."""
    x = g.x.detach().cpu().numpy()
    pts = np.column_stack([x[:, 0] * PITCH_LENGTH, x[:, 1] * PITCH_WIDTH])
    vel = np.column_stack([x[:, 2] * PITCH_LENGTH, x[:, 3] * PITCH_WIDTH])
    return pts, vel, x[:, 4] > 0.5, x[:, 5] > 0.5, x[:, 6] > 0.5


def perturbed(g: Data, key_idx: int, dx: float, dy: float) -> Data:
    """Rebuild ``g`` with the key attacker shifted by ``(dx, dy)`` metres (clamped to pitch)."""
    pts, vel, teammate, actor, keeper = _decode(g)
    pts = pts.copy()
    pts[key_idx, 0] = float(np.clip(pts[key_idx, 0] + dx, 0.0, PITCH_LENGTH))
    pts[key_idx, 1] = float(np.clip(pts[key_idx, 1] + dy, 0.0, PITCH_WIDTH))
    ball = pts[actor][0] if actor.any() else pts[key_idx]
    xf = node_features(pts, teammate, actor, keeper, ball, vel)
    ei, ea = edge_features(pts, teammate, vel)
    d = Data(
        x=torch.tensor(xf, dtype=torch.float),
        edge_index=torch.tensor(ei, dtype=torch.long),
        edge_attr=torch.tensor(ea, dtype=torch.float),
        u=g.u.clone(),
    )
    d.num_nodes = len(pts)
    return d


@torch.no_grad()
def success_surface(
    model, g: Data, device: str = "cpu", dxs=GRID_DX, dys=GRID_DY
) -> dict | None:
    """Return the P(success) surface as the key attacker is swept over the offset grid.

    Returns ``None`` if the graph has no eligible key attacker. The dict holds the
    ``surface`` (``len(dys) x len(dxs)``), the grid axes, the current-position ``base``
    probability, the ``best`` probability, and the maximising ``best_dx``/``best_dy``.
    """
    key = key_attacker_index(g)
    if key is None:
        return None
    model.eval()
    graphs = [perturbed(g, key, float(dx), float(dy)) for dy in dys for dx in dxs]
    probs: list[np.ndarray] = []
    for b in DataLoader(graphs, batch_size=256):
        b = b.to(device)
        probs.append(torch.sigmoid(model(b)["success"]).cpu().numpy())
    surf = np.concatenate(probs).reshape(len(dys), len(dxs))
    bi = int(np.argmin(np.abs(dys)))
    bj = int(np.argmin(np.abs(dxs)))
    mi, mj = np.unravel_index(int(np.argmax(surf)), surf.shape)
    return {
        "surface": surf,
        "dxs": np.asarray(dxs, dtype=float),
        "dys": np.asarray(dys, dtype=float),
        "base": float(surf[bi, bj]),
        "best": float(surf[mi, mj]),
        "best_dx": float(dxs[mj]),
        "best_dy": float(dys[mi]),
        "key_index": key,
    }
