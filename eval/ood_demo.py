"""Out-of-distribution inference hook: run the trained GAT on an arbitrary freeze-frame.

The Phase-1 broadcast demo is a two-stage pipeline: (1) YOLO player detection + homography to
top-down pitch coordinates on a video clip, then (2) run this model on the resulting
freeze-frame. Stage (1) is the separate Phase-1 CV stack (it needs the video clip and that
codebase, outside this repo); **stage (2) lives here**: given a list of player coordinates +
team/actor/keeper flags (whatever a CV pipeline or a hand-built frame produces), build the same
graph the training data uses and return the model's predictions. This makes the OOD demo a
plug-in: supply coordinates, get success / Dynamic-xT / receiver / presser predictions.

Run ``python -m eval.ood_demo`` for a synthetic example.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from data.graphs import build_data
from data.possessions import COUNTER_PATTERN
from models.gnn import GAT

CKPT = Path("results/checkpoints/gnn.pt")


def _frame(players: list[dict]):
    import pandas as pd

    rows = [
        {
            "location": [float(p["x"]), float(p["y"])],
            "teammate": bool(p.get("teammate", True)),
            "actor": bool(p.get("actor", False)),
            "keeper": bool(p.get("keeper", False)),
        }
        for p in players
    ]
    return pd.DataFrame(rows)


@torch.no_grad()
def predict_from_freeze_frame(
    players: list[dict],
    play_pattern: str = "Regular Play",
    trigger_time_s: float = 600.0,
    ckpt: Path = CKPT,
) -> dict[str, float]:
    """Run the trained GAT on a single freeze-frame given as player coordinates.

    Args:
        players: One dict per visible player: ``{x, y, teammate, actor, keeper}`` in
            StatsBomb pitch units (120 x 80, attacking left -> right).
        play_pattern: The possession's play pattern (graph-level context).
        trigger_time_s: Period-relative seconds (graph-level context).
        ckpt: Trained GAT checkpoint.

    Returns:
        ``{success, dynamic_xt, top_receiver_prob, top_presser_prob}``.
    """
    frame = _frame(players)
    actor = frame[frame["actor"].astype(bool)]
    ball = (
        [float(actor.iloc[0]["location"][0]), float(actor.iloc[0]["location"][1])]
        if len(actor)
        else [float(frame["location"].map(lambda c: c[0]).mean()), 40.0]
    )
    row = SimpleNamespace(
        trigger_x=ball[0],
        trigger_y=ball[1],
        play_pattern=play_pattern,
        from_counter=(play_pattern == COUNTER_PATTERN),
        trigger_time_s=trigger_time_s,
    )
    data = build_data(frame, row)  # no labels -> inference-only graph
    model = GAT()
    model.load_state_dict(torch.load(ckpt, weights_only=True))
    model.eval()
    out = model(data)
    recv = out.get("receiver_probs")
    press = out.get("presser_probs")
    return {
        "success": float(torch.sigmoid(out["success"])),
        "dynamic_xt": float(out["dxt"]),
        "top_receiver_prob": float(recv.max()) if recv is not None else float("nan"),
        "top_presser_prob": float(press.max()) if press is not None else float("nan"),
    }


def main() -> None:
    """Synthetic 3-attacker-vs-2-defender example through the OOD hook."""
    players = [
        {"x": 92, "y": 40, "teammate": True, "actor": True},
        {"x": 104, "y": 28, "teammate": True},
        {"x": 106, "y": 52, "teammate": True},
        {"x": 100, "y": 36, "teammate": False},
        {"x": 110, "y": 44, "teammate": False},
        {"x": 119, "y": 40, "teammate": False, "keeper": True},
    ]
    if not CKPT.exists():
        print(f"No checkpoint at {CKPT}; train the GAT first (make gnn).")
        return
    print(predict_from_freeze_frame(players))


if __name__ == "__main__":
    main()
