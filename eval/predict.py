"""Single-possession prediction demo → console report + pitch figure.

The "show it working" entry point: pick one attacking-third possession, run the
trained model on its freeze frame, and report — qualitatively — what the model
believes: P(success), xT progression, **where the key attacker will be in 1.5 s**
(the run head), and the top next-pass receiver candidates. Renders the freeze frame
with the predicted vs actual run target to ``results/figures/``.

Usage::

    python -m eval.predict --random                 # random held-out test possession
    python -m eval.predict --match 3835330 --possession 47
    python -m eval.predict --random --model transformer
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data.graphs import PITCH_LENGTH, PITCH_WIDTH, load_graph_cache
from models.gnn import GAT
from models.transformer import PlayerTransformer
from train.utils import build_splits

logger = logging.getLogger(__name__)
FIGURES_DIR = Path("results/figures")
CKPTS = {
    "gnn": Path("results/checkpoints/gnn.pt"),
    "transformer": Path("results/checkpoints/transformer.pt"),
}


def load_model(name: str) -> torch.nn.Module:
    """Load a trained checkpoint (``gnn`` or ``transformer``) in eval mode."""
    model = GAT() if name == "gnn" else PlayerTransformer()
    model.load_state_dict(torch.load(CKPTS[name], weights_only=True))
    model.eval()
    return model


def pick_graph(graphs: list, match_id: int | None, possession: int | None):
    """Find the requested graph, or a random test-set one when unspecified."""
    if match_id is not None:
        for g in graphs:
            if int(g.match_id) == match_id and (
                possession is None or int(g.possession) == possession
            ):
                return g
        raise SystemExit(f"no graph for match {match_id} possession {possession}")
    poss = pd.read_parquet("data/processed/possessions.parquet")
    test_ids = set(build_splits(poss, out_path=None)["test"])
    candidates = [g for g in graphs if int(g.match_id) in test_ids]
    return random.choice(candidates)


def report(g, out: dict[str, torch.Tensor]) -> dict:
    """Print the qualitative prediction report for one graph; return key numbers."""
    x = g.x.numpy()
    pts = x[:, 0:2] * [PITCH_LENGTH, PITCH_WIDTH]
    teammate, actor = x[:, 4] > 0.5, x[:, 5] > 0.5
    # The run head predicts the furthest-forward attacker (excl. carrier), §3 #3.
    cand = teammate & ~actor
    key_idx = int(np.where(cand)[0][np.argmax(pts[cand, 0])]) if cand.any() else None

    p_success = float(torch.sigmoid(out["success"]))
    xt = float(out["xt"])
    run_pred = out["run"].numpy().reshape(2) * [PITCH_LENGTH, PITCH_WIDTH]
    run_true = g.y_run.numpy().reshape(2) * [PITCH_LENGTH, PITCH_WIDTH]

    outcome = "SUCCESS" if float(g.y_success) else "no success"
    print(f"\nMatch {int(g.match_id)}, possession {int(g.possession)}")
    print(f"  P(success):            {p_success:.1%}   (actual outcome: {outcome})")
    print(f"  xT progression:        {xt:+.4f}  (actual: {float(g.y_xt):+.4f})")
    if key_idx is not None:
        print(f"  key attacker now at:   ({pts[key_idx, 0]:.1f}, {pts[key_idx, 1]:.1f}) m")
    print(f"  predicted in 1.5 s at: ({run_pred[0]:.1f}, {run_pred[1]:.1f}) m")
    print(f"  actually went to:      ({run_true[0]:.1f}, {run_true[1]:.1f}) m")
    print(f"  prediction error:      {np.linalg.norm(run_pred - run_true):.1f} m")

    if "receiver_probs" in out:
        probs = out["receiver_probs"].numpy().reshape(-1)[: len(pts)]
        top = np.argsort(probs)[::-1][:3]
        print("  top next-pass receivers (model):")
        for rank, i in enumerate(top, start=1):
            tag = " <- actual" if hasattr(g, "y_receiver") and float(g.y_receiver[i]) > 0.5 else ""
            print(
                f"    {rank}. player at ({pts[i, 0]:.1f}, {pts[i, 1]:.1f})  p={probs[i]:.2f}{tag}"
            )
    return {
        "pts": pts,
        "teammate": teammate,
        "actor": actor,
        "key_idx": key_idx,
        "run_pred": run_pred,
        "run_true": run_true,
        "p_success": p_success,
    }


def render(g, r: dict, model_name: str) -> Path:
    """Save a freeze-frame figure with the predicted vs actual run target."""
    from mplsoccer import Pitch

    pitch = Pitch(pitch_type="statsbomb", pitch_color="#1a3d1a", line_color="white")
    fig, ax = pitch.draw(figsize=(10, 7))
    pts, tm, ac = r["pts"], r["teammate"], r["actor"]
    pitch.scatter(pts[tm & ~ac, 0], pts[tm & ~ac, 1], ax=ax, c="#ff5555", s=180, label="attackers")
    pitch.scatter(pts[~tm, 0], pts[~tm, 1], ax=ax, c="#5599ff", s=180, label="defenders")
    if ac.any():
        pitch.scatter(
            pts[ac, 0], pts[ac, 1], ax=ax, c="yellow", s=260, marker="*", label="ball carrier"
        )
    if r["key_idx"] is not None:
        k = pts[r["key_idx"]]
        pitch.arrows(
            k[0],
            k[1],
            r["run_pred"][0],
            r["run_pred"][1],
            ax=ax,
            color="white",
            width=2,
            label="predicted run",
        )
        pitch.scatter(
            [r["run_true"][0]],
            [r["run_true"][1]],
            ax=ax,
            c="lime",
            s=220,
            marker="X",
            label="actual position +1.5 s",
        )
    outcome = "success" if float(g.y_success) else "no success"
    ax.set_title(
        f"{model_name.upper()} — match {int(g.match_id)} poss {int(g.possession)} — "
        f"P(success) = {r['p_success']:.0%} (actual: {outcome})",
        color="black",
    )
    ax.legend(loc="upper left", fontsize=8)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / f"predict_{int(g.match_id)}_{int(g.possession)}_{model_name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nFigure -> {out}")
    return out


def main() -> None:
    """Entry point: pick a possession, predict, report, render."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--match", type=int, default=None)
    parser.add_argument("--possession", type=int, default=None)
    parser.add_argument("--random", action="store_true", help="random held-out test possession")
    parser.add_argument("--model", choices=["gnn", "transformer"], default="gnn")
    parser.add_argument("--seed", type=int, default=None, help="seed for --random")
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    if args.seed is not None:
        random.seed(args.seed)
    graphs = load_graph_cache()
    g = pick_graph(graphs, args.match, args.possession)
    model = load_model(args.model)
    with torch.no_grad():
        out = model(g)
    r = report(g, out)
    if not args.no_figure:
        render(g, r, args.model)


if __name__ == "__main__":
    main()
