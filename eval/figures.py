"""Evaluation figures + run-head analysis (CLAUDE.md §7.1; PROJECT_REVIEW §8.5).

Produces, from the saved checkpoints and the held-out test set:

- ``results/figures/reliability_success.png`` — reliability diagram for the
  success head (GAT + transformer), the §7.1 calibration artefact.
- ``results/figures/xt_scatter.png`` — predicted vs observed xT progression.
- ``results/figures/run_hitrate_curve.png`` — the honest run-head story:
  hit-rate within r metres for r ∈ [1, 15], model vs the no-move baseline.
  Attackers barely move in 1.5 s, so no-move wins at tight radii; the curve
  shows where (and whether) the learned prediction overtakes it.
- ``results/run_analysis.csv`` — the numbers behind the curve (RMSE, relative
  improvement, crossover radius).

Run ``python -m eval.figures`` (needs ``graphs.pt`` + both checkpoints).
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from data.graphs import PITCH_LENGTH, PITCH_WIDTH, load_graph_cache
from models.baselines import run_baseline_nomove
from models.gnn import GAT
from models.transformer import PlayerTransformer
from train.utils import build_splits, split_graphs

logger = logging.getLogger(__name__)
FIGURES_DIR = Path("results/figures")
RUN_ANALYSIS_PATH = Path("results/run_analysis.csv")
SCALE = np.array([PITCH_LENGTH, PITCH_WIDTH])

ACCENT = "#10b981"  # emerald — single accent across all figures
INK = "#27272a"
MUTED = "#a1a1aa"


@torch.no_grad()
def collect(model: torch.nn.Module, graphs: list, device: str = "cpu") -> dict[str, np.ndarray]:
    """Run a model over graphs and return stacked predictions + targets."""
    model = model.to(device).eval()
    out: dict[str, list] = {
        "p_success": [],
        "y_success": [],
        "xt": [],
        "y_xt": [],
        "run": [],
        "y_run": [],
    }
    for batch in DataLoader(graphs, batch_size=128):
        batch = batch.to(device)
        pred = model(batch)
        out["p_success"].append(torch.sigmoid(pred["success"]).cpu().numpy())
        out["xt"].append(pred["xt"].cpu().numpy())
        out["run"].append(pred["run"].cpu().numpy())
        out["y_success"].append(batch.y_success.cpu().numpy())
        out["y_xt"].append(batch.y_xt.cpu().numpy())
        out["y_run"].append(batch.y_run.cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def load_models() -> dict[str, torch.nn.Module]:
    """Load both trained checkpoints."""
    gat = GAT()
    gat.load_state_dict(torch.load("results/checkpoints/gnn.pt", weights_only=True))
    tfm = PlayerTransformer()
    tfm.load_state_dict(torch.load("results/checkpoints/transformer.pt", weights_only=True))
    return {"GAT": gat, "Transformer": tfm}


def _style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.5)


def reliability_diagram(results: dict[str, dict], path: Path) -> None:
    """Reliability diagram (10 equal-width bins) for the success head."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1, label="perfect calibration")
    bins = np.linspace(0, 1, 11)
    for (name, r), color in zip(results.items(), [ACCENT, INK], strict=False):
        p, y = r["p_success"], r["y_success"]
        idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
        xs, ys = [], []
        for b in range(10):
            m = idx == b
            if m.sum() >= 20:
                xs.append(p[m].mean())
                ys.append(y[m].mean())
        ax.plot(xs, ys, marker="o", markersize=5, linewidth=1.5, color=color, label=name)
    ax.set_xlabel("Predicted P(success)")
    ax.set_ylabel("Observed success rate")
    ax.set_title("Success-head calibration (held-out test)")
    ax.legend(frameon=False)
    _style(ax)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def xt_scatter(results: dict[str, dict], path: Path) -> None:
    """Predicted vs observed xT progression scatter."""
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5), sharey=True)
    for ax, (name, r) in zip(np.atleast_1d(axes), results.items(), strict=False):
        ax.scatter(r["y_xt"], r["xt"], s=4, alpha=0.25, color=ACCENT, edgecolors="none")
        lim = (-0.3, 0.4)
        ax.plot(lim, lim, color=MUTED, linestyle="--", linewidth=1)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("Observed xT progression")
        ax.set_title(name)
        _style(ax)
    np.atleast_1d(axes)[0].set_ylabel("Predicted xT progression")
    fig.suptitle("xT head: predicted vs observed (held-out test)")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_hitrate_analysis(
    results: dict[str, dict], nomove: np.ndarray, y_run: np.ndarray, fig_path: Path, csv_path: Path
) -> pd.DataFrame:
    """Hit-rate-within-r curves (model vs no-move) + the summary table."""
    radii = np.arange(1.0, 15.5, 0.5)
    err_nomove = np.linalg.norm((nomove - y_run) * SCALE, axis=1)
    curves = {"no-move baseline": err_nomove}
    for name, r in results.items():
        curves[name] = np.linalg.norm((r["run"] - y_run) * SCALE, axis=1)

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"no-move baseline": MUTED, "GAT": INK, "Transformer": ACCENT}
    rows = []
    for name, err in curves.items():
        hit = [(err <= rad).mean() for rad in radii]
        ax.plot(radii, hit, linewidth=1.8, color=colors.get(name, INK), label=name)
        rows.append(
            {
                "model": name,
                "rmse_m": float(np.sqrt((err**2).mean())),
                "median_err_m": float(np.median(err)),
                "hit_3m": float((err <= 3).mean()),
                "hit_5m": float((err <= 5).mean()),
                "hit_10m": float((err <= 10).mean()),
            }
        )
    # Crossover: first radius where the best model overtakes no-move.
    best = min((r for r in rows if r["model"] != "no-move baseline"), key=lambda r: r["rmse_m"])
    nomove_row = next(r for r in rows if r["model"] == "no-move baseline")
    cross = next(
        (
            float(rad)
            for rad in radii
            if (curves[best["model"]] <= rad).mean() > (err_nomove <= rad).mean()
        ),
        None,
    )
    ax.axvline(cross, color=ACCENT, linestyle=":", linewidth=1) if cross else None
    ax.set_xlabel("Radius r (metres)")
    ax.set_ylabel("Hit-rate within r")
    ax.set_title("Run head: hit-rate vs no-move (held-out test, t + 1.5 s)")
    ax.legend(frameon=False)
    _style(ax)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    table = pd.DataFrame(rows)
    table["rel_rmse_improvement"] = 1 - table["rmse_m"] / nomove_row["rmse_m"]
    table.to_csv(csv_path, index=False)
    if cross:
        logger.info("run head: %s overtakes no-move beyond r = %.1f m", best["model"], cross)
    return table


def main() -> None:
    """Generate all evaluation figures and the run-analysis table."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    poss = pd.read_parquet("data/processed/possessions.parquet")
    graphs = load_graph_cache()
    test = split_graphs(graphs, build_splits(poss)["test"])
    logger.info("figures: %d test graphs", len(test))

    results = {name: collect(model, test) for name, model in load_models().items()}
    nomove = run_baseline_nomove(test)
    y_run = results["GAT"]["y_run"]

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    reliability_diagram(results, FIGURES_DIR / "reliability_success.png")
    xt_scatter(results, FIGURES_DIR / "xt_scatter.png")
    table = run_hitrate_analysis(
        results, nomove, y_run, FIGURES_DIR / "run_hitrate_curve.png", RUN_ANALYSIS_PATH
    )
    print("\n" + table.round(4).to_string(index=False))
    print(f"\nFigures -> {FIGURES_DIR}\nTable   -> {RUN_ANALYSIS_PATH}")


if __name__ == "__main__":
    main()
