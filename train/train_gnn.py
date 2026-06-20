"""Train the GAT on the player graphs (CLAUDE.md §5–§7).

Loads the cached graphs, splits by match, trains the multi-task GAT with AdamW +
cosine annealing and early stopping, and reports per-head metrics on the held-out test
set. The receiver head (RESEARCH_INTEGRATION §3.1) is trained alongside by default,
adding two-stage success metrics (``success2_*``) and receiver top-k accuracy.

Run ``python -m train.train_gnn`` (full) or ``--smoke`` for a quick check.
``--run-label alt`` is the §3 #3 ablation: train/evaluate the run head on the
next-pass-receiver location instead of the nearest-neighbour-matched attacker.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch

from data.graphs import (
    GRAPH_CACHE_PATH,
    PITCH_LENGTH,
    build_graph_cache,
    load_graph_cache,
    reflect_graph,
)
from eval.metrics import run_metrics, success_metrics, xt_metrics
from models.gnn import GAT
from train.losses import multitask_loss, rebalance
from train.utils import SEED, build_splits, load_config, set_seed, split_graphs

logger = logging.getLogger(__name__)
CKPT_DIR = Path("results/checkpoints")


@torch.no_grad()
def evaluate(model, loader: DataLoader, device: str) -> dict[str, float]:
    """Collect predictions over a loader and compute per-head metrics."""
    model.eval()
    keys = ("success", "xt", "run")
    ps: dict[str, list] = {k: [] for k in keys}
    ys: dict[str, list] = {k: [] for k in keys}
    p_two: list[np.ndarray] = []
    recv_hits1: list[np.ndarray] = []
    recv_hits3: list[np.ndarray] = []
    press_hits1: list[np.ndarray] = []
    press_hits3: list[np.ndarray] = []
    aux: dict[str, list[np.ndarray]] = {k: [] for k in (
        "defs_p", "defs_y", "dxt_p", "dxt_y", "dl_p", "dl_y", "dl_m", "xp_p", "xp_y"
    )}
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        if "defsuccess" in out and hasattr(batch, "y_defsuccess"):
            aux["defs_p"].append(torch.sigmoid(out["defsuccess"]).cpu().numpy())
            aux["defs_y"].append(batch.y_defsuccess.cpu().numpy())
        if "dxt" in out and hasattr(batch, "y_dxt"):
            aux["dxt_p"].append(out["dxt"].cpu().numpy())
            aux["dxt_y"].append(batch.y_dxt.cpu().numpy())
        if "defline" in out and hasattr(batch, "y_defline"):
            aux["dl_p"].append(out["defline"].cpu().numpy())
            aux["dl_y"].append(batch.y_defline.cpu().numpy())
            aux["dl_m"].append((batch.has_defline.view(-1) > 0.5).cpu().numpy())
        if "xpass_dense" in out and hasattr(batch, "y_xpass"):
            yr_dense, _ = to_dense_batch(batch.y_receiver, batch.batch)
            ti = yr_dense.argmax(dim=1, keepdim=True)
            xpt = out["xpass_dense"].gather(1, ti).squeeze(1).cpu().numpy()
            xm = (batch.has_xpass.view(-1) > 0.5).cpu().numpy()
            aux["xp_p"].append(xpt[xm])
            aux["xp_y"].append(batch.y_xpass.cpu().numpy()[xm])
        ps["success"].append(torch.sigmoid(out["success"]).cpu().numpy())
        ps["xt"].append(out["xt"].cpu().numpy())
        ps["run"].append(out["run"].cpu().numpy())
        ys["success"].append(batch.y_success.cpu().numpy())
        ys["xt"].append(batch.y_xt.cpu().numpy())
        ys["run"].append(batch.y_run.cpu().numpy())
        if "success_two" in out and hasattr(batch, "y_receiver"):
            p_two.append(out["success_two"].cpu().numpy())
            y_dense, _ = to_dense_batch(batch.y_receiver, batch.batch)
            mask = (batch.has_receiver.view(-1) > 0.5).cpu().numpy()
            if mask.any():
                probs = out["receiver_probs"]
                true_idx = y_dense.argmax(dim=1)
                top3 = probs.topk(min(3, probs.size(1)), dim=1).indices
                hit1 = (probs.argmax(dim=1) == true_idx).cpu().numpy()
                hit3 = (top3 == true_idx.unsqueeze(1)).any(dim=1).cpu().numpy()
                recv_hits1.append(hit1[mask])
                recv_hits3.append(hit3[mask])
        if "presser_probs" in out and hasattr(batch, "y_presser"):
            yp_dense, _ = to_dense_batch(batch.y_presser, batch.batch)
            pmask = (batch.has_presser.view(-1) > 0.5).cpu().numpy()
            if pmask.any():
                pp = out["presser_probs"]
                ptrue = yp_dense.argmax(dim=1)
                ptop3 = pp.topk(min(3, pp.size(1)), dim=1).indices
                ph1 = (pp.argmax(dim=1) == ptrue).cpu().numpy()
                ph3 = (ptop3 == ptrue.unsqueeze(1)).any(dim=1).cpu().numpy()
                press_hits1.append(ph1[pmask])
                press_hits3.append(ph3[pmask])
    p = {k: np.concatenate(v) for k, v in ps.items()}
    y = {k: np.concatenate(v) for k, v in ys.items()}
    metrics = {
        **{f"success_{k}": v for k, v in success_metrics(y["success"], p["success"]).items()},
        **{f"xt_{k}": v for k, v in xt_metrics(y["xt"], p["xt"]).items()},
        **{f"run_{k}": v for k, v in run_metrics(y["run"], p["run"]).items()},
    }
    if p_two:
        two = np.concatenate(p_two)
        metrics.update({f"success2_{k}": v for k, v in success_metrics(y["success"], two).items()})
    if recv_hits1:
        metrics["receiver_top1"] = float(np.concatenate(recv_hits1).mean())
        metrics["receiver_top3"] = float(np.concatenate(recv_hits3).mean())
    if press_hits1:
        metrics["presser_top1"] = float(np.concatenate(press_hits1).mean())
        metrics["presser_top3"] = float(np.concatenate(press_hits3).mean())
    if aux["defs_p"]:
        dp, dy = np.concatenate(aux["defs_p"]), np.concatenate(aux["defs_y"])
        metrics.update({f"defsuccess_{k}": v for k, v in success_metrics(dy, dp).items()})
    if aux["dxt_p"]:
        dxp, dxy = np.concatenate(aux["dxt_p"]), np.concatenate(aux["dxt_y"])
        metrics.update({f"dxt_{k}": v for k, v in xt_metrics(dxy, dxp).items()})
    if aux["dl_p"]:
        dlp, dly = np.concatenate(aux["dl_p"]), np.concatenate(aux["dl_y"])
        dlm = np.concatenate(aux["dl_m"])
        if dlm.any():
            err = np.abs(dlp[dlm] - dly[dlm]) * PITCH_LENGTH
            metrics["defline_rmse_m"] = float(np.sqrt((err**2).mean()))
            metrics["defline_nomove_rmse_m"] = float(
                np.sqrt(((np.abs(dly[dlm]) * PITCH_LENGTH) ** 2).mean())
            )
    if aux["xp_p"]:
        xp, xy_ = np.concatenate(aux["xp_p"]), np.concatenate(aux["xp_y"])
        if xp.size:
            metrics.update({f"xpass_{k}": v for k, v in success_metrics(xy_, xp).items()})
    return metrics


def train(
    graphs: list,
    splits: dict[str, list[int]],
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    patience: int = 7,
    rebalance_weights: bool = False,
    with_receiver: bool = True,
    w_receiver: float = 1.0,
    two_stage: bool = True,
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    device: str | None = None,
    model: torch.nn.Module | None = None,
    seed: int = SEED,
    augment_reflect: bool = False,
) -> tuple[torch.nn.Module, dict[str, float]]:
    """Train a model (default: a fresh GAT) and return ``(model, test_metrics)``.

    ``model`` lets ``train_transformer`` reuse this loop with the same protocol;
    ``weights`` lets the ablation suite train single-task variants (e.g. ``(1, 0, 0)``);
    ``augment_reflect`` adds laterally-mirrored copies of the **train** split only.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)
    train_graphs = split_graphs(graphs, splits["train"])
    if augment_reflect:  # double the train set with the valid lateral pitch reflection
        train_graphs = train_graphs + [reflect_graph(g) for g in train_graphs]
    tr = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    va = DataLoader(split_graphs(graphs, splits["val"]), batch_size=batch_size)
    te = DataLoader(split_graphs(graphs, splits["test"]), batch_size=batch_size)

    model = (model if model is not None else GAT(with_receiver=with_receiver)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_val, best_state, stale = float("inf"), None, 0

    for epoch in range(epochs):
        model.train()
        last_components = {}
        for batch in tr:
            batch = batch.to(device)
            opt.zero_grad()
            loss, last_components = multitask_loss(
                model(batch), batch, weights, w_receiver=w_receiver, two_stage=two_stage
            )
            loss.backward()
            opt.step()
        sched.step()
        # CLAUDE.md §5.5's inverse-magnitude rebalance is OFF by default: with a BCE
        # success loss (~0.7) alongside tiny MSE losses it drives w_success → 0 and
        # starves the (primary) success head. Fixed (1,1,1) keeps success learning.
        if rebalance_weights and epoch == 0 and last_components:
            weights = rebalance(last_components)

        val_loss = sum(
            multitask_loss(
                model(b.to(device)),
                b.to(device),
                weights,
                w_receiver=w_receiver,
                two_stage=two_stage,
            )[0].item()
            for b in va
        ) / max(len(va), 1)
        logger.info(
            "epoch %d  val_loss=%.4f  weights=%s",
            epoch,
            val_loss,
            tuple(round(w, 2) for w in weights),
        )
        if val_loss < best_val:
            best_val, best_state, stale = (
                val_loss,
                {k: v.cpu() for k, v in model.state_dict().items()},
                0,
            )
        else:
            stale += 1
            if stale >= patience:
                logger.info("early stopping at epoch %d", epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, evaluate(model, te, device)


def use_alt_run_label(graphs: list) -> list:
    """§3 #3 ablation: keep graphs with a next-pass label and use it as the run target."""
    out = []
    for g in graphs:
        if float(g.has_run_alt) > 0.5:
            g.y_run = g.y_run_alt
            out.append(g)
    return out


def main() -> None:
    """Entry point: load graphs + splits, train, save checkpoint + print metrics.

    Hyperparameter defaults come from the YAML config (CLAUDE.md §6, one YAML per
    run); CLI flags override.
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="train/configs/gnn_default.yaml")
    cfg_path = pre.parse_known_args()[0].config
    cfg = load_config(cfg_path)
    tr_cfg = cfg.get("train", {})

    parser = argparse.ArgumentParser(parents=[pre])
    parser.add_argument("--epochs", type=int, default=tr_cfg.get("max_epochs", 50))
    parser.add_argument("--batch-size", type=int, default=tr_cfg.get("batch_size", 32))
    parser.add_argument("--smoke", action="store_true", help="2 epochs on a small subset")
    parser.add_argument("--limit", type=int, default=None, help="matches to build if cache absent")
    parser.add_argument("--no-receiver", action="store_true", help="disable the receiver head")
    parser.add_argument(
        "--run-label",
        choices=["nn", "alt"],
        default="nn",
        help="run target: nn = furthest-forward attacker (default), alt = next-pass receiver",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    poss = pd.read_parquet("data/processed/possessions.parquet")
    if GRAPH_CACHE_PATH.exists():
        graphs = load_graph_cache()
    else:
        logger.warning("graph cache missing — building (limit=%s)", args.limit or 10)
        graphs = build_graph_cache(poss, out_path=None, limit=args.limit or 10)
    splits = build_splits(poss)

    if args.run_label == "alt":
        graphs = use_alt_run_label(graphs)
        logger.info("run-label alt: %d graphs with a next-pass target", len(graphs))

    epochs = 2 if args.smoke else args.epochs
    if args.smoke:
        keep = set(splits["train"][:6] + splits["val"][:3] + splits["test"][:3])
        graphs = [g for g in graphs if int(g.match_id) in keep]

    model, metrics = train(
        graphs,
        splits,
        epochs=epochs,
        batch_size=args.batch_size,
        lr=float(tr_cfg.get("lr", 1e-4)),
        weight_decay=float(tr_cfg.get("weight_decay", 1e-4)),
        patience=int(tr_cfg.get("early_stopping_patience", 7)),
        rebalance_weights=bool(cfg.get("loss", {}).get("adaptive_reweight", False)),
        with_receiver=not args.no_receiver,
    )
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    # Suffix non-default run labels so ablation runs never clobber the primary ckpt.
    stem = "gnn_smoke" if args.smoke else ("gnn" if args.run_label == "nn" else "gnn_run_alt")
    ckpt = CKPT_DIR / f"{stem}.pt"
    torch.save(model.state_dict(), ckpt)
    print("\nTest metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"\nSaved checkpoint -> {ckpt}")

    if not args.smoke and args.run_label == "nn":  # consolidate next to the baselines
        table_path = Path("results/final_table.csv")
        base = Path("results/baselines.csv")
        frames = [pd.read_csv(base)] if base.exists() else []
        if table_path.exists():  # keep non-baseline rows (e.g. Transformer)
            old = pd.read_csv(table_path)
            frames.append(old[~old["model"].str.startswith("B") & (old["model"] != "GAT")])
        frames.append(pd.DataFrame([{"model": "GAT", **metrics}]))
        pd.concat(frames, ignore_index=True).to_csv(table_path, index=False)
        print(f"Wrote {table_path}")


if __name__ == "__main__":
    main()
