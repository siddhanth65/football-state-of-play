"""Train the temporal sequence model and compare it to the single-frame GAT (REPORT §6).

The scientific question: does seeing the **build-up** (a short sequence of freeze-frames)
predict possession success better than a single trigger frame? Same match-level split,
optimiser, and metrics as the single-frame models, so the comparison is apples-to-apples.

Run ``python -m train.train_temporal`` (``--rebuild`` to (re)build the sequence cache).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from data.sequences import SEQUENCE_CACHE_PATH, build_possession_sequences, load_sequence_cache
from eval.metrics import success_metrics, xt_metrics
from models.temporal import TemporalGAT, TemporalTransformer
from train.utils import build_splits, set_seed

logger = logging.getLogger(__name__)
CKPT = Path("results/checkpoints/temporal.pt")
RESULT_PATH = Path("results/temporal.csv")


def collate(batch: list[dict]):
    """Flatten a batch of sequence records into one PyG Batch + per-sequence lengths."""
    frames, lengths, ys, yx = [], [], [], []
    for r in batch:
        frames.extend(r["frames"])
        lengths.append(len(r["frames"]))
        ys.append(r["y_success"])
        yx.append(r["y_xt"])
    return (
        Batch.from_data_list(frames),
        torch.tensor(lengths),
        torch.tensor(ys, dtype=torch.float),
        torch.tensor(yx, dtype=torch.float),
    )


def _subset(records: list[dict], match_ids: list[int]) -> list[dict]:
    keep = set(match_ids)
    return [r for r in records if r["match_id"] in keep]


@torch.no_grad()
def evaluate(model, loader, device: str) -> dict[str, float]:
    """Success + xT metrics on a loader."""
    model.eval()
    ps, ys, px, yx = [], [], [], []
    for frames, lengths, ysb, yxb in loader:
        out = model(frames.to(device), lengths.to(device))
        ps.append(torch.sigmoid(out["success"]).cpu().numpy())
        ys.append(ysb.numpy())
        px.append(out["xt"].cpu().numpy())
        yx.append(yxb.numpy())
    ps, ys = np.concatenate(ps), np.concatenate(ys)
    px, yx = np.concatenate(px), np.concatenate(yx)
    metrics = {f"success_{k}": v for k, v in success_metrics(ys, ps).items()}
    metrics.update({f"xt_{k}": v for k, v in xt_metrics(yx, px).items()})
    return metrics


def train_temporal(
    records: list[dict],
    splits: dict[str, list[int]],
    epochs: int = 40,
    batch_size: int = 32,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    patience: int = 7,
    device: str | None = None,
    seed: int = 42,
    model: nn.Module | None = None,
) -> tuple[nn.Module, dict[str, float]]:
    """Train a temporal model (default ``TemporalGAT``) and return ``(model, test_metrics)``."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)
    tr = DataLoader(
        _subset(records, splits["train"]), batch_size=batch_size, shuffle=True, collate_fn=collate
    )
    va = DataLoader(_subset(records, splits["val"]), batch_size=batch_size, collate_fn=collate)
    te = DataLoader(_subset(records, splits["test"]), batch_size=batch_size, collate_fn=collate)

    model = (model if model is not None else TemporalGAT()).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    bce, mse = nn.BCEWithLogitsLoss(), nn.MSELoss()
    best, best_state, stale = float("inf"), None, 0

    for epoch in range(epochs):
        model.train()
        for frames, lengths, ys, yx in tr:
            opt.zero_grad()
            out = model(frames.to(device), lengths.to(device))
            loss = bce(out["success"], ys.to(device)) + mse(out["xt"], yx.to(device))
            loss.backward()
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            vl, nb = 0.0, 0
            for frames, lengths, ys, yx in va:
                out = model(frames.to(device), lengths.to(device))
                vl += (bce(out["success"], ys.to(device)) + mse(out["xt"], yx.to(device))).item()
                nb += 1
            vl /= max(nb, 1)
        logger.info("epoch %d val_loss=%.4f", epoch, vl)
        if vl < best:
            best, best_state, stale = vl, {k: v.cpu() for k, v in model.state_dict().items()}, 0
        else:
            stale += 1
            if stale >= patience:
                logger.info("early stopping at epoch %d", epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, evaluate(model, te, device)


def main() -> None:
    """Build/load sequences, train the temporal model, save checkpoint + metrics."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--rebuild", action="store_true", help="(re)build the sequence cache")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    poss = pd.read_parquet("data/processed/possessions.parquet")
    if args.rebuild or not SEQUENCE_CACHE_PATH.exists():
        records = build_possession_sequences(poss)
    else:
        records = load_sequence_cache()
    splits = build_splits(poss)

    # Train both temporal aggregators on the same sequence cache + split: GRU vs
    # self-attention over the build-up frames. Same 64-d GAT frame encoder, so the
    # comparison isolates the aggregator (and both against the single-frame models).
    variants = {"TemporalGAT": TemporalGAT, "TemporalTransformer": TemporalTransformer}
    rows: list[dict] = []
    CKPT.parent.mkdir(parents=True, exist_ok=True)
    for name, factory in variants.items():
        set_seed(42)
        model, metrics = train_temporal(records, splits, epochs=args.epochs, model=factory())
        torch.save(model.state_dict(), CKPT.with_name(f"{name.lower()}.pt"))
        rows.append({"model": name, **metrics})
        print(f"\n{name} test metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
    pd.DataFrame(rows).to_csv(RESULT_PATH, index=False)
    print("\n  (single-frame GAT success AUC ~0.71 / transformer ~0.744 for comparison)")
    print(f"\nWrote {RESULT_PATH}")


if __name__ == "__main__":
    main()
