"""Transformer over player tokens — the secondary model (CLAUDE.md §5.3).

Each visible player is one token: the node features from :mod:`data.graphs`
(positions, velocities, role flags, distances/angles — so the "positional encoding"
is the projected ``(x, y, vx, vy, ...)`` itself) projected by a 2-layer MLP into the
model width. A learned ``[CLS]`` token is prepended; its final embedding, concatenated
with the graph-level globals, is the readout the shared multi-task heads consume.

Consumes the same ``torch_geometric`` batches as the GAT (``to_dense_batch`` does the
padding), so it plugs into the identical training loop and metrics.
"""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.utils import to_dense_batch

from data.graphs import N_GLOBAL_FEATURES, N_NODE_FEATURES
from models.heads import MultiTaskHeads


class PlayerTransformer(nn.Module):
    """3-layer transformer encoder over player tokens with a CLS readout."""

    def __init__(
        self,
        node_dim: int = N_NODE_FEATURES,
        global_dim: int = N_GLOBAL_FEATURES,
        d_model: int = 128,
        heads: int = 4,
        layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(node_dim, d_model), nn.ReLU(), nn.Linear(d_model, d_model)
        )
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model,
            heads,
            dim_feedforward=2 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.proj = nn.Sequential(
            nn.Linear(d_model + global_dim, d_model), nn.ReLU(), nn.Dropout(dropout)
        )
        self.heads = MultiTaskHeads(d_model)

    def embed_graph(self, data) -> torch.Tensor:
        """Compute the CLS-token graph embedding for a (possibly batched) ``Data``."""
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)
        dense, mask = to_dense_batch(data.x, batch)  # (B, N_max, F), (B, N_max)
        tokens = self.embed(dense)
        b = tokens.size(0)
        seq = torch.cat([self.cls.expand(b, -1, -1), tokens], dim=1)
        pad = torch.cat([torch.ones(b, 1, dtype=torch.bool, device=mask.device), mask], dim=1)
        h = self.encoder(seq, src_key_padding_mask=~pad)
        return self.proj(torch.cat([h[:, 0], data.u], dim=-1))

    def forward(self, data) -> dict[str, torch.Tensor]:
        """Return ``{"success", "xt", "run"}`` head outputs."""
        return self.heads(self.embed_graph(data))
