"""Shared prediction heads for the GNN and transformer (CLAUDE.md §5.4).

Three linear heads on a graph-level embedding:
- ``success`` — 1 logit (BCEWithLogits target).
- ``xt`` — 1 value (MSE target: xT progression).
- ``run`` — 2 values (MSE target: normalised run-target ``(x, y)``).

These are the three shared heads. The ``H_receiver`` node head and the two-stage
success decomposition (RESEARCH_INTEGRATION §3.1) are **implemented on the node
embeddings inside** :class:`models.gnn.GAT` (they need per-node outputs, which a
graph-level head cannot produce); the transformer uses only the three heads here.
"""

from __future__ import annotations

import torch
from torch import nn


class MultiTaskHeads(nn.Module):
    """Linear success / xT / run heads over a shared embedding."""

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.success = nn.Linear(in_dim, 1)
        self.xt = nn.Linear(in_dim, 1)
        self.run = nn.Linear(in_dim, 2)

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return ``{"success": [B], "xt": [B], "run": [B, 2]}``."""
        return {
            "success": self.success(h).squeeze(-1),
            "xt": self.xt(h).squeeze(-1),
            "run": self.run(h),
        }
