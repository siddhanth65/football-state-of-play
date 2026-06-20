"""Graph Attention Network — the primary model (CLAUDE.md §5.2).

A GAT over the player graph from :mod:`data.graphs`: 3 GATv2 message-passing layers
(attention-weighted neighbour mixing) → mean+max pooling → concatenate the graph-level
globals → a small projection → the shared multi-task heads.

GATv2Conv consumes the edge features (``edge_dim``); ``concat=False`` averages the
attention heads so the hidden width stays constant across layers.

The adopted ``H_receiver`` head (RESEARCH_INTEGRATION §3.1; Wang et al. 2024 TacticAI,
Stöckl et al. 2021) adds two node-level heads on top of the node embeddings:
a receiver logit (softmax over the feasible teammates = P(next receiver)) and a
conditional success logit, combined into the **two-stage success decomposition**
``P(success) = Σ_i P(receiver = i) · P(success | receiver = i)``.

Reflection augmentation (RESEARCH_INTEGRATION §3.2) is planned but **not yet
implemented** — there is no pitch-reflection transform in the data or training code
(see REPORT §6 "Future work"). The module itself is reflection-agnostic.
"""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GATv2Conv, global_max_pool, global_mean_pool
from torch_geometric.utils import to_dense_batch

from data.graphs import N_EDGE_FEATURES, N_GLOBAL_FEATURES, N_NODE_FEATURES
from models.heads import MultiTaskHeads


class GAT(nn.Module):
    """Graph attention network with shared success / xT / run (+ receiver) heads."""

    def __init__(
        self,
        node_dim: int = N_NODE_FEATURES,
        edge_dim: int = N_EDGE_FEATURES,
        global_dim: int = N_GLOBAL_FEATURES,
        hidden: int = 128,  # sweep-selected (results/sweep.csv): 128 > 64 on val success AUC
        heads: int = 4,
        layers: int = 3,
        dropout: float = 0.1,
        with_receiver: bool = True,
        with_defense: bool = True,
        use_edge_attr: bool = True,
        use_globals: bool = True,
    ) -> None:
        super().__init__()
        self.use_edge_attr = use_edge_attr
        self.use_globals = use_globals
        self.convs = nn.ModuleList()
        in_dim = node_dim
        for _ in range(layers):
            self.convs.append(
                GATv2Conv(
                    in_dim,
                    hidden,
                    heads=heads,
                    concat=False,
                    edge_dim=edge_dim if use_edge_attr else None,
                    dropout=dropout,
                )
            )
            in_dim = hidden
        self.act = nn.ReLU()
        proj_in = hidden * 2 + (global_dim if use_globals else 0)
        self.proj = nn.Sequential(nn.Linear(proj_in, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.heads = MultiTaskHeads(hidden)
        # Auxiliary graph-level heads (all off the shared 'state of play' embedding; additive,
        # GAT-only). defsuccess = P(defence wins the ball back); defline = offside-line
        # displacement; dxt = Dynamic-xT (OBSO) — the configuration threat shown learnable in
        # eval/dxt.py, now a first-class head.
        self.defsuccess_head = nn.Linear(hidden, 1)
        self.defline_head = nn.Linear(hidden, 1)
        self.dxt_head = nn.Linear(hidden, 1)
        self.with_receiver = with_receiver
        self.with_defense = with_defense
        # Node-level heads over [raw node features ‖ node embedding ‖ graph embedding].
        # The raw-feature skip connection is deliberate: 3 rounds of attention on a
        # fully-connected graph over-smooth the node embeddings, so the smoothed `node_h`
        # alone cannot tell players apart (the receiver head collapsed below chance). The
        # raw features (position, distance-to-ball) restore per-node discriminability.
        node_head_in = node_dim + hidden * 2
        if with_receiver:
            self.receiver_head = nn.Linear(node_head_in, 1)
            self.cond_success_head = nn.Linear(node_head_in, 1)
            # xPass / disruption: per-teammate pass-completion probability (1 − xPass = how
            # closed the lane is). Same node-level skip input as the receiver head.
            self.xpass_head = nn.Linear(node_head_in, 1)
        if with_defense:
            # The defensive mirror of the receiver head: a softmax over the *defenders*
            # = P(this defender contests the next ball). Same skip-connection input.
            self.defense_head = nn.Linear(node_head_in, 1)

    def _encode(self, data) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(node_embeddings, graph_embedding, batch_vector)``."""
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        for conv in self.convs:
            x = self.act(conv(x, edge_index, edge_attr if self.use_edge_attr else None))
        pooled = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=-1)
        if self.use_globals:
            pooled = torch.cat([pooled, data.u], dim=-1)
        return x, self.proj(pooled), batch

    def embed(self, data) -> torch.Tensor:
        """Compute the graph-level embedding (the 'state of play')."""
        return self._encode(data)[1]

    @torch.no_grad()
    def node_attention(self, data) -> torch.Tensor:
        """Per-node importance from the GATv2 attention weights (explainability).

        Sums the (head-averaged) incoming attention each player receives across the
        message-passing layers, normalised to sum to 1 — "how much the model attends to
        this player". A single-graph ``Data`` (no batching). Returns a length-N tensor.
        """
        self.eval()
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        imp = torch.zeros(x.size(0), device=x.device)
        for conv in self.convs:
            x, (ei, alpha) = conv(
                x,
                edge_index,
                edge_attr if self.use_edge_attr else None,
                return_attention_weights=True,
            )
            x = self.act(x)
            imp.index_add_(0, ei[1], alpha.mean(dim=1))  # incoming attention per destination
        total = imp.sum()
        return imp / total if float(total) > 0 else imp

    def forward(self, data) -> dict[str, torch.Tensor]:
        """Return the head outputs for a (possibly batched) ``Data``.

        Always contains ``success`` / ``xt`` / ``run``. With the receiver head it
        also contains ``receiver_probs`` (``(B, N_max)`` dense softmax over each
        graph's feasible receivers), ``node_mask`` (the dense validity mask) and
        ``success_two`` (the two-stage success *probability*, not a logit).
        """
        node_h, g, batch = self._encode(data)
        out = self.heads(g)
        # Auxiliary graph-level heads (always present, GAT-only).
        out["defsuccess"] = self.defsuccess_head(g).squeeze(-1)
        out["defline"] = self.defline_head(g).squeeze(-1)
        out["dxt"] = self.dxt_head(g).squeeze(-1)
        node_in = None
        if self.with_receiver or self.with_defense:
            node_in = torch.cat([data.x, node_h, g[batch]], dim=-1)
        if self.with_receiver:
            r_logit = self.receiver_head(node_in).squeeze(-1)
            s_node = torch.sigmoid(self.cond_success_head(node_in).squeeze(-1))
            # Feasible receivers: teammates excluding the carrier (input flags).
            feasible = (data.x[:, 4] > 0.5) & (data.x[:, 5] < 0.5)
            r_logit = r_logit.masked_fill(~feasible, float("-inf"))
            r_dense, node_mask = to_dense_batch(r_logit, batch, fill_value=float("-inf"))
            p_recv = torch.softmax(r_dense, dim=1)
            p_recv = torch.nan_to_num(p_recv, nan=0.0)  # graphs with no feasible node
            s_dense, _ = to_dense_batch(s_node, batch)
            out["receiver_probs"] = p_recv
            out["node_mask"] = node_mask
            out["success_two"] = (p_recv * s_dense).sum(dim=1).clamp(1e-6, 1 - 1e-6)
            # xPass / disruption: per-teammate completion probability (dense, masked to
            # feasible teammates) — the lane-openness map.
            xp = torch.sigmoid(self.xpass_head(node_in).squeeze(-1))
            xp = xp.masked_fill(~feasible, 0.0)
            out["xpass_dense"], _ = to_dense_batch(xp, batch)
        if self.with_defense:
            d_logit = self.defense_head(node_in).squeeze(-1)
            # Feasible pressers: the defending team (non-teammates), keeper included.
            feasible_def = data.x[:, 4] < 0.5
            d_logit = d_logit.masked_fill(~feasible_def, float("-inf"))
            d_dense, def_mask = to_dense_batch(d_logit, batch, fill_value=float("-inf"))
            p_press = torch.nan_to_num(torch.softmax(d_dense, dim=1), nan=0.0)
            out["presser_probs"] = p_press
            out["presser_mask"] = def_mask
        return out
