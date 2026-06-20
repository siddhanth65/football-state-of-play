"""Temporal model over a sequence of freeze-frame graphs (REPORT §6, future work).

The single-frame models cannot see motion or the build-up of an attack. ``TemporalGAT``
encodes each build-up frame with the GAT encoder, then a GRU over the sequence of
graph-level embeddings predicts possession **success** and **threat** from the *evolution*
of the state of play (momentum, a developing 3v2) rather than a frozen instant.

Each frame is an independent permutation-invariant graph, so the model needs **no player
IDs and no cross-frame tracking** — it reasons over the sequence of whole-frame embeddings
(see the player-ID discussion in REPORT). Consumes :mod:`data.sequences`.
"""

from __future__ import annotations

import torch
from torch import nn

from models.gnn import GAT


class TemporalGAT(nn.Module):
    """GAT frame-encoder + GRU over the build-up sequence -> success / xT heads."""

    def __init__(
        self, hidden: int = 64, heads: int = 4, layers: int = 3, dropout: float = 0.1
    ) -> None:
        super().__init__()
        # Reuse the GAT as a per-frame encoder; ``embed`` returns its graph-level vector.
        self.encoder = GAT(
            hidden=hidden, heads=heads, layers=layers, dropout=dropout, with_receiver=False
        )
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.success = nn.Linear(hidden, 1)
        self.xt = nn.Linear(hidden, 1)

    def forward(self, frames, seq_lengths: torch.Tensor) -> dict[str, torch.Tensor]:
        """Encode each frame, then run the GRU over the per-possession sequence.

        ``frames`` is a PyG ``Batch`` of all frames of all sequences (flattened);
        ``seq_lengths`` gives the per-sequence frame counts (newest frame last).
        """
        emb = self.encoder.embed(frames)  # (total_frames, hidden)
        b = seq_lengths.size(0)
        k = int(seq_lengths.max().item())
        padded = emb.new_zeros(b, k, emb.size(1))
        idx = 0
        for i in range(b):
            length = int(seq_lengths[i].item())
            padded[i, :length] = emb[idx : idx + length]
            idx += length
        packed = nn.utils.rnn.pack_padded_sequence(
            padded, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)  # h: (1, B, hidden) — last hidden state
        g = h[-1]
        return {"success": self.success(g).squeeze(-1), "xt": self.xt(g).squeeze(-1)}


class TemporalTransformer(nn.Module):
    """GAT frame-encoder + Transformer over the build-up sequence -> success / xT heads.

    A more expressive temporal aggregator than :class:`TemporalGAT`'s GRU: self-attention
    over the K frame embeddings (with a learned ``[CLS]`` readout token and learned
    positional encodings) lets the model weight *any* build-up frame, not just integrate
    left-to-right. Same 64-d GAT frame encoder as ``TemporalGAT``, so the only difference
    is the aggregator — the clean test of whether attention over the build-up beats a GRU
    (and whether either beats the single trigger frame).
    """

    MAX_FRAMES = 16  # learned positional table size (>= data.sequences.SEQUENCE_LEN + CLS)

    def __init__(
        self,
        hidden: int = 64,
        heads: int = 4,
        layers: int = 3,
        dropout: float = 0.1,
        t_layers: int = 2,
        t_heads: int = 4,
    ) -> None:
        super().__init__()
        self.encoder = GAT(
            hidden=hidden, heads=heads, layers=layers, dropout=dropout, with_receiver=False
        )
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden))
        self.pos = nn.Parameter(torch.zeros(1, self.MAX_FRAMES, hidden))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=t_heads,
            dim_feedforward=hidden * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.tfm = nn.TransformerEncoder(enc_layer, num_layers=t_layers)
        self.success = nn.Linear(hidden, 1)
        self.xt = nn.Linear(hidden, 1)

    def forward(self, frames, seq_lengths: torch.Tensor) -> dict[str, torch.Tensor]:
        """Encode each frame, then attend over the sequence via a ``[CLS]`` readout."""
        emb = self.encoder.embed(frames)  # (total_frames, hidden)
        b = seq_lengths.size(0)
        k = int(seq_lengths.max().item())
        padded = emb.new_zeros(b, k, emb.size(1))
        pad_mask = torch.ones(b, k, dtype=torch.bool, device=emb.device)  # True = ignore
        idx = 0
        for i in range(b):
            length = int(seq_lengths[i].item())
            padded[i, :length] = emb[idx : idx + length]
            pad_mask[i, :length] = False
            idx += length
        cls = self.cls.expand(b, -1, -1)
        seq = torch.cat([cls, padded], dim=1) + self.pos[:, : k + 1]
        cls_mask = torch.zeros(b, 1, dtype=torch.bool, device=emb.device)
        full_mask = torch.cat([cls_mask, pad_mask], dim=1)
        h = self.tfm(seq, src_key_padding_mask=full_mask)
        g = h[:, 0]  # CLS readout
        return {"success": self.success(g).squeeze(-1), "xt": self.xt(g).squeeze(-1)}
