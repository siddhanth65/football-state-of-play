"""Multi-task loss for the success / xT / run (+ receiver) heads (CLAUDE.md §5.5).

``L = w_success·BCE + w_xt·MSE + w_run·MSE`` plus, when the model exposes the
``H_receiver`` outputs (RESEARCH_INTEGRATION §3.1), a receiver cross-entropy on the
graphs that have a next-pass label and a BCE on the two-stage success probability.

Weights start equal; :func:`rebalance` rescales them inversely to each task's loss
magnitude so all three contribute comparably (opt-in — see ``train_gnn``).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch


def multitask_loss(
    pred: dict[str, torch.Tensor],
    batch,
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    w_receiver: float = 1.0,
    two_stage: bool = True,
    w_defense: float = 1.0,
    w_aux: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Weighted sum of the head losses.

    Args:
        pred: Head outputs ``{"success", "xt", "run"}`` and optionally
            ``{"receiver_probs", "success_two"}``.
        batch: A ``Data``/``Batch`` carrying ``y_success`` / ``y_xt`` / ``y_run``
            (and ``y_receiver`` / ``has_receiver`` for the receiver terms).
        weights: ``(w_success, w_xt, w_run)``.
        w_receiver: Weight on the receiver CE (+ two-stage BCE) terms (0 disables).
        two_stage: Add the two-stage success BCE on ``success_two``. **The two-stage
            term shapes ``receiver_probs`` to *explain success*, which conflicts with the
            receiver CE and was found to push the receiver head below chance; disable to
            train the receiver head purely to identify the receiver.**
        w_defense: Weight on the presser cross-entropy (the defensive mirror of the
            receiver head; 0 disables). Applies on graphs with a ``y_presser`` label.
        w_aux: Weight on the auxiliary GAT heads (defensive-success, offside-line,
            Dynamic-xT, xPass/disruption); 0 disables them.

    Returns:
        ``(total_loss_tensor, {component: float})``.
    """
    w_s, w_x, w_r = weights
    l_s = F.binary_cross_entropy_with_logits(pred["success"], batch.y_success)
    l_x = F.mse_loss(pred["xt"], batch.y_xt)
    l_r = F.mse_loss(pred["run"], batch.y_run)
    total = w_s * l_s + w_x * l_x + w_r * l_r
    components = {"success": float(l_s), "xt": float(l_x), "run": float(l_r)}

    if w_receiver and "receiver_probs" in pred and hasattr(batch, "y_receiver"):
        bvec = getattr(batch, "batch", None)
        if bvec is None:
            bvec = torch.zeros(batch.y_receiver.size(0), dtype=torch.long)
        y_dense, _ = to_dense_batch(batch.y_receiver, bvec)
        mask = batch.has_receiver.view(-1) > 0.5
        if bool(mask.any()):
            p_true = (pred["receiver_probs"] * y_dense).sum(dim=1).clamp_min(1e-8)
            l_recv = -(torch.log(p_true)[mask]).mean()
            recv_term = l_recv
            components["receiver"] = float(l_recv)
            if two_stage:
                l_two = F.binary_cross_entropy(pred["success_two"], batch.y_success)
                recv_term = recv_term + l_two
                components["success_two"] = float(l_two)
            total = total + w_receiver * recv_term

    if w_defense and "presser_probs" in pred and hasattr(batch, "y_presser"):
        bvec = getattr(batch, "batch", None)
        if bvec is None:
            bvec = torch.zeros(batch.y_presser.size(0), dtype=torch.long)
        yp_dense, _ = to_dense_batch(batch.y_presser, bvec)
        pmask = batch.has_presser.view(-1) > 0.5
        if bool(pmask.any()):
            p_true = (pred["presser_probs"] * yp_dense).sum(dim=1).clamp_min(1e-8)
            l_press = -(torch.log(p_true)[pmask]).mean()
            components["presser"] = float(l_press)
            total = total + w_defense * l_press

    # Auxiliary heads (all GAT-only, additive): defensive-success, offside-line, Dynamic-xT,
    # and xPass/disruption. Each guarded by the model exposing it + the batch carrying its label.
    if w_aux and "defsuccess" in pred and hasattr(batch, "y_defsuccess"):
        l_ds = F.binary_cross_entropy_with_logits(pred["defsuccess"], batch.y_defsuccess)
        components["defsuccess"] = float(l_ds)
        total = total + w_aux * l_ds
    if w_aux and "defline" in pred and hasattr(batch, "y_defline"):
        m = batch.has_defline.view(-1) > 0.5
        if bool(m.any()):
            l_dl = F.mse_loss(pred["defline"][m], batch.y_defline[m])
            components["defline"] = float(l_dl)
            total = total + w_aux * l_dl
    if w_aux and "dxt" in pred and hasattr(batch, "y_dxt"):
        l_dx = F.mse_loss(pred["dxt"], batch.y_dxt)
        components["dxt"] = float(l_dx)
        total = total + w_aux * l_dx
    if w_aux and "xpass_dense" in pred and hasattr(batch, "y_xpass"):
        bvec = getattr(batch, "batch", None)
        if bvec is None:
            bvec = torch.zeros(batch.y_receiver.size(0), dtype=torch.long)
        yr_dense, _ = to_dense_batch(batch.y_receiver, bvec)
        true_idx = yr_dense.argmax(dim=1, keepdim=True)
        xp_true = pred["xpass_dense"].gather(1, true_idx).squeeze(1).clamp(1e-6, 1 - 1e-6)
        mxp = batch.has_xpass.view(-1) > 0.5
        if bool(mxp.any()):
            l_xp = F.binary_cross_entropy(xp_true[mxp], batch.y_xpass[mxp])
            components["xpass"] = float(l_xp)
            total = total + w_aux * l_xp
    return total, components


def rebalance(components: dict[str, float]) -> tuple[float, float, float]:
    """Inverse-magnitude weights so each task contributes comparably."""
    eps = 1e-6
    inv = {k: 1.0 / (components[k] + eps) for k in ("success", "xt", "run")}
    scale = 3.0 / sum(inv.values())
    return (inv["success"] * scale, inv["xt"] * scale, inv["run"] * scale)
