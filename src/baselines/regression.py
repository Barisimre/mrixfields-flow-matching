"""Baseline A — plain supervised regression with the flow model's U-Net.

No iterative sampling: the net maps the source slice stack straight to the target
slice stack in one forward pass. The time embedding is still in the graph (the U-Net
is unchanged) but is fed a constant `t = 0`, so it contributes a fixed bias — this is
what makes the architecture byte-identical to the flow model's while the *objective*
is the only thing under comparison.

pretrain reconstructs the clean target from a degraded copy of it (the same
`pretrain.corruption` menu the flow pretrain uses), finetune maps source -> target.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from src.models.ssim import ssim_loss


def _masked(pred, target, chan_mask=None, kind="l1"):
    """L1/L2, optionally restricted to valid channels via `chan_mask` [B, C].

    Copied from the `_masked_mse` pattern in `src/models/flow_matching.py` rather than
    imported-and-generalised, so the flow loss stays untouched."""
    if chan_mask is None:
        return F.l1_loss(pred, target) if kind == "l1" else F.mse_loss(pred, target)
    m = chan_mask[:, :, None, None]
    err = (pred - target).abs() if kind == "l1" else (pred - target) ** 2
    denom = m.sum() * pred.shape[-1] * pred.shape[-2]
    return (err * m).sum() / denom.clamp_min(1.0)


class RegressionMethod:
    """`loss()` / `generate()` pair consumed by `common.train_stage` and the val hook."""

    name = "regression"

    def __init__(self, cfg):
        rcfg = cfg.get("regression", {}) or {}
        self.kind = rcfg.get("loss", "l1")
        if self.kind not in ("l1", "l2"):
            raise SystemExit(f"regression.loss must be l1 or l2, got {self.kind!r}")
        self.ssim_weight = float(rcfg.get("ssim_weight", 0.0))

    def loss(self, net, stage, x0, x1, src_idx, tgt_idx, chan_mask):
        if x0 is None:
            raise SystemExit(
                "regression pretrain needs a degraded input: set `pretrain.corruption` "
                "(mode: degrade or mix) in the baseline config — there is nothing to "
                "regress from otherwise.")
        t = torch.zeros(x0.size(0), device=x0.device)
        pred = net(x0, x0, t, src_idx, tgt_idx)
        rec = _masked(pred, x1, chan_mask, self.kind)
        parts = {self.kind: rec.detach()}
        loss = rec
        if self.ssim_weight > 0:
            sl = ssim_loss(pred, x1, chan_mask)
            loss = loss + self.ssim_weight * sl
            parts["ssim_loss"] = sl.detach()
            parts["ssim"] = (1 - sl).detach()
        return loss, parts

    @torch.no_grad()
    def generate(self, net, x0, src_field_idx=None, tgt_field_idx=None, steps=None):
        """One forward pass. `steps` is accepted (the shared inference plumbing
        passes it) and ignored — there is nothing to integrate."""
        t = torch.zeros(x0.size(0), device=x0.device)
        return net(x0, x0, t, src_field_idx, tgt_field_idx)
