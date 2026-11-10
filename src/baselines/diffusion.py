"""Baseline B — SDEdit diffusion (x0-init) with the flow model's U-Net.

Standard eps-prediction DDPM on a **cosine** abar schedule, continuous `t in [0,1]`
fed straight into the existing time embedding. The x0-init variant is what lets the
U-Net stay byte-identical to the flow model's: the source is never an extra input
channel, it only ever enters as the *state* the sampler starts from.

    q_sample(x, t, eps) = sqrt(abar(t)) x + sqrt(1 - abar(t)) eps

pretrain  : plain DDPM denoising of the targets (the generative prior), target field
            embedding only.
finetune  : `paired` (default) noises the **source** and regresses the eps that points
            at the **target**, so training matches SDEdit sampling; `prior` is the
            literal textbook SDEdit, where the source enters at sampling time only.
sampling  : noise the source to `t_start`, then DDIM down to 0.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

COSINE_S = 0.008


def abar(t, schedule="cosine"):
    """Cumulative signal ratio at continuous time `t in [0,1]`; abar(0)=1, abar(1)~0."""
    if schedule == "cosine":
        f = torch.cos((t + COSINE_S) / (1 + COSINE_S) * math.pi / 2) ** 2
        f0 = math.cos(COSINE_S / (1 + COSINE_S) * math.pi / 2) ** 2
        return (f / f0).clamp(1e-6, 1.0)
    if schedule == "linear":
        return (1 - t).clamp(1e-6, 1.0)
    raise SystemExit(f"unknown diffusion schedule {schedule!r}")


def _masked_mse(pred, target, chan_mask=None):
    """MSE restricted to the valid channels of `chan_mask` [B, C] (retro 5T/7T gaps)."""
    if chan_mask is None:
        return F.mse_loss(pred, target)
    m = chan_mask[:, :, None, None]
    denom = m.sum() * pred.shape[-1] * pred.shape[-2]
    return (((pred - target) ** 2) * m).sum() / denom.clamp_min(1.0)


def q_sample(x, t, eps, schedule="cosine"):
    a = abar(t, schedule)[:, None, None, None]
    return a.sqrt() * x + (1 - a).sqrt() * eps


class DiffusionMethod:
    """`loss()` / `generate()` pair consumed by `common.train_stage` and the val hook."""

    name = "diffusion"

    def __init__(self, cfg):
        d = cfg.get("diffusion", {}) or {}
        self.schedule = d.get("schedule", "cosine")
        self.t_start = float(d.get("t_start", 0.6))
        self.sample_steps = int(d.get("sample_steps", 50))
        self.eta = float(d.get("eta", 0.0))
        self.mode = d.get("finetune_mode", "paired")
        if self.mode not in ("paired", "prior"):
            raise SystemExit(f"diffusion.finetune_mode must be paired|prior, got {self.mode!r}")
        # Lower bound on the training `t` in `paired` mode. The eps* that turns the
        # noised SOURCE into the TARGET is divided by sqrt(1 - abar(t)), which goes to
        # zero as t -> 0: without a floor those samples dominate the loss with a
        # ~1/sqrt(t) blow-up. 0.1 keeps the factor at ~6x. (Not needed in `prior` mode
        # or in pretrain, where eps* is the drawn noise itself.)
        self.t_min = float(d.get("t_min", 0.1))

    # ---------------------------------------------------------------- training

    def loss(self, net, stage, x0, x1, src_idx, tgt_idx, chan_mask):
        if stage == "pretrain" or self.mode == "prior":
            return self._prior_loss(net, x1, src_idx, tgt_idx, chan_mask)
        return self._paired_loss(net, x0, x1, src_idx, tgt_idx, chan_mask)

    def _prior_loss(self, net, x1, src_idx, tgt_idx, chan_mask):
        """Plain DDPM on the target: denoise a noised target back to the drawn eps."""
        t = torch.rand(x1.size(0), device=x1.device).clamp_min(1e-3)
        eps = torch.randn_like(x1)
        xt = q_sample(x1, t, eps, self.schedule)
        pred = net(xt, xt, t, src_idx, tgt_idx)
        mse = _masked_mse(pred, eps, chan_mask)
        return mse, {"eps_mse": mse.detach()}

    def _paired_loss(self, net, x0, x1, src_idx, tgt_idx, chan_mask):
        """Denoise the noised SOURCE toward the TARGET — the SDEdit training match.

        `x_t` is built from the source (exactly the state the sampler starts from),
        and the regression target is the eps that would have produced `x_t` from the
        *target*: eps* = (x_t - sqrt(abar) x1) / sqrt(1 - abar). At t_start = 1 this
        reduces to plain DDPM toward the target."""
        b = x0.size(0)
        t = self.t_min + (self.t_start - self.t_min) * torch.rand(b, device=x0.device)
        a = abar(t, self.schedule)[:, None, None, None]
        eps = torch.randn_like(x0)
        xt = a.sqrt() * x0 + (1 - a).sqrt() * eps
        eps_star = (xt - a.sqrt() * x1) / (1 - a).sqrt()
        pred = net(xt, xt, t, src_idx, tgt_idx)
        mse = _masked_mse(pred, eps_star, chan_mask)
        return mse, {"eps_mse": mse.detach()}

    # ---------------------------------------------------------------- sampling

    @torch.no_grad()
    def generate(self, net, x0, src_field_idx=None, tgt_field_idx=None, steps=None):
        """SDEdit: noise the source to `t_start`, then DDIM down to t=0.

        `steps` overrides `diffusion.sample_steps`."""
        n = int(steps or self.sample_steps)
        t_start = self.t_start
        x = q_sample(x0, torch.full((x0.size(0),), t_start, device=x0.device),
                     torch.randn_like(x0), self.schedule)
        ts = torch.linspace(t_start, 0.0, n + 1, device=x0.device)
        for i in range(n):
            t_cur, t_next = ts[i], ts[i + 1]
            a = abar(t_cur.expand(1), self.schedule)[0]
            a_next = abar(t_next.expand(1), self.schedule)[0]
            tv = t_cur.expand(x0.size(0))
            eps_hat = net(x, x, tv, src_field_idx, tgt_field_idx)
            x0_hat = ((x - (1 - a).sqrt() * eps_hat) / a.sqrt()).clamp(-1, 1)
            sigma = self.eta * ((1 - a_next) / (1 - a)).sqrt() * (1 - a / a_next).sqrt()
            dir_xt = (1 - a_next - sigma ** 2).clamp_min(0).sqrt() * eps_hat
            x = a_next.sqrt() * x0_hat + dir_xt
            if self.eta > 0 and i < n - 1:
                x = x + sigma * torch.randn_like(x)
        return x
