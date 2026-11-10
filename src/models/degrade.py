"""Configurable degradations for the pretrain ``x0``.

Pretrain can bridge from pure Gaussian noise to the target (a generative prior)
or from a *degraded* target back to the clean target (a restoration prior that is
much closer to the real low->high-field task). The menu — additive noise, blur,
resolution loss, gamma, gain/bias, and a smooth bias field (MRI inhomogeneity) —
is built from the ``pretrain.corruption`` config block; see
``configs/task3_any_to_any.yaml``.

Everything runs on ``[-1,1]`` tensors ``[B, C, H, W]`` (intensity ops map to
``[0,1]`` internally). Each op fires per sample with its own probability; spatial
ops (blur, downsample) draw one random strength per batch but are still gated per
sample, which keeps them cheap while staying diverse across steps.
"""

import math
import random

import torch
import torch.nn.functional as F


def _u(lo, hi, n, device):
    return lo + (hi - lo) * torch.rand(n, device=device)


def _bern(p, n, device):
    """Per-sample on/off mask, shaped [n,1,1,1] for broadcasting over [B,C,H,W]."""
    return (torch.rand(n, device=device) < p).float().view(n, 1, 1, 1)


def _gaussian_blur(x, sigma):
    k = max(3, 2 * int(math.ceil(3 * sigma)) + 1)
    r = torch.arange(k, device=x.device, dtype=x.dtype) - k // 2
    g = torch.exp(-(r ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    c = x.shape[1]
    p = k // 2
    x = F.pad(x, (0, 0, p, p), mode="reflect")
    x = F.conv2d(x, g.view(1, 1, k, 1).expand(c, 1, k, 1), groups=c)
    x = F.pad(x, (p, p, 0, 0), mode="reflect")
    x = F.conv2d(x, g.view(1, 1, 1, k).expand(c, 1, 1, k), groups=c)
    return x


class Degrader:
    """Builds a corrupted ``x0`` from the clean target ``x1``."""

    def __init__(self, cfg):
        self.mode = cfg.get("mode", "degrade")
        self.noise_only_prob = float(cfg.get("noise_only_prob", 0.3))
        self.mask_to_brain = cfg.get("mask_to_brain", True)
        self.noise = cfg.get("noise")
        self.blur = cfg.get("blur")
        self.downsample = cfg.get("downsample")
        self.gamma = cfg.get("gamma")
        self.gain_bias = cfg.get("gain_bias")
        self.bias_field = cfg.get("bias_field")

    def __call__(self, x1):
        if self.mode == "noise":
            x0 = torch.randn_like(x1)
        else:
            x0 = self._degrade(x1)
            if self.mode == "mix" and self.noise_only_prob > 0:
                sel = _bern(self.noise_only_prob, x1.shape[0], x1.device)
                x0 = sel * torch.randn_like(x1) + (1 - sel) * x0
        if self.mask_to_brain:
            # keep the skull-stripped background exactly as the target's (=0 in [0,1],
            # -1 here): corrupt only the brain, so the bridge velocity is 0 outside it.
            x0 = torch.where(x1 > -1 + 1e-4, x0, x1)
        return x0

    def _on(self, spec):
        return spec is not None and spec.get("prob", 0) > 0

    def _degrade(self, x1):
        x = x1
        n = x.shape[0]
        # --- spatial: blur, then resolution loss (per-batch strength as a python
        #     scalar — random.uniform, not a GPU tensor, so no GPU->CPU sync) ---
        if self._on(self.blur):
            sigma = random.uniform(*self.blur["sigma"])
            a = _bern(self.blur["prob"], n, x.device)
            x = a * _gaussian_blur(x, sigma) + (1 - a) * x
        if self._on(self.downsample):
            f = random.uniform(*self.downsample["factor"])
            h, w = x.shape[-2:]
            small = F.interpolate(x, size=(max(1, int(h / f)), max(1, int(w / f))),
                                  mode="bilinear", align_corners=False)
            up = F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False)
            a = _bern(self.downsample["prob"], n, x.device)
            x = a * up + (1 - a) * x
        # --- intensity ops in [0,1] (histogram shifts) ---
        u = ((x + 1) * 0.5).clamp(0, 1)
        if self._on(self.gamma):
            g = _u(*self.gamma["range"], n, x.device).view(n, 1, 1, 1)
            a = _bern(self.gamma["prob"], n, x.device)
            u = a * u.clamp_min(1e-6).pow(g) + (1 - a) * u
        if self._on(self.gain_bias):
            gain = _u(*self.gain_bias["gain"], n, x.device).view(n, 1, 1, 1)
            bias = _u(*self.gain_bias["bias"], n, x.device).view(n, 1, 1, 1)
            a = _bern(self.gain_bias["prob"], n, x.device)
            u = a * ((u - 0.5) * gain + 0.5 + bias) + (1 - a) * u
        if self._on(self.bias_field):
            s = _u(*self.bias_field["strength"], n, x.device).view(n, 1, 1, 1)
            grid = torch.rand(n, 1, 4, 4, device=x.device) * 2 - 1
            field = F.interpolate(grid, size=u.shape[-2:], mode="bilinear", align_corners=False)
            a = _bern(self.bias_field["prob"], n, x.device)
            u = a * (u * (1 + s * field)) + (1 - a) * u
        x = u.clamp(0, 1) * 2 - 1
        # --- additive noise last (on the degraded image) ---
        if self._on(self.noise):
            sigma = _u(*self.noise["sigma"], n, x.device).view(n, 1, 1, 1)
            a = _bern(self.noise["prob"], n, x.device)
            x = x + a * sigma * torch.randn_like(x)
        return x


def build_degrader(cfg):
    """cfg = the ``pretrain.corruption`` block (or None). Returns None when pretrain
    should keep the original pure-noise bridge (block absent or ``mode: noise``)."""
    if not cfg or cfg.get("mode", "degrade") == "noise":
        return None
    return Degrader(cfg)
