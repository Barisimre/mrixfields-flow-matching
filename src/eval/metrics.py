"""nRMSE / SSIM / LPIPS, matching the official MRIxFields2026 scorer.

One implementation, so a number means the same thing everywhere. All functions take
images in **[0,1]** -- the range the challenge scores in, not the [-1,1] compute range.

The `*_torch` variants are batched GPU equivalents of the numpy/skimage ones.
`ssim_torch` reproduces
`skimage.metrics.structural_similarity` defaults exactly:

    win_size=7, uniform (box) window, unbiased covariance (NP/(NP-1)),
    K1=0.01, K2=0.03, and the `pad=(win_size-1)//2` border crop.

Because the border is cropped, the box filter is a plain `valid` convolution — the
boundary mode skimage would use never touches the retained region, so the two agree
to float precision. `check_ssim_torch()` asserts that against skimage.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity

_BOX: dict[tuple, torch.Tensor] = {}
_LPIPS = None


# ---------------------------------------------------------------- numpy (scorer)

def nrmse(pred, target):
    """||pred - target|| / ||target||, the scorer's normalized RMSE."""
    p, t = pred.astype(np.float64), target.astype(np.float64)
    n = np.linalg.norm(t)
    return float(np.linalg.norm(p - t) / n) if n > 1e-10 else 0.0


def ssim2d(pred, target):
    """SSIM on one 2D slice; `data_range` from the target, as the scorer does."""
    p, t = pred.astype(np.float64), target.astype(np.float64)
    dr = t.max() - t.min()
    if dr < 1e-10:
        return 1.0
    return float(structural_similarity(p, t, data_range=dr))


def ssim_stack(pred, target, slice_axis=2):
    """Mean per-slice SSIM over a stack, with one `data_range` for the whole stack
    (matching `Evaluation/evaluate.compute_ssim`). Flat slices are skipped."""
    p, t = pred.astype(np.float64), target.astype(np.float64)
    dr = t.max() - t.min()
    if dr < 1e-10:
        return 1.0
    vals = []
    for i in range(p.shape[slice_axis]):
        sl = [slice(None)] * p.ndim
        sl[slice_axis] = i
        sl = tuple(sl)
        if t[sl].max() - t[sl].min() < 1e-10:
            continue
        vals.append(structural_similarity(p[sl], t[sl], data_range=dr))
    return float(np.mean(vals)) if vals else 1.0


# ------------------------------------------------------------------ torch (fast)

def _box(win, channels, device, dtype):
    key = (win, channels, device, dtype)
    w = _BOX.get(key)
    if w is None:
        w = torch.full((channels, 1, win, win), 1.0 / (win * win), device=device, dtype=dtype)
        _BOX[key] = w
    return w


def ssim_torch(pred, target, win=7, data_range=None):
    """Batched SSIM -> [B, C]. `pred`/`target` are [B, C, H, W] in [0,1].

    `data_range` defaults to the per-(sample, channel) target min-max, which is what
    the scorer uses. Pairs whose target is flat score 1.0, again as the scorer does.
    """
    b, c = pred.shape[:2]
    p, t = pred.float(), target.float()
    if data_range is None:
        flat = t.reshape(b, c, -1)
        data_range = (flat.amax(-1) - flat.amin(-1))[:, :, None, None]
    flat_tgt = data_range.reshape(b, c) < 1e-10
    dr = data_range.clamp_min(1e-10)

    w = _box(win, c, p.device, p.dtype)
    ux, uy = F.conv2d(p, w, groups=c), F.conv2d(t, w, groups=c)
    uxx, uyy = F.conv2d(p * p, w, groups=c), F.conv2d(t * t, w, groups=c)
    uxy = F.conv2d(p * t, w, groups=c)
    # skimage's unbiased covariance normalisation: NP / (NP - 1), NP = win**ndim.
    cov_norm = (win * win) / (win * win - 1)
    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)

    c1, c2 = (0.01 * dr) ** 2, (0.03 * dr) ** 2
    smap = ((2 * ux * uy + c1) * (2 * vxy + c2)) / ((ux * ux + uy * uy + c1) * (vx + vy + c2))
    out = smap.mean(dim=(-2, -1))
    return torch.where(flat_tgt, torch.ones_like(out), out)


def nrmse_torch(pred, target):
    """Batched nRMSE -> [B, C]; [B, C, H, W] in [0,1]."""
    p = pred.float().flatten(2)
    t = target.float().flatten(2)
    n = t.norm(dim=-1)
    return torch.where(n > 1e-10, (p - t).norm(dim=-1) / n.clamp_min(1e-10),
                       torch.zeros_like(n))


def lpips_torch(pred, target, device, batch_size=64):
    """Batched LPIPS(alex) -> [B, C]. Inputs [B, C, H, W] in [0,1]; each channel is a
    separate grayscale image, replicated to 3 for the backbone (never treated as RGB).
    Loaded through `src.utils.lpips_shim` (this env's torchvision is ABI-broken)."""
    global _LPIPS
    if _LPIPS is None:
        from src.utils.lpips_shim import load_lpips
        _LPIPS = load_lpips(device=device)
    b, c = pred.shape[:2]
    p = (pred.float().reshape(b * c, 1, *pred.shape[-2:]) * 2 - 1).repeat(1, 3, 1, 1)
    t = (target.float().reshape(b * c, 1, *target.shape[-2:]) * 2 - 1).repeat(1, 3, 1, 1)
    outs = []
    with torch.no_grad():
        for i in range(0, p.shape[0], batch_size):
            outs.append(_LPIPS(p[i:i + batch_size], t[i:i + batch_size]).flatten())
    return torch.cat(outs).reshape(b, c)


def check_ssim_torch(shape=(2, 3, 96, 112), seed=0, tol=1e-4, device="cpu"):
    """Assert `ssim_torch` matches skimage on random data. Returns the max abs diff."""
    g = np.random.default_rng(seed)
    p = g.random(shape).astype(np.float32)
    t = np.clip(g.random(shape) * 0.9 + p * 0.1, 0, 1).astype(np.float32)
    mine = ssim_torch(torch.from_numpy(p).to(device), torch.from_numpy(t).to(device))
    ref = np.array([[ssim2d(p[b, c], t[b, c]) for c in range(shape[1])]
                    for b in range(shape[0])])
    diff = float(np.abs(mine.cpu().numpy() - ref).max())
    assert diff < tol, f"ssim_torch deviates from skimage by {diff:.2e} (> {tol:.0e})"
    return diff
