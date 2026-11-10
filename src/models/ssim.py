"""Differentiable SSIM (Wang et al. 2004) for use as a training loss.

The leaderboard scores SSIM, but the bridge only ever sees a velocity MSE — a
per-voxel criterion blind to local structure. This gives the finetune stage an
image-space structural term on the endpoint estimate. Gaussian 11x11 window,
`valid` convolution (no border padding), computed per channel.

Range: `ssim_loss` takes the `[-1,1]` compute-range tensors and maps them to `[0,1]`
(`data_range=1.0`) before comparing — the range the graders score in. This is NOT
cosmetic. SSIM's luminance term `(2 m1 m2 + C1) / (m1^2 + m2^2 + C1)` is not invariant
to an affine shift: on `[0,1]` data the window means are non-negative, but on `[-1,1]`
they straddle zero, and any window whose mean lands near 0 collapses the denominator
to C1 and makes the term explode. Measured on a real 3T->7T pair: 0.79 scored in
`[0,1]` vs 0.36 in `[-1,1]` with `data_range=2`. Only the contrast/structure half of
SSIM survives the rescaling; the luminance half does not.
"""

import torch
import torch.nn.functional as F

_WINDOWS: dict[tuple, torch.Tensor] = {}


def _window(win_size, sigma, channels, device, dtype):
    key = (win_size, sigma, channels, device, dtype)
    w = _WINDOWS.get(key)
    if w is None:
        coords = torch.arange(win_size, device=device, dtype=dtype) - (win_size - 1) / 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        w = (g[:, None] * g[None, :]).expand(channels, 1, win_size, win_size).contiguous()
        _WINDOWS[key] = w
    return w


def ssim_per_channel(pred, target, data_range=2.0, win_size=11, sigma=1.5):
    """Mean SSIM per sample and channel -> [B, C]."""
    c = pred.shape[1]
    w = _window(win_size, sigma, c, pred.device, pred.dtype)
    mu1 = F.conv2d(pred, w, groups=c)
    mu2 = F.conv2d(target, w, groups=c)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sig1 = F.conv2d(pred * pred, w, groups=c) - mu1_sq
    sig2 = F.conv2d(target * target, w, groups=c) - mu2_sq
    sig12 = F.conv2d(pred * target, w, groups=c) - mu1_mu2

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    smap = ((2 * mu1_mu2 + c1) * (2 * sig12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sig1 + sig2 + c2))
    return smap.mean(dim=(-2, -1))


def ssim_loss(pred, target, chan_mask=None):
    """1 - SSIM, averaged over valid channels (``chan_mask`` [B, C]).

    ``pred``/``target`` are the ``[-1,1]`` compute-range tensors; both are mapped to
    ``[0,1]`` here (see the module docstring — SSIM is only well-behaved on
    non-negative data). ``pred`` is deliberately not clamped: the net's output is
    unbounded, and clamping would zero the gradient exactly where it overshoots.

    Always evaluated in fp32: the squares and the variance subtraction above lose
    too much precision under bf16/fp16 autocast.
    """
    with torch.autocast(device_type=pred.device.type, enabled=False):
        p = (pred.float() + 1) / 2
        t = (target.float() + 1) / 2
        s = ssim_per_channel(p, t, data_range=1.0)
        if chan_mask is None:
            return 1 - s.mean()
        m = chan_mask.to(s.dtype)
        return 1 - (s * m).sum() / m.sum().clamp_min(1.0)
