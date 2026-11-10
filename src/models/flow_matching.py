"""Rectified-flow bridge between registered source and target slices.

Because prospective pairs are spatially registered (same subject, MNI space),
we learn a straight-line bridge from the source slice x0 to the target slice x1
instead of starting from noise. Path: x_t = (1-t) x0 + t x1, target velocity
u = x1 - x0. Inference integrates dx/dt = v_theta from t=0 (source) to t=1.
"""

import torch
import torch.nn.functional as F

from src.models.ssim import ssim_loss


def amp_dtype(device):
    """Autocast dtype for `device`. bf16's tensor cores need Ampere+ (compute
    >= 8.0); on older cards (e.g. Turing Quadro RTX 6000, 7.5) bf16 has NO
    tensor-core path and runs on the FP32 units — ~6x slower for the same config
    than fp16, which Turing *does* accelerate. So bf16 on Ampere+, fp16 elsewhere.
    fp16 needs a GradScaler (bf16 does not); callers gate a scaler on this."""
    if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8:
        return torch.bfloat16
    return torch.float16


def pad_to_multiple(x, m=8):
    h, w = x.shape[-2:]
    ph, pw = (-h) % m, (-w) % m
    pad = (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2)
    return F.pad(x, pad), pad


def unpad(x, pad):
    left, right, top, bot = pad
    h, w = x.shape[-2:]
    return x[..., top:h - bot, left:w - right]


@torch.no_grad()
def ema_update(ema, net, decay=0.999):
    for pe, pn in zip(ema.parameters(), net.parameters(), strict=True):
        pe.mul_(decay).add_(pn, alpha=1 - decay)


def _masked_mse(pred, target, chan_mask=None):
    """MSE, optionally restricted to valid channels via ``chan_mask`` [B, C]."""
    if chan_mask is None:
        return F.mse_loss(pred, target)
    m = chan_mask[:, :, None, None]
    se = ((pred - target) ** 2 * m).sum()
    denom = m.sum() * pred.shape[-1] * pred.shape[-2]
    return se / denom.clamp_min(1.0)


def bridge_loss(net, x0, x1, src_field_idx=None, tgt_field_idx=None, chan_mask=None,
                endpoint_l2_weight=0.0, ssim_weight=0.0, probe_ssim=False):
    """Velocity regression. ``chan_mask`` [B, C] restricts the loss to valid channels
    (some contrasts missing); ``src_field_idx``/``tgt_field_idx`` [B] condition on field.

    Returns ``(loss, parts)``: the scalar to backprop plus a dict of **detached** terms
    for logging, so wandb can show what each term contributes instead of one opaque
    number. ``parts`` always carries ``vel``; the weighted extras and the raw endpoint
    SSIM appear when they are computed.

    ``endpoint_l2_weight`` > 0 adds an image-space L2 between the one-step target
    estimate ``x1_hat = x_t + (1-t) v`` and the real target ``x1`` — the source->target
    endpoint the metric actually scores — on top of the velocity MSE. It reuses the
    same forward pass (no extra net eval). Note this equals a ``(1-t)^2``-weighted
    velocity error, so it emphasises endpoint fidelity at small t.

    ``ssim_weight`` > 0 adds ``1 - SSIM(x1_hat, x1)`` on the same endpoint estimate — a
    local-structure criterion the velocity MSE cannot express, and one of the three
    scored metrics. Unlike the L2 term it does NOT reduce to a reweighted velocity
    error: SSIM's per-window mean/variance/covariance make it sensitive to contrast
    and texture rather than to per-voxel offsets.

    ``probe_ssim`` measures that same endpoint SSIM without adding it to the loss, so a
    run with ``ssim_weight=0`` still reports the quantity its A/B partner optimises.
    """
    t = torch.rand(x0.size(0), device=x0.device)
    tb = t[:, None, None, None]
    xt = (1 - tb) * x0 + tb * x1
    v = net(xt, x0, t, src_field_idx, tgt_field_idx)
    vel = _masked_mse(v, x1 - x0, chan_mask)
    parts = {"vel": vel.detach()}
    if endpoint_l2_weight <= 0 and ssim_weight <= 0 and not probe_ssim:
        return vel, parts
    # TODO: optional stronger variant — L2 on a short leak-free rollout from the source
    # (like refine_gan._rollout) instead of this one-step estimate.
    x1_hat = xt + (1 - tb) * v
    loss = vel
    if endpoint_l2_weight > 0:
        l2 = _masked_mse(x1_hat, x1, chan_mask)
        loss = loss + endpoint_l2_weight * l2
        parts["endpoint_l2"] = l2.detach()
    if ssim_weight > 0:
        sl = ssim_loss(x1_hat, x1, chan_mask)
        loss = loss + ssim_weight * sl
        parts["ssim_loss"] = sl.detach()
        parts["endpoint_ssim"] = (1 - sl).detach()
    elif probe_ssim:
        with torch.no_grad():
            parts["endpoint_ssim"] = 1 - ssim_loss(x1_hat.detach(), x1, chan_mask)
    return loss, parts


def noise_bridge_loss(net, x1, src_field_idx=None, tgt_field_idx=None, chan_mask=None):
    """Pretrain loss: bridge from Gaussian noise to a target image (learns the field
    prior). Same signature as bridge_loss; pretrain passes only ``tgt_field_idx``."""
    x0 = torch.randn_like(x1)
    if chan_mask is not None:
        x0 = x0 * chan_mask[:, :, None, None]
    return bridge_loss(net, x0, x1, src_field_idx, tgt_field_idx, chan_mask)


# Left-right (sagittal-mirror) axis of an axial slice. `load_nifti` reorients to RAS+,
# so axis 0 of the volume runs right->left and axis 1 runs posterior->anterior; slicing
# along axis 2 leaves [H=364 (R-L), W=436 (P-A)], i.e. H is the mirror axis. Flipping
# W instead would swap front and back of the head — anatomically invalid, not an
# augmentation. Verified visually on a real 3T T1W slice.
LR_FLIP_DIM = -2


def random_lr_flip(p, *tensors):
    """Mirror the left-right axis per sample, with ONE decision shared by all tensors.

    Brains are MNI-registered, so the mirror image is anatomically plausible and the
    net sees each paired slice in two orientations — cheap extra diversity for a
    finetune set of only three subjects. Source and target must be flipped together
    (they are spatially registered); passing them in one call is what guarantees that.
    """
    if p <= 0 or not tensors:
        return tensors
    flip = torch.rand(tensors[0].shape[0], device=tensors[0].device) < p
    if not bool(flip.any()):
        return tensors
    out = []
    for x in tensors:
        x = x.clone()
        x[flip] = x[flip].flip(LR_FLIP_DIM)
        out.append(x)
    return out


def drop_source_channels(x0, p=0.0):
    """Randomly zero whole source channels per sample (robustness to a missing
    contrast at inference). At least one channel is always kept."""
    if p <= 0:
        return x0
    keep = (torch.rand(x0.shape[0], x0.shape[1], 1, 1, device=x0.device) > p).float()
    allzero = keep.sum(dim=1, keepdim=True) == 0
    keep = torch.where(allzero, torch.ones_like(keep), keep)
    return x0 * keep


@torch.no_grad()
def sample(net, x0, src_field_idx=None, tgt_field_idx=None, steps=10, heun=True):
    x = x0
    ts = torch.linspace(0, 1, steps + 1, device=x0.device)
    for i in range(steps):
        t0, t1 = ts[i], ts[i + 1]
        dt = t1 - t0
        tv = t0.expand(x0.size(0))
        v = net(x, x0, tv, src_field_idx, tgt_field_idx)
        if heun and i < steps - 1:
            x_euler = x + dt * v
            tv1 = t1.expand(x0.size(0))
            v1 = net(x_euler, x0, tv1, src_field_idx, tgt_field_idx)
            x = x + dt * 0.5 * (v + v1)
        else:
            x = x + dt * v
    return x
