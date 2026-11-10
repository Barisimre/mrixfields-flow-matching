"""Conditional 2D U-Net that predicts a flow-matching velocity field.

Input  : x_t (the interpolant), optionally concatenated with the source x0.
  - concat_source=True  -> in_ch = 2*out_ch ([x_t, x0]); the legacy/leaky form.
  - concat_source=False -> in_ch = out_ch   (x_t only);  the leak-free form.
Output : velocity, out_ch channels.
With concat_source the net could algebraically recover x1 from (x_t, x0, t) — a
shortcut leak. Without it the source still enters as the t=0 integration state
(x_t=x0 at t=0), just not as a persistent channel. The flag picks between the two;
in_ch is derived from it so they can't drift.
Conditioning: continuous time t plus optional discrete source-field and target-field
indices, all folded into a shared embedding added inside every residual block.
Task 3's unified any->any model enables the field embeddings to condition on both
the source and target field strength (num_fields=0 disables field conditioning).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    """Residual block with FiLM-style conditioning.

    Default (`scale_shift=False`) is DDPM-style additive (beta-only) bias.
    With `scale_shift=True` the conditioning modulates the second norm as
    `norm2(h) * (1 + gamma) + beta` (scale-shift-norm, ADM). The gamma/beta
    halves come from a single Linear whose output is split; the gamma half is
    zero-initialized so scaling starts at 1 (stable warm start).
    """

    def __init__(self, in_ch, out_ch, emb_dim, groups=8, scale_shift=False):
        super().__init__()
        self.scale_shift = scale_shift
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb = nn.Linear(emb_dim, out_ch * 2 if scale_shift else out_ch)
        if scale_shift:
            nn.init.zeros_(self.emb.weight)
            nn.init.zeros_(self.emb.bias)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        if self.scale_shift:
            gamma, beta = self.emb(emb)[:, :, None, None].chunk(2, dim=1)
            h = self.norm2(h) * (1 + gamma) + beta
            h = self.conv2(F.silu(h))
        else:
            h = h + self.emb(emb)[:, :, None, None]
            h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class AttnBlock(nn.Module):
    def __init__(self, ch, groups=8):
        super().__init__()
        self.norm = nn.GroupNorm(groups, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        q = q.reshape(b, c, h * w).transpose(1, 2)
        k = k.reshape(b, c, h * w)
        v = v.reshape(b, c, h * w).transpose(1, 2)
        attn = torch.softmax(q @ k / math.sqrt(c), dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, c, h, w)
        return x + self.proj(out)


class Down(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Up(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return self.op(F.interpolate(x, scale_factor=2, mode="nearest"))


class FlowUNet(nn.Module):
    def __init__(self, out_ch=1, in_ch=None, concat_source=True, base=64,
                 mults=(1, 2, 4, 4), num_fields=0,
                 attn_at_bottleneck=True, scale_shift=False, num_res_blocks=2):
        super().__init__()
        self.concat_source = concat_source
        expected_in = out_ch * (2 if concat_source else 1)
        if in_ch is None:
            in_ch = expected_in
        elif in_ch != expected_in:
            raise ValueError(
                f"in_ch={in_ch} inconsistent with out_ch={out_ch}, "
                f"concat_source={concat_source} (expected {expected_in})")
        emb_dim = base * 4
        def rb(i, o):
            return ResBlock(i, o, emb_dim, scale_shift=scale_shift)
        self.time_mlp = nn.Sequential(nn.Linear(base, emb_dim), nn.SiLU(),
                                      nn.Linear(emb_dim, emb_dim))
        self.src_field_emb = nn.Embedding(num_fields, emb_dim) if num_fields else None
        self.tgt_field_emb = nn.Embedding(num_fields, emb_dim) if num_fields else None

        # We zero init the source embedding, since in the pre training this field
        # doesnt do anything. Training also should only pass through this if there
        # is a source unit
        if self.src_field_emb is not None:
            nn.init.zeros_(self.src_field_emb.weight)
        self.base = base

        self.stem = nn.Conv2d(in_ch, base, 3, padding=1)

        chs = [base * m for m in mults]
        self.downs = nn.ModuleList()
        skip_chs = [base]
        prev = base
        for i, ch in enumerate(chs):
            self.downs.append(rb(prev, ch))
            for _ in range(num_res_blocks - 1):
                self.downs.append(rb(ch, ch))
            skip_chs += [ch] * num_res_blocks
            prev = ch
            if i < len(chs) - 1:
                self.downs.append(Down(ch))
                skip_chs.append(ch)

        self.mid1 = rb(prev, prev)
        self.mid_attn = AttnBlock(prev) if attn_at_bottleneck else nn.Identity()
        self.mid2 = rb(prev, prev)

        self.ups = nn.ModuleList()
        for i, ch in enumerate(reversed(chs)):
            for j in range(num_res_blocks + 1):
                cur = prev if j == 0 else ch
                self.ups.append(rb(cur + skip_chs.pop(), ch))
            prev = ch
            if i < len(chs) - 1:
                self.ups.append(Up(ch))

        self.out = nn.Sequential(nn.GroupNorm(8, prev), nn.SiLU(),
                                 nn.Conv2d(prev, out_ch, 3, padding=1))

    def forward(self, xt, x0, t, src_field_idx=None, tgt_field_idx=None):
        x = torch.cat([xt, x0], dim=1) if self.concat_source else xt
        emb = self.time_mlp(timestep_embedding(t, self.base))
        if self.src_field_emb is not None and src_field_idx is not None:
            emb = emb + self.src_field_emb(src_field_idx)
        if self.tgt_field_emb is not None and tgt_field_idx is not None:
            emb = emb + self.tgt_field_emb(tgt_field_idx)

        h = self.stem(x)
        skips = [h]
        for layer in self.downs:
            h = layer(h, emb) if isinstance(layer, ResBlock) else layer(h)
            skips.append(h)

        h = self.mid1(h, emb)
        h = self.mid_attn(h)
        h = self.mid2(h, emb)

        for layer in self.ups:
            if isinstance(layer, ResBlock):
                h = layer(torch.cat([h, skips.pop()], dim=1), emb)
            else:
                h = layer(h)
        return self.out(h)
