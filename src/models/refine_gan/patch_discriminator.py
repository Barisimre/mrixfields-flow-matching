"""Lightweight multi-scale conditional PatchGAN for the adversarial refinement
stage (`finetune_refine_gan`).

Judges the one-step target estimate `x1_hat` against real target slices.
Conditioning mirrors `FlowUNet` *minus the time embedding*:

  * the source slice is concatenated to the input (pix2pix-style spatial
    conditioning) -> `in_ch = 6` (3 estimate + 3 source contrast channels);
  * the source/target field strengths enter as two learned embeddings
    (`src_field_emb + tgt_field_emb`) applied FiLM-style inside every block.

Time is deliberately excluded: D judges a *clean* target estimate, whose real
distribution is time-independent. Biasing `t`->1 (handled by the training loop)
keeps `x1_hat` a good estimate without giving D a `t` it could use to excuse a
poor one.

Multi-scale follows pix2pixHD (Wang et al. 2018): the same PatchGAN is run at a
few input resolutions (full, /2, ...), the input average-pooled between scales.
`forward` returns `(logits, feats)`: one patch-logit map per scale (for the hinge
loss) plus the per-layer activations per scale (for the pix2pixHD feature-matching
loss); the hinge + feature-matching terms live in the training loop, not here.

References:
  * PatchGAN + conditional (source-concatenated) discriminator:
    Isola et al. (2017), "Image-to-Image Translation with Conditional Adversarial
    Networks" (pix2pix). arXiv:1611.07004
  * Multi-scale discriminator:
    Wang et al. (2018), "High-Resolution Image Synthesis and Semantic Manipulation
    with Conditional GANs" (pix2pixHD). arXiv:1711.11585
  * FiLM conditioning (feature-wise (gamma, beta) modulation):
    Perez et al. (2018), "FiLM: Visual Reasoning with a General Conditioning
    Layer". AAAI. arXiv:1709.07871
  * Spectral normalization on the discriminator convs:
    Miyato et al. (2018), "Spectral Normalization for Generative Adversarial
    Networks". arXiv:1802.05957
"""

import torch.nn as nn


class FiLM(nn.Module):
    """Feature-wise linear modulation from a conditioning vector (Perez et al.
    2018, arXiv:1709.07871).

    Projects `cond` to per-channel (gamma, beta) and returns `(1+gamma)*x+beta`.
    gamma/beta share one Linear, split in half; both halves are zero-init so the
    block starts as a no-op (stable warm start), matching FlowUNet's scale-shift.
    """

    def __init__(self, ch, emb_dim):
        super().__init__()
        self.proj = nn.Linear(emb_dim, ch * 2)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, cond):
        gamma, beta = self.proj(cond)[:, :, None, None].chunk(2, dim=1)
        return x * (1 + gamma) + beta


class DBlock(nn.Module):
    """Strided conv -> FiLM -> LeakyReLU. FiLM carries the affine, so the conv
    runs bias-free; spectral norm on the conv keeps D Lipschitz-bounded."""

    def __init__(self, in_ch, out_ch, emb_dim, stride=2, use_sn=True):
        super().__init__()
        conv = nn.Conv2d(in_ch, out_ch, 4, stride=stride, padding=1, bias=False)
        self.conv = nn.utils.spectral_norm(conv) if use_sn else conv
        self.film = FiLM(out_ch, emb_dim)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, cond):
        return self.act(self.film(self.conv(x), cond))


class PatchGAN(nn.Module):
    """One PatchGAN scale: a short strided-conv stack ending in a 1-channel
    patch-logit map. Channels double per layer, capped at `base*8`."""

    def __init__(self, in_ch, base, n_layers, emb_dim, use_sn=True):
        super().__init__()
        self.blocks = nn.ModuleList()
        ch = in_ch
        out = base
        for i in range(n_layers):
            stride = 2 if i < n_layers - 1 else 1
            self.blocks.append(DBlock(ch, out, emb_dim, stride=stride, use_sn=use_sn))
            ch = out
            out = min(out * 2, base * 8)
        final = nn.Conv2d(ch, 1, 4, stride=1, padding=1)
        self.final = nn.utils.spectral_norm(final) if use_sn else final

    def forward(self, x, cond):
        feats = []
        for block in self.blocks:
            x = block(x, cond)
            feats.append(x)
        return feats, self.final(x)


class MultiScalePatchDiscriminator(nn.Module):
    """Lightweight multi-scale conditional PatchGAN.

    in_ch       : channels of [x1_hat | source] (default 6 = 3+3).
    num_fields  : size of the src/tgt field embedding tables (Task 3: 5).
    base        : channels of the first conv; kept small for a light D.
    n_layers    : strided convs per scale.
    num_scales  : PatchGANs run at successively halved resolutions.
    use_sn      : spectral norm on every conv.
    """

    def __init__(self, in_ch=6, num_fields=5, base=32, n_layers=3,
                 num_scales=2, emb_dim=None, use_sn=True):
        super().__init__()
        emb_dim = emb_dim or base * 4
        self.src_field_emb = nn.Embedding(num_fields, emb_dim)
        self.tgt_field_emb = nn.Embedding(num_fields, emb_dim)
        self.scales = nn.ModuleList(
            PatchGAN(in_ch, base, n_layers, emb_dim, use_sn=use_sn)
            for _ in range(num_scales)
        )
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)

    def forward(self, x, src_field_idx, tgt_field_idx):
        """Returns (logits, feats): `logits` is one patch-logit map per scale (for
        the hinge loss); `feats` is the list of per-layer activations per scale
        (for the feature-matching loss). Callers that only need logits unpack the
        first element."""
        cond = self.src_field_emb(src_field_idx) + self.tgt_field_emb(tgt_field_idx)
        logits, feats = [], []
        for i, scale in enumerate(self.scales):
            f, logit = scale(x, cond)
            logits.append(logit)
            feats.append(f)
            if i < len(self.scales) - 1:
                x = self.downsample(x)
        return logits, feats
