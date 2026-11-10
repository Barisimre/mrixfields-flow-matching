"""Adversarial refinement stage (`refine_gan`) for the flow bridge.

The adversarial term judges the model's actual generation (a leak-free Euler rollout
from the source, run with gradients so the signal reaches the whole trajectory); the
teacher-forced velocity MSE is the faithfulness anchor, optionally joined by a
`1 - SSIM` term on that same rollout (`ssim_weight`). Multi-scale conditional
PatchGAN, hinge loss, TTUR, D warmup.
"""

from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.data.utils import FIELD_TO_DOMAIN
from src.models.flow_matching import (
    amp_dtype,
    drop_source_channels,
    ema_update,
    pad_to_multiple,
    random_lr_flip,
)
from src.models.refine_gan.patch_discriminator import MultiScalePatchDiscriminator
from src.models.ssim import ssim_loss


def hinge_d_loss(real_logits, fake_logits):
    """Multi-scale hinge loss for D. Inputs are lists of patch-logit maps (one per
    scale); averaged over scales so the magnitude is scale-count independent."""
    loss = 0.0
    for real, fake in zip(real_logits, fake_logits, strict=True):
        loss = loss + F.relu(1 - real).mean() + F.relu(1 + fake).mean()
    return loss / len(real_logits)


def hinge_g_loss(fake_logits):
    """Non-saturating hinge loss for G (maximise D's score on the fakes)."""
    return -sum(fake.mean() for fake in fake_logits) / len(fake_logits)


def feature_matching_loss(real_feats, fake_feats):
    """pix2pixHD feature-matching loss (Wang et al. 2018, arXiv:1711.11585): L1
    between D's intermediate activations on the real vs the fake, summed over every
    layer and scale and averaged. Real features are detached (a fixed target). This
    is a much gentler, better-aligned signal than the raw adversarial logit — it
    pulls the generation toward the real's multi-scale statistics (sharpness)
    without the high-variance min-max gradient, so it stabilises the stage."""
    loss = 0.0
    n = 0
    for real_scale, fake_scale in zip(real_feats, fake_feats, strict=True):
        for real, fake in zip(real_scale, fake_scale, strict=True):
            loss = loss + F.l1_loss(fake, real.detach())
            n += 1
    return loss / max(n, 1)


def _field_idx(fields, device):
    return torch.tensor([FIELD_TO_DOMAIN[f] for f in fields], device=device)


def _rollout(net, x0, src_idx, tgt_idx, steps, jitter=0.0):
    """Leak-free generation: integrate dx/dt = v from t=0 (x_t = x0) with `steps`
    Euler steps. It never sees x1; differentiable, so the adversarial gradient
    reaches every step. With jitter=0 this is exactly what inference produces.

    `jitter` (in 0-1000 timestep units) adds a fresh per-sample offset
    U(-jitter, jitter)/1000 to each step's conditioning time, so the GAN sharpening
    isn't tied to one fixed step schedule. The Euler dt stays 1/steps, so the
    rollout still advances t=0 -> t=1; only the time fed to the net is perturbed."""
    x = x0
    for i in range(steps):
        tv = torch.full((x0.size(0),), i / steps, device=x0.device)
        if jitter:
            tv = (tv + torch.empty_like(tv).uniform_(-jitter, jitter) / 1000.0).clamp(0, 1)
        v = net(x, x0, tv, src_idx, tgt_idx)
        x = x + (1.0 / steps) * v
    return x.clamp(-1, 1)


def _mse(net, x0, x1, t, src_idx, tgt_idx):
    """Teacher-forced velocity MSE — the faithfulness anchor on t~U(0,1)."""
    tb = t[:, None, None, None]
    xt = (1 - tb) * x0 + tb * x1
    v = net(xt, x0, t, src_idx, tgt_idx)
    return F.mse_loss(v, x1 - x0)


def train_refine_gan(net, ema, loader, device, pcfg, mcfg, cfg, out_dir,
                     all_pairs, multi, chan_drop, amp_enabled, wb=None, step=0):
    """The adversarial refinement loop. `wb` is the open wandb run for scalars,
    logged under `refine_gan/...` on the shared `step`; returns the updated step."""
    if not all_pairs or mcfg.get("num_fields", 0) <= 0:
        raise SystemExit("finetune_refine_gan is Task 3 only (needs all_pairs + num_fields>0)")

    dcfg = mcfg.get("discriminator", {})  # architecture lives in network.yaml
    # D still conditions on the source: in = [target | source] = 2 * out_ch
    # channels, independent of the generator's input (now x_t only).
    out_ch = mcfg.get("out_ch", 3)
    disc = MultiScalePatchDiscriminator(
        in_ch=2 * out_ch, num_fields=mcfg.get("num_fields", 5),
        base=dcfg.get("base", 32), n_layers=dcfg.get("n_layers", 3),
        num_scales=dcfg.get("num_scales", 2),
        use_sn=dcfg.get("spectral_norm", False)).to(device)
    disc_params = sum(p.numel() for p in disc.parameters())
    print(f"[refine_gan] discriminator: {disc_params / 1e6:.2f}M params")
    if wb is not None:
        wb.summary["params/discriminator_M"] = disc_params / 1e6

    # TTUR: D lr above G; low momentum.
    g_opt = torch.optim.AdamW(net.parameters(), lr=pcfg["lr"], betas=(0.5, 0.9))
    d_opt = torch.optim.AdamW(disc.parameters(), lr=pcfg.get("d_lr", 4e-4), betas=(0.0, 0.9))

    adv_weight = pcfg.get("adv_weight", 0.05)
    adv_ramp = pcfg.get("adv_ramp_steps", 0)  # ramp adv_weight 0->target over N G-steps; 0 = off
    fm_weight = pcfg.get("fm_weight", 10.0)   # pix2pixHD feature-matching weight
    d_steps = pcfg.get("d_steps", 2)        # critic updates per generator update
    d_warmup = pcfg.get("d_warmup_steps", 0)  # train D alone first (G frozen)
    rollout_steps = pcfg.get("rollout_steps", 2)  # Euler steps for the adversarial generation
    rollout_jitter = pcfg.get("rollout_jitter_range", 0)  # +/- per-step t jitter (0-1000); 0 = off
    ema_decay = pcfg.get("ema_decay", 0.999)
    flip_p = pcfg.get("flip_prob", 0.0)      # sagittal-mirror augmentation (pair-consistent)
    # 1 - SSIM between the rollout generation and the real target: a structural
    # faithfulness anchor on the SAME tensor D judges, so it constrains exactly what
    # the adversarial term is free to hallucinate (the velocity MSE only constrains
    # the teacher-forced field). Epoch-gated like the finetune term.
    ssim_weight = pcfg.get("ssim_weight", 0.0)
    ssim_delay = pcfg.get("ssim_delay_epochs", 0)
    probe_ssim = pcfg.get("probe_rollout_ssim", True)
    print(f"[refine_gan] adv_weight={adv_weight} adv_ramp_steps={adv_ramp} "
          f"fm_weight={fm_weight} d_steps={d_steps} "
          f"d_warmup_steps={d_warmup} rollout_steps={rollout_steps} "
          f"rollout_jitter_range={rollout_jitter} flip_prob={flip_p} "
          f"ssim_weight={ssim_weight} ssim_delay_epochs={ssim_delay}")

    # bf16 on Ampere+, fp16 on older cards (Turing needs it for tensor cores).
    dt = amp_dtype(device) if amp_enabled else torch.bfloat16
    use_scaler = amp_enabled and dt == torch.float16
    # Separate scalers per optimizer: D steps d_steps times per iter, G once.
    d_scaler = torch.amp.GradScaler(device.type, enabled=use_scaler)
    g_scaler = torch.amp.GradScaler(device.type, enabled=use_scaler)

    def autocast():
        return torch.autocast(device_type=device.type, dtype=dt, enabled=amp_enabled)

    local = 0  # stage-local step (drives the D warmup)
    for epoch in range(pcfg["epochs"]):
        net.train()
        disc.train()
        sums = {"g": 0.0, "d": 0.0, "mse": 0.0, "adv": 0.0, "fm": 0.0, "ssim": 0.0, "n": 0}
        cur_ssim_w = ssim_weight if epoch >= ssim_delay else 0.0
        pbar = tqdm(loader, desc=f"gan epoch {epoch}")
        max_iters = pcfg.get("max_iters")       # smoke test only; None in a real run
        for it, batch in enumerate(pbar):
            if max_iters and it >= max_iters:
                print(f"[smoke test] stopping epoch after {max_iters} iterations")
                break
            x0, _ = pad_to_multiple(batch["source"].to(device, non_blocking=True))
            x1, _ = pad_to_multiple(batch["target"].to(device, non_blocking=True))
            x0, x1 = random_lr_flip(flip_p, x0, x1)   # one decision for the pair
            if multi:
                x0 = drop_source_channels(x0, chan_drop)
            src_idx = _field_idx(batch["source_field"], device)
            tgt_idx = _field_idx(batch["target_field"], device)
            t_mse = torch.rand(x0.size(0), device=device)

            warming = local < d_warmup  # train D alone first (G frozen, no graph)

            # leak-free rollout fake: grad for the G term, frozen during warmup
            with autocast(), torch.no_grad() if warming else nullcontext():
                fake_x1 = _rollout(net, x0, src_idx, tgt_idx, rollout_steps, rollout_jitter)
            fake_in = torch.cat([fake_x1, x0], dim=1)
            real_in = torch.cat([x1, x0], dim=1)

            # --- discriminator step(s) ---
            for _ in range(d_steps):
                with autocast():
                    d_real, _ = disc(real_in, src_idx, tgt_idx)
                    d_fake, _ = disc(fake_in.detach(), src_idx, tgt_idx)
                    d_loss = hinge_d_loss(d_real, d_fake)
                d_opt.zero_grad(set_to_none=True)
                d_scaler.scale(d_loss).backward()
                d_scaler.step(d_opt)
                d_scaler.update()

            # linear adv-weight ramp over the first `adv_ramp` post-warmup G-steps,
            # so G isn't shocked the moment it starts fooling D.
            adv_w = adv_weight
            if adv_ramp:
                adv_w = adv_weight * min(1.0, max(0, local - d_warmup) / adv_ramp)

            # --- generator step (skipped during D warmup) ---
            if not warming:
                with autocast():
                    mse = _mse(net, x0, x1, t_mse, src_idx, tgt_idx)
                    g_fake, fake_feats = disc(fake_in, src_idx, tgt_idx)
                    with torch.no_grad():
                        _, real_feats = disc(real_in, src_idx, tgt_idx)
                    g_adv = hinge_g_loss(g_fake)
                    fm = feature_matching_loss(real_feats, fake_feats)
                    g_loss = mse + adv_w * g_adv + fm_weight * fm
                    if cur_ssim_w > 0:
                        sl = ssim_loss(fake_x1, x1)
                        g_loss = g_loss + cur_ssim_w * sl
                        rollout_ssim = 1 - sl.detach()
                    elif probe_ssim:
                        with torch.no_grad():
                            rollout_ssim = 1 - ssim_loss(fake_x1.detach(), x1)
                    else:
                        rollout_ssim = torch.zeros((), device=device)
                g_opt.zero_grad(set_to_none=True)
                g_scaler.scale(g_loss).backward()
                g_scaler.step(g_opt)
                g_scaler.update()
                ema_update(ema, net, ema_decay)
            else:
                # monitor only: MSE + G's objective value without updating G
                with autocast(), torch.no_grad():
                    mse = _mse(net, x0, x1, t_mse, src_idx, tgt_idx)
                    rollout_ssim = (1 - ssim_loss(fake_x1, x1) if probe_ssim
                                    else torch.zeros((), device=device))
                g_adv = hinge_g_loss([f.detach() for f in d_fake])
                fm = torch.zeros((), device=device)
                g_loss = mse + adv_w * g_adv

            sums["g"] += float(g_loss.detach())
            sums["d"] += float(d_loss.detach())
            sums["mse"] += float(mse.detach())
            sums["adv"] += float(g_adv.detach())
            sums["fm"] += float(fm.detach())
            sums["ssim"] += float(rollout_ssim)
            sums["n"] += 1
            pbar.set_postfix(g=sums["g"] / sums["n"], d=sums["d"] / sums["n"],
                             mse=float(mse.detach()), adv=float(g_adv.detach()),
                             fm=float(fm.detach()), ssim=float(rollout_ssim))
            if wb is not None:
                wb.log({"refine_gan/rollout_ssim": float(rollout_ssim),
                        "refine_gan/ssim_w": cur_ssim_w,
                        "refine_gan/g_loss": float(g_loss.detach()),
                        "refine_gan/d_loss": float(d_loss.detach()),
                        "refine_gan/mse": float(mse.detach()),
                        "refine_gan/g_adv": float(g_adv.detach()),
                        "refine_gan/fm": float(fm.detach()),
                        "refine_gan/adv_w": adv_w,
                        "refine_gan/d_real": float(d_real[0].mean().detach()),
                        "refine_gan/d_fake": float(d_fake[0].mean().detach()),
                        "refine_gan/epoch": epoch}, step=step)
            local += 1
            step += 1

        if wb is not None:
            n = max(sums["n"], 1)
            wb.log({"refine_gan/epoch_g_loss": sums["g"] / n,
                    "refine_gan/epoch_d_loss": sums["d"] / n,
                    "refine_gan/epoch_mse": sums["mse"] / n,
                    "refine_gan/epoch_adv": sums["adv"] / n,
                    "refine_gan/epoch_fm": sums["fm"] / n,
                    "refine_gan/epoch_rollout_ssim": sums["ssim"] / n,
                    "refine_gan/epoch": epoch}, step=step)

        if (epoch + 1) % pcfg.get("save_every", 1) == 0 or epoch + 1 == pcfg["epochs"]:
            ckpt = Path(out_dir) / f"checkpoint_epoch{epoch + 1}.pth"
            raw = getattr(net, "_orig_mod", net)  # unwrap torch.compile
            torch.save({"generator": raw.state_dict(), "ema": ema.state_dict(),
                        "discriminator": disc.state_dict(), "config": cfg,
                        "epoch": epoch + 1}, ckpt)
            print(f"saved {ckpt}")
    return step
