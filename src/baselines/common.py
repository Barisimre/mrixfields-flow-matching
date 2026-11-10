"""Shared training machinery for the paper baselines (regression, diffusion).

Deliberately a copy of the pieces of `src/scripts/train_flow.py` the baselines need
(EMA, gradient norm, loaders, GPU batch scaling, wandb init) rather than an import of
its private helpers: the flow pipeline must not change to accommodate a baseline.

The one genuinely new thing is the **wall-clock budget**: each stage runs until
`max_hours` is spent (checked after every epoch and every `budget_check_every` steps),
not for a fixed number of epochs. That is the "same amount of compute" knob the
baseline comparison is built on.
"""
from __future__ import annotations

import copy
import os
import time
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from src.data.dataset import (
    H5MultiContrastPairedDataset,
    H5MultiContrastUnpairedDataset,
    MultiContrastPairedDataset,
    MultiContrastUnpairedDataset,
)
from src.data.utils import ABBR_TO_SPLIT, FIELD_STRENGTHS, FIELD_TO_DOMAIN
from src.models.degrade import build_degrader
from src.models.flow_matching import (
    amp_dtype,
    drop_source_channels,
    ema_update,
    pad_to_multiple,
    random_lr_flip,
)


def stage_dir(method):
    """Checkpoint subdir under $OUTPUT_DIR. Deliberately NOT one of the flow model's
    ("pretrain"/"finetune"/"refine_gan") so `run_task3.sh`'s `latest()` can never pick
    a baseline checkpoint up by accident."""
    return f"baseline_{method}"


def field_idx_tensor(fields, device):
    """Map a batch of field-strength strings to a LongTensor of domain indices."""
    return torch.tensor([FIELD_TO_DOMAIN[f] for f in fields], device=device)


def raw_of(net):
    """The underlying module (unwrap torch.compile) for state_dict load/save/copy."""
    return getattr(net, "_orig_mod", net)


def fresh_ema(net):
    ema = copy.deepcopy(raw_of(net)).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    return ema


def grad_norm(net, max_norm=float("inf")):
    """Total gradient L2 norm; clips only if a finite `max_norm` was asked for.

    With no clipping this is read-only — it must not perturb the gradients."""
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    if not grads:
        return 0.0
    if max_norm < float("inf"):
        return float(torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm))
    return float(torch.linalg.vector_norm(
        torch.stack([torch.linalg.vector_norm(g.detach()) for g in grads])))


def make_loader(stage, pcfg, h5, data_root, rt):
    """One ConcatDataset over all fields (pretrain) or all 20 directed pairs (finetune).

    Identical sampling to the flow stages, so the only thing that differs between a
    baseline and the flow model is the training objective."""
    split = ABBR_TO_SPLIT[pcfg["split"]]
    if stage == "pretrain":
        ds = ConcatDataset([
            H5MultiContrastUnpairedDataset(h5, split, f) if h5
            else MultiContrastUnpairedDataset(data_root, split, f)
            for f in FIELD_STRENGTHS])
    else:
        ds = ConcatDataset([
            H5MultiContrastPairedDataset(h5, split, s, t) if h5
            else MultiContrastPairedDataset(data_root, split, s, t)
            for s in FIELD_STRENGTHS for t in FIELD_STRENGTHS if s != t])
    nw = pcfg.get("num_workers", 4)
    loader_kw = {}
    if nw > 0:
        loader_kw["prefetch_factor"] = rt.get("prefetch_factor", 2)
        loader_kw["persistent_workers"] = rt.get("persistent_workers", False)
    loader = DataLoader(ds, batch_size=pcfg["batch_size"], shuffle=True,
                        num_workers=nw, drop_last=True, pin_memory=True, **loader_kw)
    return loader, len(ds)


def apply_gpu_batch_scale(cfg, device):
    """Double every stage's batch size on a big card (>32 GB), as train_flow does."""
    info = {"name": "cpu", "vram_gb": 0.0, "batch_scale": 1}
    if device.type != "cuda":
        return info
    props = torch.cuda.get_device_properties(device)
    vram_gb = props.total_memory / 1024**3
    scale = 2 if vram_gb > 32 else 1
    info = {"name": props.name, "vram_gb": vram_gb, "batch_scale": scale}
    print(f"[gpu] {props.name} | {vram_gb:.1f} GB VRAM | batch_scale x{scale}")
    if scale > 1:
        for blk in ("pretrain", "finetune"):
            bs = cfg.get(blk, {}).get("batch_size")
            if bs is not None:
                cfg[blk]["batch_size"] = bs * scale
                print(f"[gpu] {blk}.batch_size {bs} -> {cfg[blk]['batch_size']}")
    return info


def setup_runtime(cfg, device):
    """Apply the config `runtime:` throughput toggles; returns whether amp is on."""
    rt = cfg.get("runtime", {})
    if rt.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True
    if rt.get("tf32", False):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return rt.get("amp", False) and device.type == "cuda"


def init_wandb(cfg, config_path, task_name, gpu_info, n_params):
    """ONE wandb run for the whole baseline pipeline, with a codebase snapshot."""
    if not os.environ.get("WANDB_PROJECT"):
        return None
    import wandb
    wb = wandb.init(project=os.environ["WANDB_PROJECT"],
                    group=os.environ.get("WANDB_GROUP", task_name),
                    name=task_name, job_type="train", config=cfg,
                    reinit="create_new", save_code=True)
    wb.summary["params/generator_M"] = n_params / 1e6
    wb.summary["gpu/name"] = gpu_info["name"]
    wb.summary["gpu/vram_gb"] = gpu_info["vram_gb"]
    wb.summary["gpu/batch_scale"] = gpu_info["batch_scale"]
    wb.config.update({"gpu": gpu_info}, allow_val_change=True)
    cfg_path = Path(config_path)
    for f in [cfg_path, cfg_path.parent / cfg.get("network", "")]:
        if f.name and f.is_file():
            wb.save(str(f), base_path=str(f.parent), policy="now")
    try:
        import zipfile
        repo = Path(__file__).resolve().parents[2]
        code_ext = {".py", ".yaml", ".yml", ".sh", ".sbatch", ".toml", ".md", ".cfg"}
        skip = {"data", "outputs", "model_weights", ".git", "__pycache__",
                ".venv", "wandb", ".ipynb_checkpoints"}
        zip_path = Path(wb.dir) / "codebase.zip"
        n_code = 0
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for dirpath, dirnames, filenames in os.walk(repo):
                dirnames[:] = [d for d in dirnames if d not in skip]
                for fn in filenames:
                    p = Path(dirpath) / fn
                    if p.suffix in code_ext:
                        z.write(p, p.relative_to(repo))
                        n_code += 1
        wb.save(str(zip_path), base_path=str(zip_path.parent), policy="now")
        print(f"[wandb] logged codebase snapshot ({n_code} files -> codebase.zip)")
    except Exception as e:  # noqa: BLE001
        print(f"[wandb] codebase snapshot failed: {e}")
    return wb


def _batch_tensors(stage, batch, device, flip_p, chan_drop, degrader):
    """(x0, x1, src_idx, tgt_idx, chan_mask) for one batch, padded and augmented.

    pretrain: x1 is the target image, x0 its degraded copy (None when no corruption
    block is configured — the diffusion pretrain does not use one). Only the target
    field is conditioned on, and `chan_mask` marks the contrasts that really exist.
    finetune: x0 is the registered source, x1 the target, both fields conditioned.
    """
    if stage == "pretrain":
        x1, _ = pad_to_multiple(batch["image"].to(device, non_blocking=True))
        (x1,) = random_lr_flip(flip_p, x1)
        mask = batch["mask"].to(device, non_blocking=True) if "mask" in batch else None
        tgt_idx = field_idx_tensor(batch["field_strength"], device)
        x0 = None
        if degrader is not None:
            x0 = degrader(x1)
            if mask is not None:
                x0 = x0 * mask[:, :, None, None]
        return x0, x1, None, tgt_idx, mask
    x0, _ = pad_to_multiple(batch["source"].to(device, non_blocking=True))
    x1, _ = pad_to_multiple(batch["target"].to(device, non_blocking=True))
    x0, x1 = random_lr_flip(flip_p, x0, x1)     # one decision for the registered pair
    x0 = drop_source_channels(x0, chan_drop)
    src_idx = field_idx_tensor(batch["source_field"], device)
    tgt_idx = field_idx_tensor(batch["target_field"], device)
    return x0, x1, src_idx, tgt_idx, None


def train_stage(stage, method, net, ema, loader, device, pcfg, amp_enabled, out_dir,
                cfg, wb, step):
    """Time-budgeted epoch loop. Logs `<stage>/...` on the shared `step`, returns it.

    Runs until `pcfg["max_hours"]` of wall clock is spent (or `epochs` is reached,
    whichever comes first). The budget is checked between epochs *and* every
    `budget_check_every` steps, so a single long epoch cannot overrun the SLURM
    wall time.
    """
    opt = torch.optim.AdamW(net.parameters(), lr=pcfg["lr"], betas=(0.9, 0.99))
    ema_decay = pcfg.get("ema_decay", 0.999)
    flip_p = pcfg.get("flip_prob", 0.0)
    chan_drop = pcfg.get("channel_dropout", 0.0) if stage != "pretrain" else 0.0
    grad_clip = pcfg.get("grad_clip", 0) or float("inf")
    degrader = build_degrader(pcfg.get("corruption")) if stage == "pretrain" else None
    max_hours = float(pcfg.get("max_hours", 0) or 0)
    budget = max_hours * 3600 if max_hours > 0 else float("inf")
    check_every = int(pcfg.get("budget_check_every", 200))

    dt = amp_dtype(device) if amp_enabled else torch.bfloat16
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled and dt == torch.float16)

    t_stage = time.time()
    epochs_done = 0
    out_of_time = False
    print(f"[budget] {stage}: {max_hours or float('inf')} h, checked every "
          f"{check_every} steps")

    for epoch in range(pcfg["epochs"]):
        net.train()
        run_loss = None
        loss_sum, loss_n, n_slices = 0.0, 0, 0
        part_sums = {}
        t_epoch = time.time()
        pbar = tqdm(loader, desc=f"{stage} epoch {epoch}")
        for i, batch in enumerate(pbar):
            with torch.autocast(device_type=device.type, dtype=dt, enabled=amp_enabled):
                x0, x1, src_idx, tgt_idx, mask = _batch_tensors(
                    stage, batch, device, flip_p, chan_drop, degrader)
                loss, parts = method.loss(net, stage, x0, x1, src_idx, tgt_idx, mask)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)          # measure the norm in real units
            gnorm = grad_norm(net, grad_clip)
            scaler.step(opt)
            scaler.update()
            ema_update(ema, net, ema_decay)

            cur = float(loss.detach())
            run_loss = cur if run_loss is None else 0.98 * run_loss + 0.02 * cur
            loss_sum += cur
            loss_n += 1
            n_slices += x1.shape[0]
            for k, v in parts.items():
                part_sums[k] = part_sums.get(k, 0.0) + float(v)
            pbar.set_postfix(loss=run_loss)
            step += 1
            if wb is not None and step % 20 == 0:
                log = {f"{stage}/loss": run_loss, f"{stage}/epoch": epoch,
                       f"{stage}/grad_norm": gnorm,
                       f"{stage}/lr": opt.param_groups[0]["lr"],
                       f"{stage}/hours_used": (time.time() - t_stage) / 3600}
                log.update({f"{stage}/loss_{k}": float(v) for k, v in parts.items()})
                wb.log(log, step=step)
            if (i + 1) % check_every == 0 and time.time() - t_stage > budget:
                print(f"[budget] {stage}: {max_hours} h spent mid-epoch {epoch} — stopping")
                out_of_time = True
                pbar.close()
                break

        dt_epoch = time.time() - t_epoch
        epochs_done = epoch + 1
        if wb is not None:
            log = {f"{stage}/epoch_loss": loss_sum / max(loss_n, 1),
                   f"{stage}/epoch": epoch,
                   f"{stage}/epoch_seconds": dt_epoch,
                   f"{stage}/slices_per_sec": n_slices / max(dt_epoch, 1e-6)}
            log.update({f"{stage}/epoch_{k}": v / max(loss_n, 1) for k, v in part_sums.items()})
            wb.log(log, step=step)

        last = out_of_time or epochs_done == pcfg["epochs"]
        if epochs_done % pcfg.get("save_every", 1) == 0 or last:
            ckpt = out_dir / f"checkpoint_epoch{epochs_done}.pth"
            torch.save({"generator": raw_of(net).state_dict(), "ema": ema.state_dict(),
                        "config": cfg, "epoch": epochs_done,
                        "method": cfg.get("method")}, ckpt)
            print(f"saved {ckpt}")
        if out_of_time or time.time() - t_stage > budget:
            if not out_of_time:
                print(f"[budget] {stage}: {max_hours} h spent after epoch {epoch} — stopping")
            break

    hours = (time.time() - t_stage) / 3600
    print(f"[budget] {stage} finished: {epochs_done} epochs in {hours:.2f} h")
    if wb is not None:
        wb.log({f"{stage}/hours_used": hours, f"{stage}/epochs_completed": epochs_done},
               step=step)
        wb.summary[f"{stage}/hours_used"] = hours
        wb.summary[f"{stage}/epochs_completed"] = epochs_done
    return step
