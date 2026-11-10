"""Train the unified Task 3 any->any flow bridge: the whole pipeline in one run.

One process runs the stages in order (pretrain -> finetune -> refine_gan), flowing
the model through them in memory (each warm-starts from the previous stage's EMA)
and logging to one wandb run with stage-namespaced metrics. Run a subset for
ablations/resumes, e.g. `--stages finetune` or `--stages refine_gan --init_from X`.
"""

import argparse
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
from src.models.builder import build_flow_net
from src.models.degrade import build_degrader
from src.models.flow_matching import (
    amp_dtype,
    bridge_loss,
    drop_source_channels,
    ema_update,
    noise_bridge_loss,
    pad_to_multiple,
    random_lr_flip,
)
from src.models.refine_gan.gan_refine import train_refine_gan
from src.utils.config import load_config
from src.utils.env import get_data_dir, get_device, get_output_dir, load_env

# Stage name -> output subdir (the config block names, so checkpoint paths are stable).
STAGE_DIR = {"pretrain": "pretrain", "finetune": "finetune", "refine_gan": "refine_gan"}


def field_idx_tensor(fields, device):
    """Map a batch of field-strength strings to a LongTensor of domain indices."""
    return torch.tensor([FIELD_TO_DOMAIN[f] for f in fields], device=device)


def raw_of(net):
    """The underlying module (unwrap torch.compile) for state_dict load/save/copy."""
    return getattr(net, "_orig_mod", net)


def grad_norm(net, max_norm=float("inf")):
    """Total gradient L2 norm, clipping to `max_norm` only if one was asked for.

    With no clipping this is deliberately **read-only**: it must not perturb the
    gradients. (`clip_grad_norm_` with `max_norm=inf` would scale by 1.0 in the normal
    case but by NaN whenever the norm overflows, so it is not a safe way to just
    *measure*.)"""
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    if not grads:
        return 0.0
    if max_norm < float("inf"):
        return float(torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm))
    return float(torch.linalg.vector_norm(
        torch.stack([torch.linalg.vector_norm(g.detach()) for g in grads])))


def fresh_ema(net):
    ema = copy.deepcopy(raw_of(net)).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    return ema


def stage_pcfg(cfg, stage):
    """Resolve a stage's hyperparameters. refine_gan overlays its block on finetune."""
    if stage == "pretrain":
        return cfg["pretrain"]
    if stage == "finetune":
        return cfg["finetune"]
    if stage == "refine_gan":
        return {**cfg["finetune"], **cfg.get("finetune_refine_gan", {})}
    raise SystemExit(f"unknown stage {stage}")


def make_loader(stage, pcfg, h5, data_root, rt):
    """One ConcatDataset over all fields (pretrain) or all directed pairs (bridge)."""
    split = ABBR_TO_SPLIT[pcfg["split"]]
    if stage == "pretrain":
        parts = [H5MultiContrastUnpairedDataset(h5, split, f) if h5
                 else MultiContrastUnpairedDataset(data_root, split, f)
                 for f in FIELD_STRENGTHS]
    else:
        parts = [H5MultiContrastPairedDataset(h5, split, s, t) if h5
                 else MultiContrastPairedDataset(data_root, split, s, t)
                 for s in FIELD_STRENGTHS for t in FIELD_STRENGTHS if s != t]
    ds = ConcatDataset(parts)
    nw = pcfg.get("num_workers", 4)
    loader_kw = {}
    if nw > 0:
        loader_kw["prefetch_factor"] = rt.get("prefetch_factor", 2)
        loader_kw["persistent_workers"] = rt.get("persistent_workers", False)
    loader = DataLoader(ds, batch_size=pcfg["batch_size"], shuffle=True,
                        num_workers=nw, drop_last=True, pin_memory=True, **loader_kw)
    return loader, len(ds)


def train_bridge(stage, net, ema, loader, device, pcfg, amp_enabled, out_dir, cfg, wb, step):
    """The pretrain/finetune epoch loop. Logs `<stage>/...` on the shared `step`
    and returns the updated step. Saves checkpoints to out_dir."""
    is_pre = stage == "pretrain"
    opt = torch.optim.AdamW(net.parameters(), lr=pcfg["lr"], betas=(0.9, 0.99))
    chan_drop = pcfg.get("channel_dropout", 0.0)
    ema_decay = pcfg.get("ema_decay", 0.999)
    # Sagittal-mirror augmentation; source and target are flipped together.
    flip_p = pcfg.get("flip_prob", 0.0)
    # inf = measure the gradient norm without clipping (it is logged either way).
    grad_clip = pcfg.get("grad_clip", 0) or float("inf")
    probe_ssim = pcfg.get("probe_endpoint_ssim", True) and not is_pre
    # Finetune only: image-space source->target L2 on the endpoint estimate, added on
    # top of the velocity loss. Off (weight 0) until `endpoint_l2_delay_epochs` have
    # passed — delay 5 => active from epoch 6 onward.
    l2_weight = pcfg.get("endpoint_l2_weight", 0.0) if not is_pre else 0.0
    l2_delay = pcfg.get("endpoint_l2_delay_epochs", 0)
    # Finetune only: 1 - SSIM on the same endpoint estimate, gated the same way.
    ssim_weight = pcfg.get("ssim_weight", 0.0) if not is_pre else 0.0
    ssim_delay = pcfg.get("ssim_delay_epochs", 0)
    # Pretrain x0: None = pure-noise bridge; else corrupt the target and bridge back.
    degrader = build_degrader(pcfg.get("corruption")) if is_pre else None
    # bf16 on Ampere+, fp16 on older cards (Turing needs it for tensor cores).
    dt = amp_dtype(device) if amp_enabled else torch.bfloat16
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled and dt == torch.float16)

    for epoch in range(pcfg["epochs"]):
        net.train()
        cur_l2_w = l2_weight if epoch >= l2_delay else 0.0  # gated by the epoch delay
        cur_ssim_w = ssim_weight if epoch >= ssim_delay else 0.0
        run_loss = None
        loss_sum, loss_n = 0.0, 0
        part_sums = {}
        n_slices, t_epoch = 0, time.time()
        pbar = tqdm(loader, desc=f"{stage} epoch {epoch}")
        max_iters = pcfg.get("max_iters")       # smoke test only; None in a real run
        for it, batch in enumerate(pbar):
            if max_iters and it >= max_iters:
                print(f"[smoke test] stopping epoch after {max_iters} iterations")
                break
            with torch.autocast(device_type=device.type, dtype=dt,
                                enabled=amp_enabled):
                if is_pre:
                    x1, _ = pad_to_multiple(batch["image"].to(device, non_blocking=True))
                    (x1,) = random_lr_flip(flip_p, x1)
                    mask = batch["mask"].to(device, non_blocking=True) if "mask" in batch else None
                    tgt_idx = field_idx_tensor(batch["field_strength"], device)
                    if degrader is None:
                        loss, parts = noise_bridge_loss(net, x1, tgt_field_idx=tgt_idx,
                                                        chan_mask=mask)
                    else:
                        x0 = degrader(x1)
                        if mask is not None:
                            x0 = x0 * mask[:, :, None, None]
                        loss, parts = bridge_loss(net, x0, x1, tgt_field_idx=tgt_idx,
                                                  chan_mask=mask)
                else:
                    x0, _ = pad_to_multiple(batch["source"].to(device, non_blocking=True))
                    x1, _ = pad_to_multiple(batch["target"].to(device, non_blocking=True))
                    x0, x1 = random_lr_flip(flip_p, x0, x1)   # one decision for the pair
                    x0 = drop_source_channels(x0, chan_drop)
                    src_idx = field_idx_tensor(batch["source_field"], device)
                    tgt_idx = field_idx_tensor(batch["target_field"], device)
                    loss, parts = bridge_loss(net, x0, x1, src_field_idx=src_idx,
                                              tgt_field_idx=tgt_idx,
                                              endpoint_l2_weight=cur_l2_w,
                                              ssim_weight=cur_ssim_w, probe_ssim=probe_ssim)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            # unscale before measuring so grad_norm is in real units, not scaler units
            # (scaler.step would do this itself; doing it early changes nothing).
            scaler.unscale_(opt)
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
                       f"{stage}/grad_norm": gnorm, f"{stage}/lr": opt.param_groups[0]["lr"]}
                log.update({f"{stage}/loss_{k}": float(v) for k, v in parts.items()})
                if not is_pre:
                    log[f"{stage}/endpoint_l2_w"] = cur_l2_w
                    log[f"{stage}/ssim_w"] = cur_ssim_w
                wb.log(log, step=step)

        dt_epoch = time.time() - t_epoch
        if wb is not None:
            log = {f"{stage}/epoch_loss": loss_sum / max(loss_n, 1),
                   f"{stage}/epoch": epoch,
                   f"{stage}/epoch_seconds": dt_epoch,
                   f"{stage}/slices_per_sec": n_slices / max(dt_epoch, 1e-6)}
            log.update({f"{stage}/epoch_{k}": v / max(loss_n, 1) for k, v in part_sums.items()})
            wb.log(log, step=step)

        if (epoch + 1) % pcfg.get("save_every", 20) == 0 or epoch + 1 == pcfg["epochs"]:
            ckpt = out_dir / f"checkpoint_epoch{epoch + 1}.pth"
            torch.save({"generator": raw_of(net).state_dict(), "ema": ema.state_dict(),
                        "config": cfg, "epoch": epoch + 1}, ckpt)
            print(f"saved {ckpt}")
    return step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stages", nargs="+", default=["pretrain", "finetune", "refine_gan"],
                    choices=["pretrain", "finetune", "refine_gan"],
                    help="stages to run in ONE wandb run, in order")
    ap.add_argument("--init_from", default=None,
                    help="checkpoint to warm-start the FIRST stage (else random / prior stage EMA)")
    ap.add_argument("--h5", default=None, help="HDF5 from build_h5.py")
    ap.add_argument("--task_name", default=None,
                    help="override config task_name (run name + checkpoint subfolder)")
    ap.add_argument("--pretrain_noise", action="store_true",
                    help="force the pure-noise pretrain (ignore pretrain.corruption)")
    ap.add_argument("--max_iters", type=int, default=None,
                    help="smoke test only: stop each epoch after N iterations. "
                         "Never set for a real run.")
    args = ap.parse_args()

    load_env()
    cfg = load_config(args.config)
    if args.task_name:
        cfg["task_name"] = args.task_name
    if args.pretrain_noise and cfg.get("pretrain"):
        cfg["pretrain"]["corruption"] = {"mode": "noise"}   # -> build_degrader returns None
    data_cfg, mcfg = cfg["data"], cfg["model"]
    if not (data_cfg.get("all_pairs") and data_cfg.get("multi_contrast")):
        raise SystemExit("Task 3 config required: set data.all_pairs and data.multi_contrast")
    device = torch.device(get_device() if torch.cuda.is_available() else "cpu")

    # Detect the GPU and scale batch sizes to its VRAM. A ~48 GB card fits double
    # the batches a ~20/24 GB card does, so double every stage's batch_size when we
    # land on the big card. Threshold at 32 GB splits 24 GB from 48 GB cleanly.
    gpu_info = {"name": "cpu", "vram_gb": 0.0, "batch_scale": 1}
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        vram_gb = props.total_memory / 1024**3
        batch_scale = 2 if vram_gb > 32 else 1
        gpu_info = {"name": props.name, "vram_gb": vram_gb, "batch_scale": batch_scale}
        print(f"[gpu] {props.name} | {vram_gb:.1f} GB VRAM | batch_scale x{batch_scale}")
        if batch_scale > 1:
            for blk in ("pretrain", "finetune", "finetune_refine_gan"):
                bs = cfg.get(blk, {}).get("batch_size")
                if bs is not None:
                    cfg[blk]["batch_size"] = bs * batch_scale
                    print(f"[gpu] {blk}.batch_size {bs} -> {cfg[blk]['batch_size']}")

    # Throughput toggles (config `runtime:` block).
    rt = cfg.get("runtime", {})
    if rt.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True
    if rt.get("tf32", False):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    amp_enabled = rt.get("amp", False) and device.type == "cuda"

    data_root = None if args.h5 else get_data_dir()
    task_name = cfg.get("task_name") or "task3_any_to_any"
    # Every run saves under its own {run}_{timestamp} subfolder so a new run never
    # overwrites an older run's checkpoints. Shared across this run's stages.
    run_tag = f"{task_name}_{time.strftime('%Y%m%d_%H%M%S')}"

    net = build_flow_net(mcfg).to(device)
    gen_params = sum(p.numel() for p in net.parameters())
    print(f"[model] {type(net).__name__} generator: {gen_params / 1e6:.2f}M params")
    if rt.get("compile", False):
        try:
            net = torch.compile(net)
            print("torch.compile enabled")
        except Exception as e:  # noqa: BLE001
            print(f"torch.compile unavailable ({e}); continuing uncompiled")

    # ONE wandb run for the whole pipeline; the merged config + raw yaml files are saved.
    wb = None
    if os.environ.get("WANDB_PROJECT"):
        import wandb
        wb = wandb.init(project=os.environ["WANDB_PROJECT"],
                        group=os.environ.get("WANDB_GROUP", task_name),
                        name=task_name, job_type="train", config=cfg,
                        reinit="create_new", save_code=True)
        wb.summary["params/generator_M"] = gen_params / 1e6
        wb.summary["gpu/name"] = gpu_info["name"]
        wb.summary["gpu/vram_gb"] = gpu_info["vram_gb"]
        wb.summary["gpu/batch_scale"] = gpu_info["batch_scale"]
        wb.config.update({"gpu": gpu_info}, allow_val_change=True)
        cfg_path = Path(args.config)
        for f in [cfg_path, cfg_path.parent / cfg.get("network", "")]:
            if f.name and f.is_file():
                wb.save(str(f), base_path=str(f.parent), policy="now")
        # Snapshot the whole codebase into the run so each run records the exact code
        # it ran. Zip it and sync via wb.save (robust: log_code's artifact path can
        # hit auth errors on some setups). Skip heavy/irrelevant dirs.
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

    # warm = state_dict the next stage starts from (--init_from, then each stage's EMA).
    warm = None
    if args.init_from:
        st = torch.load(args.init_from, map_location=device, weights_only=False)
        warm = st.get("ema", st.get("generator", st))
        print(f"warm-starting first stage ({args.stages[0]}) from {args.init_from}")
    else:
        print(f"NO --init_from: first stage ({args.stages[0]}) starts from RANDOM init")

    step = 0
    for stage in args.stages:
        pcfg = stage_pcfg(cfg, stage)
        if args.max_iters:                      # smoke-test truncation, see --max_iters
            pcfg = {**pcfg, "max_iters": args.max_iters}
        if warm is not None:
            raw_of(net).load_state_dict(warm)
        ema = fresh_ema(net)
        loader, n = make_loader(stage, pcfg, args.h5, data_root, rt)
        out_dir = Path(get_output_dir()) / STAGE_DIR[stage] / run_tag
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n==================== [{stage}] slices: {n} | step: {step} ====================")
        print(f"[ckpt] saving to {out_dir}")

        if stage == "refine_gan":
            step = train_refine_gan(net, ema, loader, device, pcfg, mcfg, cfg, out_dir,
                                    all_pairs=True, multi=True,
                                    chan_drop=pcfg.get("channel_dropout", 0.0),
                                    amp_enabled=amp_enabled, wb=wb, step=step)
        else:
            step = train_bridge(stage, net, ema, loader, device, pcfg, amp_enabled,
                                out_dir, cfg, wb, step)
        warm = copy.deepcopy(ema.state_dict())  # next stage continues from this EMA

    if wb is not None:
        wb.finish()


if __name__ == "__main__":
    main()
