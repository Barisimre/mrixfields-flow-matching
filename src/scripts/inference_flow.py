"""Flow-matching inference: integrate the ODE bridge slice-by-slice over each
source volume to produce the target field, for a given (--source, --target) pair.
Output is restored to native NIfTI orientation and brain-masked.

Usage:
    python -m src.scripts.inference_flow --config <cfg> --source 0.1T --target 7T \
        --checkpoint <ckpt> --input_dir <val>/ --output_dir <out>/
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from tqdm import tqdm

from src.data.utils import (
    FIELD_TO_DOMAIN,
    MODALITIES,
    extract_subject_id,
    load_nifti,
    save_nifti,
)
from src.models.builder import build_flow_net
from src.models.flow_matching import pad_to_multiple, sample, unpad
from src.utils.config import load_config
from src.utils.env import get_device, get_inference_dir, load_env
from src.utils.zclip_constants import Z_CLIP_RANGE


def load_net(cfg, ckpt_path, device):
    mcfg = cfg["model"]
    net = build_flow_net(mcfg)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    net.load_state_dict(state.get("ema", state.get("generator", state)))
    return net.to(device).eval()


def _sl(ndim, axis, i):
    s = [slice(None)] * ndim
    s[axis] = i
    return tuple(s)


def _field_idx(val, b, device):
    """Broadcast a single field-domain index over a slice-batch (or None)."""
    return None if val is None else torch.full((b,), val, dtype=torch.long, device=device)


def _slice_indices(n, z_range):
    """Slice indices to integrate. ``z_range`` (z0, z1) restricts to the
    evaluated axial slab; None runs the whole volume. Slices outside the range
    stay zero in the output (the submission only reads the slab, so the rest is
    never scored)."""
    if z_range is None:
        return list(range(n))
    z0, z1 = z_range
    return list(range(max(0, z0), min(n, z1)))


def _autocast(device):
    return torch.autocast(device_type=device.type, dtype=torch.float16,
                          enabled=device.type == "cuda")


@torch.no_grad()
def predict_volume(net, volume, steps, device, slice_axis=2, batch_size=32,
                   src_field_idx=None, tgt_field_idx=None, z_range=None):
    """Single-contrast. Slices are independent, so they run in batches of
    ``batch_size`` through the ODE (same result as one-at-a-time, far faster)."""
    out = np.zeros_like(volume, dtype=np.float32)
    idx_all = _slice_indices(volume.shape[slice_axis], z_range)
    for start in range(0, len(idx_all), batch_size):
        idxs = idx_all[start:start + batch_size]
        slabs = [torch.from_numpy(volume[_sl(volume.ndim, slice_axis, i)] * 2.0 - 1.0).float()
                 for i in idxs]
        x0 = torch.stack(slabs)[:, None].to(device)  # [B, 1, H, W]
        x0, pad = pad_to_multiple(x0)
        b = x0.size(0)
        with _autocast(device):
            x1 = sample(net, x0, src_field_idx=_field_idx(src_field_idx, b, device),
                        tgt_field_idx=_field_idx(tgt_field_idx, b, device), steps=steps)
        x1 = unpad(x1, pad).float().cpu().numpy()  # [B, 1, H, W]
        for bi, i in enumerate(idxs):
            out[_sl(volume.ndim, slice_axis, i)] = np.clip(x1[bi, 0], -1, 1) * 0.5 + 0.5
    return out


@torch.no_grad()
def predict_volume_mc(net, volumes, steps, device, slice_axis=2, batch_size=32,
                      src_field_idx=None, tgt_field_idx=None, z_range=None):
    """Multi-contrast: ``volumes`` is a list of 3 source volumes (same grid),
    returns a list of 3 predicted target volumes. Slices run in batches."""
    outs = [np.zeros_like(v, dtype=np.float32) for v in volumes]
    ndim = volumes[0].ndim
    n = min(v.shape[slice_axis] for v in volumes)
    idx_all = _slice_indices(n, z_range)
    for start in range(0, len(idx_all), batch_size):
        idxs = idx_all[start:start + batch_size]
        stacks = [torch.stack([torch.from_numpy(v[_sl(ndim, slice_axis, i)] * 2.0 - 1.0).float()
                               for v in volumes]) for i in idxs]
        x0 = torch.stack(stacks).to(device)  # [B, C, H, W]
        x0, pad = pad_to_multiple(x0)
        b = x0.size(0)
        with _autocast(device):
            x1 = sample(net, x0, src_field_idx=_field_idx(src_field_idx, b, device),
                        tgt_field_idx=_field_idx(tgt_field_idx, b, device), steps=steps)
        x1 = unpad(x1, pad).float().cpu().numpy()  # [B, C, H, W]
        for bi, i in enumerate(idxs):
            sl = _sl(ndim, slice_axis, i)
            for c in range(len(volumes)):
                outs[c][sl] = np.clip(x1[bi, c], -1, 1) * 0.5 + 0.5
    return outs


def restore_and_save(pred, src_path, out_path):
    """Restore native orientation, mask by the source brain, write NIfTI."""
    orig = nib.load(str(src_path))
    _, can_affine = load_nifti(src_path)
    tf = nib.orientations.ornt_transform(
        nib.io_orientation(can_affine), nib.io_orientation(orig.affine))
    pred = nib.orientations.apply_orientation(pred, tf)
    pred = pred * (orig.get_fdata(dtype=np.float32) > 1e-6)
    save_nifti(pred, orig.affine, out_path, header=orig.header)


def _cond_indices(cfg):
    """(src_field_idx, tgt_field_idx) when the net is field-conditioned, else (None, None)."""
    if not cfg["model"].get("num_fields", 0):
        return None, None
    return (FIELD_TO_DOMAIN[cfg["data"]["source_field"]],
            FIELD_TO_DOMAIN[cfg["data"]["target_field"]])


def run_single(net, cfg, in_dir, out_dir, steps, device, slice_axis, batch_size, z_range):
    src_idx, tgt_idx = _cond_indices(cfg)
    src, tgt = cfg["data"]["source_field"], cfg["data"]["target_field"]
    files = sorted(in_dir.rglob("*.nii.gz"))
    for path in tqdm(files, desc=f"{src}->{tgt}"):
        data, _ = load_nifti(path)
        pred = predict_volume(net, data, steps, device, slice_axis, batch_size,
                              src_field_idx=src_idx, tgt_field_idx=tgt_idx, z_range=z_range)
        restore_and_save(pred, path, out_dir / path.relative_to(in_dir))


def run_multi(net, cfg, in_dir, out_dir, steps, device, slice_axis, batch_size, z_range,
              subject_ids=None):
    """in_dir is the split root holding <modality>/<source_field>/*.nii.gz.

    For each subject with all 3 source contrasts, predict the 3 target contrasts
    jointly and save each under out_dir/<modality>/ (source filename preserved,
    matching the single-contrast layout). ``subject_ids`` restricts to the
    released inputs for this source field (the data dir also ships held-out GT
    subjects at non-source fields)."""
    src_idx, tgt_idx = _cond_indices(cfg)
    src, tgt = cfg["data"]["source_field"], cfg["data"]["target_field"]
    by_mod = {m: {extract_subject_id(p.name): p
                  for p in sorted((in_dir / m / src).glob("*.nii.gz"))}
              for m in MODALITIES}
    subjects = sorted(set.intersection(*[set(by_mod[m]) for m in MODALITIES]), key=str)
    if subject_ids is not None:
        # extract_subject_id yields e.g. "P_0001"; --subject_ids are bare "0001".
        subjects = [s for s in subjects if s.split("_")[-1] in subject_ids]
    for sid in tqdm(subjects, desc=f"{src}->{tgt}"):
        paths = [by_mod[m][sid] for m in MODALITIES]
        vols = [load_nifti(p)[0] for p in paths]
        preds = predict_volume_mc(net, vols, steps, device, slice_axis, batch_size,
                                  src_field_idx=src_idx, tgt_field_idx=tgt_idx, z_range=z_range)
        for m, path, pred in zip(MODALITIES, paths, preds, strict=True):
            restore_and_save(pred, path, out_dir / m / path.name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input_dir", required=True,
                    help="single: a <modality>/<field> dir; multi: the split root")
    load_env()
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32, help="slices per net call (more VRAM)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--source", default=None, help="source field, e.g. 0.1T (overrides config)")
    ap.add_argument("--target", default=None, help="target field, e.g. 3T (overrides config)")
    ap.add_argument("--subject_ids", default=None,
                    help="comma-separated subject IDs to infer (multi-contrast only); "
                         "default: every subject present at the source field")
    ap.add_argument("--full-volume", dest="full_volume", action="store_true",
                    help="integrate every axial slice; default only the evaluated "
                         f"slab Z_CLIP_RANGE={Z_CLIP_RANGE} (the rest is never scored)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.source:
        cfg["data"]["source_field"] = args.source
    if args.target:
        cfg["data"]["target_field"] = args.target
    device = torch.device((args.device or get_device()) if torch.cuda.is_available() else "cpu")
    net = load_net(cfg, args.checkpoint, device)
    slice_axis = cfg["data"].get("slice_axis", 2)

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir or get_inference_dir())
    out_dir.mkdir(parents=True, exist_ok=True)

    z_range = None if args.full_volume else Z_CLIP_RANGE
    sids = set(args.subject_ids.split(",")) if args.subject_ids else None

    if cfg["data"].get("multi_contrast", False):
        run_multi(net, cfg, in_dir, out_dir, args.steps, device, slice_axis, args.batch_size,
                  z_range, subject_ids=sids)
    else:
        run_single(net, cfg, in_dir, out_dir, args.steps, device, slice_axis, args.batch_size,
                   z_range)

    print(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
