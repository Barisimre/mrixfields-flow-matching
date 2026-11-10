"""Full-volume NIfTI inference for a baseline checkpoint .

Mirrors `src/scripts/inference_flow.py` — slice-by-slice along axis 2, source brain
mask, native orientation/affine restored, `Z_CLIP_RANGE` slab by default — with the
flow `sample()` swapped for the method's `generate()`. The helpers that do not depend
on the integrator (`restore_and_save`, `_slice_indices`, `_field_idx`, `_sl`) are
imported from `inference_flow` rather than copied; that file is never edited.

    python -m src.baselines.infer_baseline --config configs/baseline_regression.yaml \
        --checkpoint <ckpt> --source 0.1T --target 7T --input_dir $VAL_DIR \
        --output_dir $INFERENCE_DIR/task3_baseline_regression/0.1T_to_7T/pred
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.baselines.diffusion import DiffusionMethod
from src.baselines.regression import RegressionMethod
from src.data.utils import FIELD_TO_DOMAIN, MODALITIES, extract_subject_id, load_nifti
from src.models.builder import build_flow_net
from src.models.flow_matching import pad_to_multiple, unpad
from src.scripts.inference_flow import _field_idx, _sl, _slice_indices, restore_and_save
from src.utils.config import load_config
from src.utils.env import get_device, get_inference_dir, load_env
from src.utils.zclip_constants import Z_CLIP_RANGE

METHODS = {"regression": RegressionMethod, "diffusion": DiffusionMethod}


def load_net(cfg, ckpt_path, device):
    net = build_flow_net(cfg["model"])
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    net.load_state_dict(state.get("ema", state.get("generator", state)))
    return net.to(device).eval()


def _autocast(device):
    return torch.autocast(device_type=device.type, dtype=torch.float16,
                          enabled=device.type == "cuda")


@torch.no_grad()
def predict_volume_mc(method, net, volumes, steps, device, slice_axis=2, batch_size=32,
                      src_field_idx=None, tgt_field_idx=None, z_range=None):
    """Multi-contrast: 3 source volumes on the same grid -> 3 predicted target volumes."""
    outs = [np.zeros_like(v, dtype=np.float32) for v in volumes]
    ndim = volumes[0].ndim
    n = min(v.shape[slice_axis] for v in volumes)
    idx_all = _slice_indices(n, z_range)
    for start in range(0, len(idx_all), batch_size):
        idxs = idx_all[start:start + batch_size]
        stacks = [torch.stack([torch.from_numpy(v[_sl(ndim, slice_axis, i)] * 2.0 - 1.0).float()
                               for v in volumes]) for i in idxs]
        x0 = torch.stack(stacks).to(device)          # [B, C, H, W]
        x0, pad = pad_to_multiple(x0)
        b = x0.size(0)
        with _autocast(device):
            x1 = method.generate(net, x0, _field_idx(src_field_idx, b, device),
                                 _field_idx(tgt_field_idx, b, device), steps)
        x1 = unpad(x1, pad).float().cpu().numpy()
        for bi, i in enumerate(idxs):
            sl = _sl(ndim, slice_axis, i)
            for c in range(len(volumes)):
                outs[c][sl] = np.clip(x1[bi, c], -1, 1) * 0.5 + 0.5
    return outs


def run_multi(method, net, cfg, in_dir, out_dir, steps, device, slice_axis, batch_size,
              z_range, subject_ids=None):
    """`in_dir` is the split root holding <modality>/<source_field>/*.nii.gz."""
    src, tgt = cfg["data"]["source_field"], cfg["data"]["target_field"]
    src_idx, tgt_idx = FIELD_TO_DOMAIN[src], FIELD_TO_DOMAIN[tgt]
    by_mod = {m: {extract_subject_id(p.name): p
                  for p in sorted((in_dir / m / src).glob("*.nii.gz"))}
              for m in MODALITIES}
    subjects = sorted(set.intersection(*[set(by_mod[m]) for m in MODALITIES]), key=str)
    if subject_ids is not None:
        subjects = [s for s in subjects if s.split("_")[-1] in subject_ids]
    for sid in tqdm(subjects, desc=f"{src}->{tgt}"):
        paths = [by_mod[m][sid] for m in MODALITIES]
        vols = [load_nifti(p)[0] for p in paths]
        preds = predict_volume_mc(method, net, vols, steps, device, slice_axis, batch_size,
                                  src_field_idx=src_idx, tgt_field_idx=tgt_idx,
                                  z_range=z_range)
        for m, path, pred in zip(MODALITIES, paths, preds, strict=True):
            restore_and_save(pred, path, out_dir / m / path.name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input_dir", required=True, help="the split root (<mod>/<field>/*.nii.gz)")
    load_env()
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--method", default=None, choices=sorted(METHODS))
    ap.add_argument("--steps", type=int, default=0,
                    help="sampler steps; 0 = the method default (1 for regression, "
                         "diffusion.sample_steps for diffusion)")
    ap.add_argument("--batch_size", type=int, default=32, help="slices per net call")
    ap.add_argument("--device", default=None)
    ap.add_argument("--source", required=True, help="source field, e.g. 0.1T")
    ap.add_argument("--target", required=True, help="target field, e.g. 7T")
    ap.add_argument("--subject_ids", default=None,
                    help="comma-separated subject IDs to infer; default: every subject "
                         "present at the source field")
    ap.add_argument("--full-volume", dest="full_volume", action="store_true",
                    help="integrate every axial slice; default only the evaluated slab "
                         f"Z_CLIP_RANGE={Z_CLIP_RANGE}")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["data"]["source_field"] = args.source
    cfg["data"]["target_field"] = args.target
    method_name = args.method or cfg.get("method")
    if method_name not in METHODS:
        raise SystemExit(f"--method / config `method:` must be one of {sorted(METHODS)}")
    method = METHODS[method_name](cfg)

    device = torch.device((args.device or get_device()) if torch.cuda.is_available() else "cpu")
    net = load_net(cfg, args.checkpoint, device)

    out_dir = Path(args.output_dir or get_inference_dir())
    out_dir.mkdir(parents=True, exist_ok=True)
    z_range = None if args.full_volume else Z_CLIP_RANGE
    sids = set(args.subject_ids.split(",")) if args.subject_ids else None

    run_multi(method, net, cfg, Path(args.input_dir), out_dir, args.steps or None, device,
              cfg["data"].get("slice_axis", 2), args.batch_size, z_range, subject_ids=sids)
    print(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
