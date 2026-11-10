"""Train a paper baseline (regression or SDEdit diffusion) — pretrain then finetune.

Mirrors `src/scripts/train_flow.py`: one process, one wandb run, stage-namespaced
metrics on a shared step, the model flowing pretrain -> finetune in memory via the
previous stage's EMA. The difference is that each stage runs to a **wall-clock
budget** (`max_hours`) rather than a fixed epoch count, and the objective comes from
`src/baselines/{regression,diffusion}.py` instead of the flow bridge.

    python -m src.baselines.train_baseline --config configs/baseline_regression.yaml \
        --method regression --stages pretrain finetune --h5 $H5_PATH
"""

import argparse
import copy
import time
from pathlib import Path

import torch

from src.baselines.common import (
    apply_gpu_batch_scale,
    fresh_ema,
    init_wandb,
    make_loader,
    raw_of,
    setup_runtime,
    stage_dir,
    train_stage,
)
from src.baselines.diffusion import DiffusionMethod
from src.baselines.regression import RegressionMethod
from src.models.builder import build_flow_net
from src.utils.config import load_config
from src.utils.env import get_data_dir, get_device, get_output_dir, load_env

METHODS = {"regression": RegressionMethod, "diffusion": DiffusionMethod}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--method", default=None, choices=sorted(METHODS),
                    help="overrides the config `method:` key")
    ap.add_argument("--stages", nargs="+", default=["pretrain", "finetune"],
                    choices=["pretrain", "finetune"],
                    help="stages to run in ONE wandb run, in order")
    ap.add_argument("--init_from", default=None,
                    help="checkpoint to warm-start the FIRST stage (else random init)")
    ap.add_argument("--h5", default=None, help="HDF5 from build_h5.py")
    ap.add_argument("--task_name", default=None,
                    help="override config task_name (run name + checkpoint subfolder)")
    args = ap.parse_args()

    load_env()
    cfg = load_config(args.config)
    method_name = args.method or cfg.get("method")
    if method_name not in METHODS:
        raise SystemExit(f"--method / config `method:` must be one of {sorted(METHODS)}")
    cfg["method"] = method_name
    if args.task_name:
        cfg["task_name"] = args.task_name
    data_cfg = cfg["data"]
    if not (data_cfg.get("all_pairs") and data_cfg.get("multi_contrast")):
        raise SystemExit("Task 3 config required: set data.all_pairs and data.multi_contrast")

    device = torch.device(get_device() if torch.cuda.is_available() else "cpu")
    gpu_info = apply_gpu_batch_scale(cfg, device)
    amp_enabled = setup_runtime(cfg, device)
    rt = cfg.get("runtime", {})

    method = METHODS[method_name](cfg)
    data_root = None if args.h5 else get_data_dir()
    task_name = cfg.get("task_name") or f"baseline_{method_name}"
    run_tag = f"{task_name}_{time.strftime('%Y%m%d_%H%M%S')}"

    net = build_flow_net(cfg["model"]).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"[model] {type(net).__name__} ({method_name}): {n_params / 1e6:.2f}M params")
    if rt.get("compile", False):
        try:
            net = torch.compile(net)
            print("torch.compile enabled")
        except Exception as e:  # noqa: BLE001
            print(f"torch.compile unavailable ({e}); continuing uncompiled")

    wb = init_wandb(cfg, args.config, task_name, gpu_info, n_params)

    warm = None
    if args.init_from:
        st = torch.load(args.init_from, map_location=device, weights_only=False)
        warm = st.get("ema", st.get("generator", st))
        print(f"warm-starting first stage ({args.stages[0]}) from {args.init_from}")
    else:
        print(f"NO --init_from: first stage ({args.stages[0]}) starts from RANDOM init")

    step = 0
    for stage in args.stages:
        pcfg = cfg[stage]
        if warm is not None:
            raw_of(net).load_state_dict(warm)
        ema = fresh_ema(net)
        loader, n = make_loader(stage, pcfg, args.h5, data_root, rt)
        out_dir = Path(get_output_dir()) / stage_dir(method_name) / run_tag / stage
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n============ [{method_name}/{stage}] slices: {n} | step: {step} ============")
        print(f"[ckpt] saving to {out_dir}")
        step = train_stage(stage, method, net, ema, loader, device, pcfg, amp_enabled,
                           out_dir, cfg, wb, step)
        warm = copy.deepcopy(ema.state_dict())   # next stage continues from this EMA

    if wb is not None:
        wb.finish()


if __name__ == "__main__":
    main()
