"""torchvision-free LPIPS(alex).

This env runs a torch newer than its torchvision (ABI mismatch), so `import
torchvision` — and therefore the `lpips` package that imports it — crashes with
``operator torchvision::nms does not exist``. LPIPS(alex) only needs
AlexNet.features, whose ImageNet weights are already cached on disk, so we inject
a pure-torch AlexNet as ``torchvision.models.alexnet`` before importing lpips. The
real lpips forward / calibration layers then run unchanged.

If a working torchvision is present we do nothing and use the real package.

    from src.utils.lpips_shim import load_lpips
    fn = load_lpips(device="cuda")          # LPIPS module, .eval()
    d = fn(pred_bchw, tgt_bchw)             # inputs in [-1, 1], 3 channels
"""
from __future__ import annotations

import os
import sys
import types

import torch
import torch.nn as nn

_ALEX_CACHE = os.path.expanduser(
    "~/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth")


def _alexnet(pretrained=True, weights=None, **_kw):
    """Pure-torch AlexNet exposing ``.features`` (all lpips touches), with the
    cached ImageNet weights loaded."""
    feats = nn.Sequential(
        nn.Conv2d(3, 64, 11, 4, 2), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2),
        nn.Conv2d(64, 192, 5, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2),
        nn.Conv2d(192, 384, 3, padding=1), nn.ReLU(inplace=True),
        nn.Conv2d(384, 256, 3, padding=1), nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2),
    )
    if pretrained:
        if not os.path.exists(_ALEX_CACHE):
            raise FileNotFoundError(
                f"cached AlexNet weights not found at {_ALEX_CACHE}; a working "
                "torchvision is needed to download them once.")
        sd = torch.load(_ALEX_CACHE, map_location="cpu", weights_only=True)
        feats.load_state_dict({k[len("features."):]: v for k, v in sd.items()
                               if k.startswith("features.")})
    m = nn.Module()
    m.features = feats
    return m


def _install_fake_torchvision():
    try:
        import torchvision  # noqa: F401  (real one works -> nothing to do)
        return
    except Exception:  # noqa: BLE001 - any import failure -> inject the shim
        pass
    tv = types.ModuleType("torchvision")
    tvm = types.ModuleType("torchvision.models")
    tvm.alexnet = _alexnet
    tv.models = tvm
    sys.modules["torchvision"] = tv
    sys.modules["torchvision.models"] = tvm


def load_lpips(net="alex", device="cpu"):
    """Return an ``lpips.LPIPS`` module in eval mode. Only ``net='alex'`` is
    guaranteed when torchvision is broken (the only cached backbone)."""
    _install_fake_torchvision()
    if net != "alex" and "torchvision" in sys.modules and \
            isinstance(sys.modules["torchvision"], types.ModuleType) and \
            not hasattr(sys.modules["torchvision"], "__version__"):
        raise ValueError(f"net={net!r} needs a working torchvision; only 'alex' "
                         "is available through the shim.")
    import lpips
    return lpips.LPIPS(net=net).to(device).eval()
