"""Preprocessing transforms for MRI volumes and 2D slices."""


import numpy as np
import torch


class NormalizeMinMax:
    """Min-max normalize a volume to [0, 1], with optional percentile clipping.

    If `clip_percentile` is set (e.g. 99.5), values are first clipped to that
    percentile then min-max rescaled. If `clip_percentile` is None (default),
    the volume is returned as float32 unchanged.
    """

    def __init__(self, clip_percentile: float | None = None):
        self.clip_percentile = clip_percentile

    def __call__(self, data: np.ndarray) -> np.ndarray:
        data = data.astype(np.float32)
        if self.clip_percentile is None:
            return data
        upper = np.percentile(data, self.clip_percentile)
        data = np.clip(data, 0, upper)
        dmin, dmax = data.min(), data.max()
        if dmax - dmin > 1e-8:
            data = (data - dmin) / (dmax - dmin)
        return data


class ToTensor:
    """Convert numpy array to PyTorch tensor, adding channel dimension."""

    def __call__(self, data: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(data.copy()).unsqueeze(0).float()


class ScaleToMinusOneOne:
    """Scale tensor from [0, 1] to [-1, 1]: x * 2 - 1.

    The [-1, 1] range the flow net computes in end to end.
    """

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return data * 2.0 - 1.0


class Compose:
    """Chain multiple transforms together."""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, data: np.ndarray) -> np.ndarray | torch.Tensor:
        for t in self.transforms:
            data = t(data)
        return data
