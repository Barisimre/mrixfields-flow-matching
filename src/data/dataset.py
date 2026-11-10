"""PyTorch Dataset classes for the Task 3 any->any flow bridge.

Data structure:
    {data_root}/{split}/{modality}/{field_strength}/*.nii.gz

Output tensors are scaled to [-1, 1]. The 3 contrasts (T1W, T2W, T2FLAIR) are
stacked as channels in MODALITIES order.

    - MultiContrastPairedDataset   : registered source->target triplets (finetune)
    - MultiContrastUnpairedDataset : single-field target triplets + valid-channel
                                     mask (pretrain prior)
    - H5MultiContrast{Paired,Unpaired}Dataset : the same data read from the packed
                                     HDF5 (tools/build_h5.py), ~300x faster to index.

The NIfTI-backed classes keep every volume they touch in RAM (`cached_volume` below).
Read that comment before changing anything here: it is what makes reading the released
NIfTIs fast enough to train on, and it is deliberately float16 so the tensors it yields
are bit-identical to the h5-backed path.
"""

import os
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import Compose, NormalizeMinMax, ScaleToMinusOneOne, ToTensor
from .utils import MODALITIES, extract_subject_id, list_nifti_files, load_nifti

# ---------------------------------------------------------------------------------
# In-RAM volume cache -- read this before touching the NIfTI-backed datasets.
#
# WHY IT EXISTS. Reading a slice straight from a NIfTI is unusably slow -- `__getitem__`
# decompresses a whole 231 MB volume to take one slice, six times per sample -- so
# every volume is decompressed once, on first touch, and kept in memory.
#
# WHY float16, WHICH IS NOT A MEMORY TRICK. tools/build_h5.py stores float16 and the
# h5 reader casts back to float32 per slice. Caching float32 here would mean a
# NIfTI-backed run trains on numbers that differ from a packed-h5 run in the third
# decimal. Casting to float16 makes the cached array byte-identical to the h5 contents
# and the tensors bit-identical to
# H5MultiContrastPairedDataset's. Slice std (the sample filter) is still computed in
# float32, matching how build_h5.py computed the stored `slice_std`.
#
# WHAT IT COSTS. Task 3's paired split is 45 volumes (3 subjects x 5 fields x 3
# contrasts) at 364x436x364 float16 = ~5.2 GB, shared by all 20 directed-pair datasets
# through this one module-level dict and inherited copy-on-write by forked DataLoader
# workers. The unpaired (pretrain) split is far larger: 1939 volumes would be ~220 GB,
# so an unpaired run either has the RAM or sets MRIX_CACHE_VOLUMES=0 and pays the
# decompression per access.
#
# MRIX_CACHE_VOLUMES=0 disables caching. The float16 cast is NOT disabled with it:
# fidelity to the h5 is a correctness property, the cache is only speed.
# ---------------------------------------------------------------------------------

_VOLUME_CACHE: dict[str, np.ndarray] = {}
CACHE_VOLUMES = os.environ.get("MRIX_CACHE_VOLUMES", "1") != "0"


def cached_volume(path):
    """Volume at `path` as float16, decompressed once per process. See the note above."""
    key = str(Path(path).resolve())
    vol = _VOLUME_CACHE.get(key)
    if vol is None:
        vol = load_nifti(path)[0].astype(np.float16)
        if CACHE_VOLUMES:
            _VOLUME_CACHE[key] = vol
    return vol


def _slice_f32(volume, slice_axis, idx):
    """One slice as float32 -- the same dtype the h5 reader hands the transform."""
    sl = [slice(None)] * volume.ndim
    sl[slice_axis] = idx
    return np.asarray(volume[tuple(sl)], dtype=np.float32)


def _default_transform() -> Compose:
    """Normalize to [0,1], to tensor, scale to [-1,1] (the range the net expects)."""
    return Compose([NormalizeMinMax(), ToTensor(), ScaleToMinusOneOne()])


def _stack_channel(transform, slc2d):
    """Transform a 2D slice and drop the leading channel -> [H, W] tensor in [-1,1]."""
    return transform(slc2d).squeeze(0)


class MultiContrastPairedDataset(Dataset):
    """Paired source->target bridge with the 3 contrasts stacked as channels.

    Channel order is MODALITIES = [T1W, T2W, T2FLAIR]. Only subjects that have
    all three contrasts at BOTH source and target field are used (registered
    triplets). Returns ``source``/``target`` tensors of shape [3, H, W].
    """

    def __init__(self, data_root, split, source_field, target_field,
                 transform=None, slice_axis=2, min_slice_std=0.01):
        self.data_root = Path(data_root)
        self.split = split
        self.source_field = source_field
        self.target_field = target_field
        self.slice_axis = slice_axis
        self.min_slice_std = min_slice_std
        self.transform = transform or _default_transform()

        # src_paths[sid][modality] -> Path ; same for tgt
        self.src_paths: dict[str, dict[str, Path]] = {}
        self.tgt_paths: dict[str, dict[str, Path]] = {}
        self.samples: list[tuple[str, int]] = []
        self._index_data()

    def _by_id(self, field):
        return {m: {extract_subject_id(p.name): p
                    for p in list_nifti_files(self.data_root, self.split, m, field)}
                for m in MODALITIES}

    def _index_data(self):
        src, tgt = self._by_id(self.source_field), self._by_id(self.target_field)
        common = set(src[MODALITIES[0]])
        for m in MODALITIES:
            common &= set(src[m].keys()) & set(tgt[m].keys())

        for sid in sorted(common):
            self.src_paths[sid] = {m: src[m][sid] for m in MODALITIES}
            self.tgt_paths[sid] = {m: tgt[m][sid] for m in MODALITIES}
            src_vols = [cached_volume(self.src_paths[sid][m]) for m in MODALITIES]
            tgt_vols = [cached_volume(self.tgt_paths[sid][m]) for m in MODALITIES]
            n = min(v.shape[self.slice_axis] for v in src_vols + tgt_vols)
            for i in range(n):
                # keep if any source contrast AND any target contrast has signal
                # (max per-channel std, matching the h5-backed dataset / stored stat).
                # float32 on purpose: build_h5.py computed the stored slice_std from
                # the float32 volume, so the filter must too or the sample sets drift.
                s_std = max(_slice_f32(v, self.slice_axis, i).std() for v in src_vols)
                t_std = max(_slice_f32(v, self.slice_axis, i).std() for v in tgt_vols)
                if s_std > self.min_slice_std and t_std > self.min_slice_std:
                    self.samples.append((sid, i))

    def _get_slice(self, volume, idx):
        sl = [slice(None)] * volume.ndim
        sl[self.slice_axis] = idx
        return volume[tuple(sl)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sid, i = self.samples[index]
        src = torch.stack([_stack_channel(self.transform,
                          _slice_f32(cached_volume(self.src_paths[sid][m]), self.slice_axis, i))
                          for m in MODALITIES])
        tgt = torch.stack([_stack_channel(self.transform,
                          _slice_f32(cached_volume(self.tgt_paths[sid][m]), self.slice_axis, i))
                          for m in MODALITIES])
        return {"source": src, "target": tgt, "subject": sid, "slice_idx": i,
                "source_field": self.source_field, "target_field": self.target_field}


class MultiContrastUnpairedDataset(Dataset):
    """Single-field target images with the 3 contrasts stacked as channels.

    Used to pretrain the multi-contrast target prior. Subjects with only some
    contrasts (e.g. retrospective 5T/7T, where the contrasts come from different
    subjects) are still used: missing channels are zeroed and a ``mask`` of shape
    [3] marks which channels are valid so the loss can ignore the rest.

    Returns ``image`` [3, H, W] and ``mask`` [3] (1.0 where the channel is real).
    """

    def __init__(self, data_root, split, field_strength,
                 transform=None, slice_axis=2, min_slice_std=0.01):
        self.data_root = Path(data_root)
        self.split = split
        self.field_strength = field_strength
        self.slice_axis = slice_axis
        self.min_slice_std = min_slice_std
        self.transform = transform or _default_transform()

        self.paths: dict[str, dict[str, Path]] = {}   # sid -> {modality: Path}
        self.samples: list[tuple[str, int]] = []
        self._index_data()

    def _index_data(self):
        by_mod = {m: {extract_subject_id(p.name): p
                      for p in list_nifti_files(self.data_root, self.split, m, self.field_strength)}
                  for m in MODALITIES}
        subjects = set().union(*[set(by_mod[m]) for m in MODALITIES])

        for sid in sorted(subjects):
            present = {m: by_mod[m][sid] for m in MODALITIES if sid in by_mod[m]}
            self.paths[sid] = present
            vols = [cached_volume(p) for p in present.values()]
            n = min(v.shape[self.slice_axis] for v in vols)
            for i in range(n):
                # max per-channel std over present contrasts (matches the h5 path)
                # float32 std, as in the paired class -- see the cache note.
                if max(_slice_f32(v, self.slice_axis, i).std()
                       for v in vols) > self.min_slice_std:
                    self.samples.append((sid, i))

    def _get_slice(self, volume, idx):
        sl = [slice(None)] * volume.ndim
        sl[self.slice_axis] = idx
        return volume[tuple(sl)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sid, i = self.samples[index]
        present = self.paths[sid]
        chans, mask = [], []
        ref = None
        for m in MODALITIES:
            if m in present:
                ch = _stack_channel(self.transform,
                                    _slice_f32(cached_volume(present[m]), self.slice_axis, i))
                ref = ch
                chans.append(ch)
                mask.append(1.0)
            else:
                chans.append(None)  # fill after we know the shape
                mask.append(0.0)
        chans = [torch.zeros_like(ref) if c is None else c for c in chans]
        return {"image": torch.stack(chans), "mask": torch.tensor(mask),
                "subject": sid, "slice_idx": i, "field_strength": self.field_strength}


class _H5Mixin:
    """Lazy, fork-safe HDF5 handle (opened per worker on first access)."""

    def _h5(self):
        h = getattr(self, "_handle", None)
        if h is None:
            h = self._handle = h5py.File(self.h5_path, "r")
        return h

    def _group(self, h, split, mod, field):
        path = f"{split}/{mod}/{field}"
        return h[path] if path in h else None

    def _read_slice(self, key, i):
        ds = self._h5()[key]
        sl = np.asarray(ds[:, :, i], dtype=np.float32)
        scale = float(ds.attrs.get("scale", 1.0))
        return sl / scale if scale != 1.0 else sl


class H5MultiContrastPairedDataset(_H5Mixin, Dataset):
    """h5-backed MultiContrastPairedDataset: 3-channel source/target [3, H, W]."""

    def __init__(self, h5_path, split, source_field, target_field,
                 transform=None, slice_axis=2, min_slice_std=0.01):
        self.h5_path = str(h5_path)
        self.split = split
        self.source_field = source_field
        self.target_field = target_field
        self.slice_axis = slice_axis
        self.min_slice_std = min_slice_std
        self.transform = transform or _default_transform()
        self._handle = None
        self.samples: list[tuple[str, int]] = []
        self._index_data()
        self._handle = None  # reopened per worker

    def _index_data(self):
        h = self._h5()
        src_g = {m: self._group(h, self.split, m, self.source_field) for m in MODALITIES}
        tgt_g = {m: self._group(h, self.split, m, self.target_field) for m in MODALITIES}
        if any(g is None for g in {**src_g, **tgt_g}.values()):
            return
        common = set(src_g[MODALITIES[0]].keys())
        for m in MODALITIES:
            common &= set(src_g[m].keys()) & set(tgt_g[m].keys())
        for sid in sorted(common):
            src_std = np.stack([src_g[m][sid].attrs["slice_std"] for m in MODALITIES])
            tgt_std = np.stack([tgt_g[m][sid].attrs["slice_std"] for m in MODALITIES])
            n = min(src_std.shape[1], tgt_std.shape[1])
            for i in range(n):
                if src_std[:, i].max() > self.min_slice_std and \
                        tgt_std[:, i].max() > self.min_slice_std:
                    self.samples.append((sid, i))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sid, i = self.samples[index]
        src = torch.stack([_stack_channel(
            self.transform, self._read_slice(f"{self.split}/{m}/{self.source_field}/{sid}", i))
            for m in MODALITIES])
        tgt = torch.stack([_stack_channel(
            self.transform, self._read_slice(f"{self.split}/{m}/{self.target_field}/{sid}", i))
            for m in MODALITIES])
        return {"source": src, "target": tgt, "subject": sid, "slice_idx": i,
                "source_field": self.source_field, "target_field": self.target_field}


class H5MultiContrastUnpairedDataset(_H5Mixin, Dataset):
    """h5-backed MultiContrastUnpairedDataset: 3-channel target [3, H, W] + mask [3]."""

    def __init__(self, h5_path, split, field_strength,
                 transform=None, slice_axis=2, min_slice_std=0.01):
        self.h5_path = str(h5_path)
        self.split = split
        self.field_strength = field_strength
        self.slice_axis = slice_axis
        self.min_slice_std = min_slice_std
        self.transform = transform or _default_transform()
        self._handle = None
        self.samples: list[tuple[str, tuple[str, ...], int]] = []
        self._index_data()
        self._handle = None

    def _index_data(self):
        h = self._h5()
        groups = {m: self._group(h, self.split, m, self.field_strength) for m in MODALITIES}
        subjects = set().union(*[set(g.keys()) for g in groups.values() if g is not None])
        for sid in sorted(subjects):
            present = tuple(m for m in MODALITIES if groups[m] is not None and sid in groups[m])
            std = np.stack([groups[m][sid].attrs["slice_std"] for m in present])
            for i in range(std.shape[1]):
                if std[:, i].max() > self.min_slice_std:
                    self.samples.append((sid, present, i))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sid, present, i = self.samples[index]
        chans, mask, ref = [], [], None
        for m in MODALITIES:
            if m in present:
                key = f"{self.split}/{m}/{self.field_strength}/{sid}"
                ch = _stack_channel(self.transform, self._read_slice(key, i))
                ref = ch
                chans.append(ch)
                mask.append(1.0)
            else:
                chans.append(None)
                mask.append(0.0)
        chans = [torch.zeros_like(ref) if c is None else c for c in chans]
        return {"image": torch.stack(chans), "mask": torch.tensor(mask),
                "subject": sid, "slice_idx": i, "field_strength": self.field_strength}
