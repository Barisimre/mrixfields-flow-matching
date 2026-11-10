"""Pack the whole dataset into one HDF5 file for fast per-slice reads.

The training datasets read one axial slice at a time, but a `.nii.gz` must be
fully gunzipped (~900 MB) to reach any slice. This packs every volume into a
chunked HDF5 dataset (one chunk per axial slice) so a slice read decompresses
only that chunk, with the fast `lzf` codec.

Layout (slice_axis=2, native (X, Y, Z) shape preserved):
    /{split}/{modality}/{field}/{subject_id}   float16, chunks (X, Y, 1), lzf
        .attrs["slice_std"]  float32[Z]  per-slice std (skip empty slices w/o reading pixels)
        .attrs["filename"]   original .nii.gz name

Volumes are stored raw (range left as-is); the dataset transform still does the
per-slice NormalizeMinMax -> [-1,1]. Empty (background) slices are kept so z
indices stay aligned across contrasts/fields, and they compress to almost
nothing under lzf.

Usage (defaults: --data_root ./data, --out $H5_PATH from .env):
    python -m tools.build_h5
    python -m tools.build_h5 --data_root data --out /path/to/mrix.h5
"""

import argparse
import os
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from src.data.utils import (
    FIELD_STRENGTHS,
    MODALITIES,
    extract_subject_id,
    list_nifti_files,
    load_nifti,
)
from src.utils.env import load_env

# The three released splits:
#   Training_retrospective  — unpaired, one field per subject, disjoint IDs per
#                             field, some contrasts missing (esp. 5T/7T).
#   Training_prospective    — 3 paired subjects at all 5 fields.
#   Validating_prospective  — inputs only, one field per subject, IDs differ per field.
DEFAULT_SPLITS = ["Training_retrospective", "Training_prospective", "Validating_prospective"]


def main():
    load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--out", default=os.environ.get("H5_PATH") or None)
    ap.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32", "uint16"])
    ap.add_argument("--compression", default="lzf", choices=["lzf", "gzip", "none"])
    ap.add_argument("--gzip_level", type=int, default=1)
    ap.add_argument("--slice_axis", type=int, default=2)
    args = ap.parse_args()
    if not args.out:
        raise SystemExit("--out is required (or set H5_PATH in .env)")

    data_root = Path(args.data_root)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    comp = None if args.compression == "none" else args.compression
    comp_opts = args.gzip_level if args.compression == "gzip" else None

    # Collect every (split, modality, field, path) up front for a clean progress bar.
    jobs = []
    for split in args.splits:
        for mod in MODALITIES:
            for field in FIELD_STRENGTHS:
                for path in list_nifti_files(data_root, split, mod, field):
                    jobs.append((split, mod, field, path))
    if not jobs:
        raise SystemExit(f"No .nii.gz found under {data_root} for splits {args.splits}")

    # Write to a temp file and rename on success, so the real path only ever holds a
    # complete pack — an interrupted/failed run never leaves a half-written h5 there.
    tmp = out.with_name(out.name + ".tmp")
    n_written = 0
    per_split = defaultdict(int)
    failed = []          # (path, error) for files that couldn't be read
    shapes = set()       # distinct volume shapes seen (should be one)
    odd_range = 0        # volumes with values outside [0,1] (pipeline assumes [0,1])
    with h5py.File(tmp, "w") as h5:
        h5.attrs["data_root"] = str(data_root)
        h5.attrs["slice_axis"] = args.slice_axis
        for split, mod, field, path in tqdm(jobs, desc="packing"):
            sid = extract_subject_id(path.name)
            key = f"{split}/{mod}/{field}/{sid}"
            if key in h5:
                continue
            try:
                data, _ = load_nifti(path)  # (X, Y, Z) RAS+ canonical, float32
            except Exception as e:  # noqa: BLE001 -- one bad file must not abort the pack
                failed.append((str(path), repr(e)))
                continue

            shapes.add(data.shape)
            if data.min() < -0.01 or data.max() > 1.01:
                odd_range += 1
            std = data.std(axis=tuple(i for i in range(data.ndim) if i != args.slice_axis))
            chunks = list(data.shape)
            chunks[args.slice_axis] = 1

            store = data.astype(args.dtype) if args.dtype != "uint16" else \
                np.clip(data, 0, 1).astype(np.float32)
            if args.dtype == "uint16":
                store = (store * 65535.0).round().astype(np.uint16)

            ds = h5.create_dataset(key, data=store, chunks=tuple(chunks),
                                   compression=comp, compression_opts=comp_opts)
            ds.attrs["slice_std"] = std.astype(np.float32)
            ds.attrs["filename"] = path.name
            ds.attrs["scale"] = 65535.0 if args.dtype == "uint16" else 1.0
            n_written += 1
            per_split[split] += 1

    if n_written == 0:
        tmp.unlink(missing_ok=True)
        raise SystemExit("packed 0 volumes — aborting; any existing h5 left untouched")
    os.replace(tmp, out)

    size_gb = out.stat().st_size / 1e9
    print(f"wrote {n_written}/{len(jobs)} volumes -> {out}  "
          f"({size_gb:.1f} GB, {args.dtype}/{args.compression})")
    for split in args.splits:
        print(f"    {split:24} {per_split[split]:5} volumes")

    if len(shapes) > 1:
        print(f"  note: {len(shapes)} distinct volume shapes: {sorted(shapes)}")
    if odd_range:
        print(f"  WARN: {odd_range} volume(s) had values outside [0,1] (pipeline assumes [0,1])")
    if failed:
        print(f"  WARN: skipped {len(failed)} unreadable file(s):")
        for p, e in failed[:20]:
            print(f"    {p}  ({e})")
        if len(failed) > 20:
            print(f"    ... and {len(failed) - 20} more")


if __name__ == "__main__":
    main()
