"""The evaluated axial slab.

MRIxFields2026 scores only a central slab of each volume, so inference integrates
just these slices by default (`--full-volume` overrides). Keep the range in one place:
a prediction and a reference cut at different z ranges would have the same shape but
show different slices, and metrics would silently be computed on misaligned voxels.
"""

#: (z_start_inclusive, z_end_exclusive) — axial slice indices on the
#: 0.5 mm isotropic grid (364×436×364 dataset volume). 30 slices total,
#: covering physical z ≈ +3.0 mm to +17.5 mm in MNI space.
Z_CLIP_RANGE = (150, 180)
