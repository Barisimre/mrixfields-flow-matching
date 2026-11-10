# Conditional Flow Matching for Cross-Field MRI Harmonisation

Code for our MRIxFields2026 (MICCAI 2026) Task 3 method: **one** model that translates
brain MRI between **any** two of five field strengths (0.1T, 1.5T, 3T, 5T, 7T) for
three contrasts (T1w, T2w, T2-FLAIR), i.e. 20 directed field pairs x 3 contrasts with
a single set of weights.

## Method

Source and target volumes are spatially registered, so the flow runs directly from
the source slice to the target slice instead of from noise:

- path `x_t = (1 - t) x0 + t x1`, velocity target `x1 - x0`
- the network sees `x_t` only; the source enters as the `t = 0` state of the integration
- the three contrasts are stacked as channels; the timestep and the source and target
  field strengths are embedded and injected through FiLM
- inference integrates the velocity from `t = 0` to `t = 1` with Heun steps (5 by default)

Training has three stages, run by one script:

| stage | data | what it learns |
|---|---|---|
| `pretrain` | unpaired retrospective cohort | a restoration prior: degraded copy of a slice -> the clean slice |
| `finetune` | paired travelling subjects, all 20 directed pairs | the source -> target bridge |
| `refine_gan` | same paired data | sharpening: a conditional multi-scale PatchGAN judges a multi-step rollout from the source, anchored by the velocity loss and `1 - SSIM` |

## Layout

```
configs/      task3_any_to_any.yaml (all stages), network.yaml (U-Net + discriminator),
              baseline_{regression,diffusion}.yaml
src/models/   flow_unet.py, flow_matching.py (losses, sampler), degrade.py (pretrain
              corruptions), ssim.py, refine_gan/ (adversarial stage)
src/data/     NIfTI- and HDF5-backed slice datasets
src/scripts/  train_flow.py, inference_flow.py
src/eval/     metrics.py (nRMSE / SSIM / LPIPS)
src/baselines/  regression and diffusion baselines on the same U-Net
tools/        build_h5.py (pack the dataset for fast slice reads)
run_task3.sh, run_baseline.sh   launchers
```

## Setup

```bash
conda env create -f environment.yml && conda activate mrix   # or: pip install -r requirements.txt
cp .env.example .env                                          # set DATA_DIR
```

Data are the MRIxFields2026 challenge volumes, laid out as
`DATA_DIR/<split>/<modality>/<field>/<subject>.nii.gz` with splits
`Training_retrospective`, `Training_prospective`, `Validating_prospective`, modalities
`T1W`, `T2W`, `T2FLAIR` and fields `0.1T`, `1.5T`, `3T`, `5T`, `7T`. The data are not
part of this repository.

Run everything from the repository root; `src` and `tools` are imported as packages.

## Usage

Optionally pack the dataset once into an HDF5 file (much faster slice reads):

```bash
python -m tools.build_h5 --data_root $DATA_DIR --out $DATA_DIR/mrix.h5
```

Train all three stages in one process (drop `--h5` to read the NIfTIs directly):

```bash
python -m src.scripts.train_flow --config configs/task3_any_to_any.yaml \
    --stages pretrain finetune refine_gan --h5 $DATA_DIR/mrix.h5
```

Run a subset with `--stages`, and warm-start the first stage with `--init_from <ckpt>`.
Checkpoints go to `OUTPUT_DIR/<stage>/<run>_<timestamp>/checkpoint_epochN.pth`. Set
`WANDB_PROJECT` to log to Weights & Biases.

Translate one field pair:

```bash
python -m src.scripts.inference_flow --config configs/task3_any_to_any.yaml \
    --checkpoint <ckpt> --source 0.1T --target 7T --steps 5 \
    --input_dir $DATA_DIR/Validating_prospective --output_dir outputs/0.1T_to_7T
```

By default only the evaluated axial slab (slices 150-179) is predicted; add
`--full-volume` for every slice. `run_task3.sh` wraps training and inference over all
20 pairs (`DATA_DIR=... STAGES="pretrain finetune refine_gan infer" bash run_task3.sh`).

Baselines from the paper (same U-Net, conditioning and data):

```bash
DATA_DIR=... METHOD=regression bash run_baseline.sh    # or METHOD=diffusion
```

## Results

Challenge evaluation set, averaged over the three contrasts and all 20 directed pairs:

| method | sampler | SSIM | LPIPS | nRMSE |
|---|---|---|---|---|
| Identity (copy source) | - | 0.836 | 0.157 | 0.522 |
| Regression (same U-Net) | 1 forward pass | 0.906 | 0.110 | 0.246 |
| Diffusion (same U-Net) | 50 DDIM steps | 0.896 | 0.121 | 0.249 |
| Ours, pretrained only | 5 Heun steps | 0.837 | 0.147 | 0.468 |
| Ours, + finetuning | 5 Heun steps | 0.885 | 0.103 | 0.268 |
| **Ours, + adversarial refinement** | 5 Heun steps | **0.909** | **0.089** | **0.227** |

## Citation

Paper: <https://papers.miccai.org/miccai-2026-sat/MRIxFields2026_012.html>

```bibtex
@InProceedings{ImrBar_Conditional_MICCAISAT2026,
    author = {Imre, Baris AND Salehi, Aram AND Baljer, Levente AND Webb, Andrew AND Staring, Marius AND Ilicak, Efe},
    title = {{Conditional Flow Matching for Cross-Field MRI Harmonisation}},
    booktitle = {Medical Image Computing and Computer Assisted Intervention -- MICCAI 2026 Workshops and Challenges},
    year = {2026},
    publisher = {Springer Nature Switzerland},
    volume = {LNCS 17274}
}
```

## License

MIT, see `LICENSE`.
