#!/bin/bash
# Launcher for the flow-matching model. Activate the environment first, then
# `bash run_task3.sh` from the repo root. Every value honours an env override
# (`${VAR:-default}`). The training stages run in ONE process / ONE wandb run;
# `infer` loops over the 20 directed field pairs.
set -euo pipefail

# ------------------------------ CONFIG (edit me) -----------------------------
GPU="${GPU:-0}"
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
DATA_DIR="${DATA_DIR:?set DATA_DIR to the challenge data root}"
VAL_DIR="${VAL_DIR:-$DATA_DIR/Validating_prospective}"  # infer reads <VAL_DIR>/<modality>/<field>/
H5_PATH="${H5_PATH:-}"                             # packed data from tools/build_h5 ("" = read niftis)
OUTPUT_DIR="${OUTPUT_DIR:-$REPO/model_weights}"
INFERENCE_DIR="${INFERENCE_DIR:-$REPO/outputs}"

FIELDS="${FIELDS:-0.1T 1.5T 3T 5T 7T}"
STAGES="${STAGES:-pretrain finetune refine_gan infer}"
PRETRAIN_CKPT="${PRETRAIN_CKPT:-}"                 # warm-start finetune ("" = latest)
FROM_SCRATCH="${FROM_SCRATCH:-0}"                  # 1 = finetune from random init (no pretrain)
FINETUNE_CKPT="${FINETUNE_CKPT:-}"                 # warm-start a refine_gan-only run ("" = latest)
INFER_CKPT="${INFER_CKPT:-}"                       # checkpoint for infer ("" = latest refine_gan, else finetune)
INFER_STEPS="${INFER_STEPS:-5}"                    # Heun steps
INFER_BATCH="${INFER_BATCH:-32}"
WANDB_PROJECT="${WANDB_PROJECT:-}"                 # "" = wandb off
TASK_NAME="${TASK_NAME:-}"                         # override run name + checkpoint subfolder ("" = config)
PRETRAIN_NOISE="${PRETRAIN_NOISE:-0}"              # 1 = pure-noise pretrain (ignore the corruption block)

CFG="${CFG:-configs/task3_any_to_any.yaml}"
# -----------------------------------------------------------------------------

cd "$REPO"
export CUDA_VISIBLE_DEVICES="$GPU"
export DATA_DIR OUTPUT_DIR INFERENCE_DIR           # src/utils/env.py reads these

H5_ARG=""
if [ -n "$H5_PATH" ]; then H5_ARG="--h5 $H5_PATH"; fi
if [ -n "$WANDB_PROJECT" ]; then export WANDB_PROJECT; fi

has_stage() { case " $STAGES " in *" $1 "*) return 0;; *) return 1;; esac; }
# Newest checkpoint under a stage dir: runs save to <stage>/<run>_<timestamp>/.
latest() { ls -t "$1"/*/checkpoint_epoch*.pth "$1"/*.pth 2>/dev/null | head -1 || true; }
require() { [ -n "$1" ] || { echo "ERROR: no checkpoint found in $2"; exit 1; }; }

PRE_W="$OUTPUT_DIR/pretrain"
FT_W="$OUTPUT_DIR/finetune"
RG_W="$OUTPUT_DIR/refine_gan"

# --- training: the requested stages run in one process ---
# If pretrain is skipped, the first stage warm-starts from the prior stage's checkpoint.
TRAIN_STAGES=""
for s in pretrain finetune refine_gan; do has_stage "$s" && TRAIN_STAGES="$TRAIN_STAGES $s"; done
if [ -n "$TRAIN_STAGES" ]; then
  echo "==================== train:$TRAIN_STAGES ===================="
  INIT_ARG=""
  first="$(echo $TRAIN_STAGES | awk '{print $1}')"
  if [ "$first" = "finetune" ] && [ "$FROM_SCRATCH" != "1" ]; then
    CK="${PRETRAIN_CKPT:-$(latest "$PRE_W")}"
    if [ -n "$CK" ]; then
      INIT_ARG="--init_from $CK"
    else
      echo "WARNING: no pretrain checkpoint in $PRE_W -> finetune starts from random init"
    fi
  elif [ "$first" = "refine_gan" ]; then
    CK="${FINETUNE_CKPT:-$(latest "$RG_W")}"; [ -n "$CK" ] || CK="$(latest "$FT_W")"
    require "$CK" "$FT_W"
    INIT_ARG="--init_from $CK"
  fi
  NAME_ARG=""; [ -n "$TASK_NAME" ] && NAME_ARG="--task_name $TASK_NAME"
  NOISE_ARG=""; [ "$PRETRAIN_NOISE" = "1" ] && NOISE_ARG="--pretrain_noise"
  python -m src.scripts.train_flow --config "$CFG" --stages $TRAIN_STAGES $INIT_ARG $H5_ARG $NAME_ARG $NOISE_ARG
fi

# --- infer: predictions for every directed pair, for every subject at the source field ---
if has_stage infer; then
  # Prefer the adversarially-refined weights, else the finetuned bridge.
  CK="${INFER_CKPT:-$(latest "$RG_W")}"; [ -n "$CK" ] || CK="$(latest "$FT_W")"
  require "$CK" "$FT_W"
  echo "[infer] checkpoint: $CK"
  for src in $FIELDS; do
    for tgt in $FIELDS; do
      [ "$src" = "$tgt" ] && continue
      echo "==================== ${src} -> ${tgt} ===================="
      python -m src.scripts.inference_flow --config "$CFG" --source "$src" --target "$tgt" \
        --checkpoint "$CK" --input_dir "$VAL_DIR" \
        --output_dir "$INFERENCE_DIR/task3/${src}_to_${tgt}/pred" \
        --steps "$INFER_STEPS" --batch_size "$INFER_BATCH"
    done
  done
fi
echo "ALL DONE."
