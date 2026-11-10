#!/bin/bash
# Launcher for the paper baselines (regression / SDEdit diffusion), which share the
# flow model's U-Net. Activate the environment first, then e.g.
# `METHOD=regression bash run_baseline.sh` from the repo root.
# Every value honours an env override (`${VAR:-default}`).
set -euo pipefail

# ------------------------------ CONFIG (edit me) -----------------------------
GPU="${GPU:-0}"
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
DATA_DIR="${DATA_DIR:?set DATA_DIR to the challenge data root}"
VAL_DIR="${VAL_DIR:-$DATA_DIR/Validating_prospective}"   # infer reads <VAL_DIR>/<modality>/<field>/
H5_PATH="${H5_PATH:-}"                            # packed data from tools/build_h5 ("" = read niftis)
OUTPUT_DIR="${OUTPUT_DIR:-$REPO/model_weights}"
INFERENCE_DIR="${INFERENCE_DIR:-$REPO/outputs}"

METHOD="${METHOD:-regression}"                    # regression | diffusion
CFG="${CFG:-configs/baseline_${METHOD}.yaml}"
STAGES="${STAGES:-pretrain finetune}"             # + optional `infer`
FIELDS="${FIELDS:-0.1T 1.5T 3T 5T 7T}"
INIT_CKPT="${INIT_CKPT:-}"                        # warm-start the FIRST stage ("" = random)
INFER_CKPT="${INFER_CKPT:-}"                      # checkpoint for infer ("" = latest)
INFER_STEPS="${INFER_STEPS:-0}"                   # 0 = the method's own default
INFER_BATCH="${INFER_BATCH:-32}"
WANDB_PROJECT="${WANDB_PROJECT:-}"                # "" = wandb off
TASK_NAME="${TASK_NAME:-}"                        # "" = config task_name
# -----------------------------------------------------------------------------

cd "$REPO"
export CUDA_VISIBLE_DEVICES="$GPU"
export DATA_DIR OUTPUT_DIR INFERENCE_DIR          # src/utils/env.py reads these

case "$METHOD" in
  regression|diffusion) ;;
  *) echo "ERROR: METHOD must be regression or diffusion (got '$METHOD')"; exit 1;;
esac

H5_ARG=""
if [ -n "$H5_PATH" ]; then H5_ARG="--h5 $H5_PATH"; fi
if [ -n "$WANDB_PROJECT" ]; then export WANDB_PROJECT; fi

has_stage() { case " $STAGES " in *" $1 "*) return 0;; *) return 1;; esac; }
# Baseline checkpoints live under their own dir, apart from the flow model's.
BASE_W="$OUTPUT_DIR/baseline_$METHOD"
latest() { ls -t "$BASE_W"/*/finetune/checkpoint_epoch*.pth \
                  "$BASE_W"/*/*/checkpoint_epoch*.pth 2>/dev/null | head -1 || true; }

TRAIN_STAGES=""
for s in pretrain finetune; do has_stage "$s" && TRAIN_STAGES="$TRAIN_STAGES $s"; done
if [ -n "$TRAIN_STAGES" ]; then
  echo "==================== $METHOD train:$TRAIN_STAGES ===================="
  INIT_ARG=""; [ -n "$INIT_CKPT" ] && INIT_ARG="--init_from $INIT_CKPT"
  NAME_ARG=""; [ -n "$TASK_NAME" ] && NAME_ARG="--task_name $TASK_NAME"
  python -m src.baselines.train_baseline --config "$CFG" --method "$METHOD" \
    --stages $TRAIN_STAGES $INIT_ARG $H5_ARG $NAME_ARG
fi

# --- infer: predictions for every directed pair, for every subject at the source field ---
if has_stage infer; then
  CK="${INFER_CKPT:-$(latest)}"
  [ -n "$CK" ] || { echo "ERROR: no baseline checkpoint found in $BASE_W"; exit 1; }
  echo "[infer] checkpoint: $CK"
  for src in $FIELDS; do
    for tgt in $FIELDS; do
      [ "$src" = "$tgt" ] && continue
      echo "==================== ${src} -> ${tgt} ===================="
      python -m src.baselines.infer_baseline --config "$CFG" --method "$METHOD" \
        --source "$src" --target "$tgt" --checkpoint "$CK" \
        --input_dir "$VAL_DIR" \
        --output_dir "$INFERENCE_DIR/task3_baseline_$METHOD/${src}_to_${tgt}/pred" \
        --steps "$INFER_STEPS" --batch_size "$INFER_BATCH"
    done
  done
fi
echo "ALL DONE."
