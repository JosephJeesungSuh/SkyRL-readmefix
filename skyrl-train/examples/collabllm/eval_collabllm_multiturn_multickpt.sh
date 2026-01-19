#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/eval_collabllm_multiturn.sh"

# Optional: override these from the wrapper without editing run_one.sh
# export CKPT_FORMAT="models--mycollabllm--math-500--qwen2p5-1p5b-instruct--global_step_}"
# export DATA_DIR="$HOME/data/collabllm"
# export NUM_GPUS=1
# export LOGGER="wandb"
# export INFERENCE_BACKEND="vllm"
# export MAX_TURNS=4

for step in $(seq 150 10 650); do
  export GLOBAL_STEP="$step"
  export TONE="default"
  export MODEL_NAME="qwen2p5-1p5b-instruct-training-with-angry"
  export NUM_GPUS=2
  echo "===== Running GLOBAL_STEP=${GLOBAL_STEP} ====="
  GLOBAL_STEP=$GLOBAL_STEP TONE=$TONE MODEL_NAME=$MODEL_NAME NUM_GPUS=$NUM_GPUS bash $BASE_SCRIPT
done
