#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/eval_collabllm_multiturn.sh"

# Optional: override these from the wrapper without editing run_one.sh
# export CKPT_FORMAT="models--mycollabllm--math-500--qwen2p5-0p5b-instruct--global_step_}"
# export DATA_DIR="$HOME/data/collabllm"
# export NUM_GPUS=1
# export LOGGER="wandb"
# export INFERENCE_BACKEND="vllm"
# export MAX_TURNS=4

for step in $(seq 120 10 650); do
  export GLOBAL_STEP="$step"
  echo "===== Running GLOBAL_STEP=${GLOBAL_STEP} ====="
  GLOBAL_STEP=$GLOBAL_STEP bash $BASE_SCRIPT
done
