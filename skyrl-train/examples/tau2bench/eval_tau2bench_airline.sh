#!/bin/bash
set -x

# Evaluation script for Tau2-Bench Airline domain
# Adapted from collabllm_multiturn evaluation example

# Configuration
: "${DOMAIN:=airline}"
: "${NUM_GPUS:=1}"
: "${LOGGER:=wandb}"
: "${INFERENCE_BACKEND:=vllm}"
: "${MAX_TURNS:=30}"

# Set tau2 data directory (required for tau2-bench to find domain data files)
export TAU2_DATA_DIR="${TAU2_DATA_DIR:-$HOME/projs/tau2-bench/data}"

# Checkpoint configuration : base model name and global step
: "${MODEL_NAME:=qwen2p5-1p5b-instruct}"
: "${GLOBAL_STEP:=0}"

CKPT_FORMAT="models--mytau2bench--${DOMAIN}--$MODEL_NAME--global_step_"
CKPT_DIR="$HOME/hf/hub/${CKPT_FORMAT}${GLOBAL_STEP}"
if [ "$GLOBAL_STEP" -eq 0 ]; then
  if [ "$MODEL_NAME" == "qwen2p5-1p5b-instruct" ]; then
    CKPT_DIR="Qwen/Qwen2.5-1.5B-Instruct"
  else
    echo "E: currently only MODEL_NAME=qwen2p5-1p5b-instruct is supported."
    exit 1
  fi
fi

DATA_DIR="$HOME/data/tau2bench/${DOMAIN}"

uv run --isolated echo "Evaluating Tau2-Bench ${DOMAIN} domain with model at $CKPT_DIR with data from $DATA_DIR"

uv run --isolated --extra $INFERENCE_BACKEND --extra tau2bench -m skyrl_train.entrypoints.main_generate \
  data.val_data="['$DATA_DIR/validation_debug.parquet']" \
  trainer.policy.model.path="$CKPT_DIR" \
  trainer.logger="$LOGGER" \
  generator.backend=$INFERENCE_BACKEND \
  trainer.placement.colocate_all=false \
  generator.async_engine=true \
  generator.batched=false \
  generator.num_inference_engines=$NUM_GPUS \
  generator.inference_engine_tensor_parallel_size=1 \
  generator.gpu_memory_utilization=0.9 \
  generator.eval_sampling_params.max_generate_length=4096 \
  environment.env_class=tau2bench_airline \
  environment.skyrl_gym.tau2bench_airline.domain="$DOMAIN" \
  environment.skyrl_gym.tau2bench_airline.task_split="test" \
  environment.skyrl_gym.tau2bench_airline.solo_mode=false \
  environment.skyrl_gym.tau2bench_airline.user_llm="gpt-4.1-mini" \
  environment.skyrl_gym.tau2bench_airline.user_llm_args.temperature=0.7 \
  environment.skyrl_gym.tau2bench_airline.max_turns=$MAX_TURNS \
  generator.use_conversation_multi_turn=true \
  generator.max_turns=$MAX_TURNS \
  generator.max_input_length=8192 \
  generator.eval_n_samples_per_prompt=1 \
  trainer.project_name="tau2bench-${DOMAIN}-eval" \
  trainer.run_name="tau2bench_${DOMAIN}_eval_model_${MODEL_NAME}_global_${GLOBAL_STEP}" \
  generator.rollout_log_path="$HOME/ckpts/rollout_logs/test_rollouts_tau2bench_${DOMAIN}_${MODEL_NAME}_global_${GLOBAL_STEP}_eval.jsonl" \
  $@
