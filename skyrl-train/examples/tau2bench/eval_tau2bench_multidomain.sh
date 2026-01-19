#!/bin/bash
set -x

# Evaluation script for Tau2-Bench Multi-domain
# Adapted from collabllm_multiturn evaluation example

# Set tau2 data directory (required for tau2-bench to find domain data files)
export TAU2_DATA_DIR="${TAU2_DATA_DIR:-$HOME/projs/tau2-bench/data}"

# Configuration
: "${DATA_DIR:="$HOME/data/tau2bench/multidomain"}"
: "${NUM_GPUS:=1}"
: "${LOGGER:=wandb}"
: "${INFERENCE_BACKEND:=vllm}"
: "${MAX_TURNS:=20}"

# Checkpoint configuration
: "${GLOBAL_STEP:=10}"
: "${MODEL_NAME:=qwen2p5-1p5b-instruct}"

CKPT_FORMAT="models--mytau2bench--multidomain--$MODEL_NAME--global_step_"
CKPT_DIR="$HOME/hf/hub/${CKPT_FORMAT}${GLOBAL_STEP}"

uv run --isolated --extra $INFERENCE_BACKEND --extra tau2bench -m skyrl_train.entrypoints.main_generate \
  data.val_data="['$DATA_DIR/validation.parquet']" \
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
  environment.env_class=tau2bench_multidomain \
  environment.skyrl_gym.tau2bench_multidomain.task_split="test" \
  environment.skyrl_gym.tau2bench_multidomain.solo_mode=false \
  environment.skyrl_gym.tau2bench_multidomain.user_llm="gpt-4" \
  environment.skyrl_gym.tau2bench_multidomain.user_llm_args.temperature=0.7 \
  environment.skyrl_gym.tau2bench_multidomain.max_turns=$MAX_TURNS \
  generator.use_conversation_multi_turn=true \
  generator.max_turns=$MAX_TURNS \
  generator.max_input_length=8192 \
  generator.eval_n_samples_per_prompt=1 \
  trainer.project_name="tau2bench-multidomain-eval" \
  trainer.run_name="tau2bench_multidomain_eval_model_${MODEL_NAME}_global_${GLOBAL_STEP}" \
  generator.rollout_log_path="$HOME/ckpts/rollout_logs/test_rollouts_tau2bench_multidomain_${MODEL_NAME}_global_${GLOBAL_STEP}_eval.jsonl" \
  $@
