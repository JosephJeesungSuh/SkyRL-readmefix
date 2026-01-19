#!/bin/bash
set -x

# Training script for Tau2-Bench Airline domain
# Adapted from collabllm_multiturn example

# First, generate the dataset:
# python examples/tau2bench/tau2bench_dataset.py --domain airline --env_class tau2bench_airline --output_dir $HOME/data/tau2bench/airline

# Set tau2 data directory (required for tau2-bench to find domain data files)
export TAU2_DATA_DIR="${TAU2_DATA_DIR:-$HOME/projs/tau2-bench/data}"

# Configuration
: "${DATA_DIR:="$HOME/data/tau2bench/airline"}"
: "${NUM_GPUS:=2}"
: "${LOGGER:=wandb}" # change to "console" to print to stdout
: "${INFERENCE_BACKEND:=vllm}"
: "${MAX_TURNS:=20}"
: "${DOMAIN:=airline}"

# Model configuration
: "${MODEL_PATH:="/nas/ucb/jjssuh/hf/hub/models--mygsm8k--qwen2p5-1p5b-instruct--global_step_351"}"
: "${PROJECT_NAME:="tau2bench-${DOMAIN}--qwen2p5-1p5b"}"
: "${RUN_NAME:="tau2bench_${DOMAIN}--qwen2p5-1p5b"}"
: "${CKPT_PATH:="$HOME/ckpts/tau2bench_${DOMAIN}_qwen2p5_1.5B_ckpt"}"

## always make sure no space after the backslash `\` at the end of the line

uv run --isolated --extra $INFERENCE_BACKEND --extra tau2bench -m skyrl_train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="$MODEL_PATH" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.critic_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  trainer.epochs=50 \
  trainer.eval_batch_size=16 \
  trainer.eval_before_train=true \
  trainer.eval_interval=10 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=16 \
  trainer.policy_mini_batch_size=8 \
  trainer.micro_forward_batch_size_per_gpu=2 \
  trainer.micro_train_batch_size_per_gpu=2 \
  trainer.ckpt_interval=10 \
  trainer.max_prompt_length=2048 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  trainer.logger="$LOGGER" \
  trainer.project_name="$PROJECT_NAME" \
  trainer.run_name="$RUN_NAME" \
  trainer.resume_mode=latest \
  trainer.ckpt_path="$CKPT_PATH" \
  generator.sampling_params.max_generate_length=4096 \
  generator.num_inference_engines=$NUM_GPUS \
  generator.inference_engine_tensor_parallel_size=1 \
  generator.backend=$INFERENCE_BACKEND \
  generator.n_samples_per_prompt=5 \
  generator.gpu_memory_utilization=0.8 \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=false \
  generator.use_conversation_multi_turn=true \
  generator.max_turns=$MAX_TURNS \
  generator.max_input_length=8192 \
  generator.rollout_log_path="$HOME/ckpts/rollout_logs/rollouts_tau2bench_${DOMAIN}_qwen2p5_1p5b.jsonl" \
  environment.env_class=tau2bench_airline \
  environment.skyrl_gym.tau2bench_airline.domain="$DOMAIN" \
  environment.skyrl_gym.tau2bench_airline.task_split="train" \
  environment.skyrl_gym.tau2bench_airline.solo_mode=false \
  environment.skyrl_gym.tau2bench_airline.user_llm="gpt-4" \
  environment.skyrl_gym.tau2bench_airline.user_llm_args.temperature=0.7 \
  environment.skyrl_gym.tau2bench_airline.max_turns=$MAX_TURNS \
  $@
