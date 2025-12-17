set -x

# Colocated GRPO training+generation for Qwen2.5-1.5B-Instruct on GSM8K.

# uv run examples/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
# export WANDB_API_KEY=<your_key_here>
# bash examples/gsm8k/run_gsm8k.sh

# NOTE (sumanthrh): `micro_train_batch_size_per_gpu` and `micro_forward_batch_size_per_gpu` can be tuned

# You can override the default values with e.g.: `NUM_GPUS=1 bash examples/gsm8k/run_gsm8k.sh`.

: "${DATA_DIR:="$HOME/data/collabllm"}"
: "${NUM_GPUS:=2}"
: "${LOGGER:=wandb}" # change to "console" to print to stdout

: "${INFERENCE_BACKEND:=vllm}"
# : "${INFERENCE_BACKEND:=sglang}"

: "${MAX_TURNS:=4}"

## always make sure no space after the backslash `\` at the end of the line
# environment.env_class=gsm8k \

uv run --isolated --extra $INFERENCE_BACKEND -m skyrl_train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="Qwen/Qwen2.5-3B-Instruct" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.critic_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.num_inference_engines=$NUM_GPUS \
  generator.inference_engine_tensor_parallel_size=1 \
  trainer.epochs=50 \
  trainer.eval_batch_size=32 \
  trainer.eval_before_train=true \
  trainer.eval_interval=10 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=32 \
  trainer.policy_mini_batch_size=16 \
  trainer.micro_forward_batch_size_per_gpu=4 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=10 \
  trainer.max_prompt_length=2048 \
  generator.sampling_params.max_generate_length=4096 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  generator.backend=$INFERENCE_BACKEND \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=false \
  generator.use_conversation_multi_turn=true \
  generator.max_turns=$MAX_TURNS \
  generator.max_input_length=8192 \
  environment.env_class=collabllm_math_500_multiturn \
  environment.skyrl_gym.collabllm_math_500_multiturn.llm_judge.enabled=true \
  environment.skyrl_gym.collabllm_math_500_multiturn.llm_judge.model_name="mistralai/Mistral-Small-3.1-24B-Instruct-2503" \
  environment.skyrl_gym.collabllm_math_500_multiturn.llm_judge.is_local=true \
  environment.skyrl_gym.collabllm_math_500_multiturn.llm_judge.local_port=8002 \
  environment.skyrl_gym.collabllm_math_500_multiturn.user_simulator.enabled=true \
  environment.skyrl_gym.collabllm_math_500_multiturn.user_simulator.model_name="mistralai/Mistral-Small-3.1-24B-Instruct-2503" \
  environment.skyrl_gym.collabllm_math_500_multiturn.user_simulator.is_local=true \
  environment.skyrl_gym.collabllm_math_500_multiturn.user_simulator.local_port=8002 \
  generator.user_simulator.enabled=true \
  generator.n_samples_per_prompt=5 \
  generator.gpu_memory_utilization=0.8 \
  trainer.logger="$LOGGER" \
  trainer.project_name="collabllm-test" \
  trainer.run_name="collabllm_test_multiturn" \
  trainer.resume_mode=null \
  trainer.ckpt_path="$HOME/ckpts/collabllm_qwen2p5_3B_ckpt" \
  generator.rollout_log_path="$HOME/ckpts/rollout_logs/test_rollouts_collabllm_multiturn_qwen2p5_3b.jsonl" \
  $@
