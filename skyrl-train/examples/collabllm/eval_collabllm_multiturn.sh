set -x

DATA_DIR="$HOME/data/collabllm"
NUM_GPUS=1
LOGGER="wandb"
INFERENCE_BACKEND="vllm"


# export CKPT_DIR="$HOME/ckpts/collabllm_qwen2p5_7B_ckpt/global_step_20/policy"  -- this is the original model checkpoint with model_world_size_{fsdp_size}_rank_{i}.pt files
: "${CKPT_DIR:="$HOME/hf/hub/models--mycollabllm--math-500--qwen2p5-7b-instruct--global_step_20"}"
MAX_TURNS=4

uv run --isolated --extra $INFERENCE_BACKEND -m skyrl_train.entrypoints.main_generate \
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
  environment.env_class=collabllm_math_500_multiturn \
  environment.skyrl_gym.collabllm_math_500_multiturn.llm_judge.enabled=true \
  environment.skyrl_gym.collabllm_math_500_multiturn.llm_judge.model_name="mistralai/Mistral-Small-3.1-24B-Instruct-2503" \
  environment.skyrl_gym.collabllm_math_500_multiturn.llm_judge.is_local=true \
  environment.skyrl_gym.collabllm_math_500_multiturn.llm_judge.local_port=8002 \
  environment.skyrl_gym.collabllm_math_500_multiturn.user_simulator.enabled=true \
  environment.skyrl_gym.collabllm_math_500_multiturn.user_simulator.model_name="mistralai/Mistral-Small-3.1-24B-Instruct-2503" \
  environment.skyrl_gym.collabllm_math_500_multiturn.user_simulator.is_local=true \
  environment.skyrl_gym.collabllm_math_500_multiturn.user_simulator.local_port=8002 \
  generator.use_conversation_multi_turn=true \
  generator.max_turns=$MAX_TURNS \
  generator.max_input_length=8192 \
  generator.user_simulator.enabled=true \
  generator.eval_n_samples_per_prompt=5 \
  trainer.project_name="collabllm-test" \
  trainer.run_name="collabllm_eval_multiturn" \
  generator.rollout_log_path="$HOME/ckpts/rollout_logs/test_rollouts_collabllm_multiturn_qwen2p5_7b_eval_testing.jsonl"
