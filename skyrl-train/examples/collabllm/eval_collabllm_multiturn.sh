set -x

DATA_DIR="$HOME/data/collabllm"
NUM_GPUS=1
LOGGER="wandb"
INFERENCE_BACKEND="vllm"
MAX_TURNS=4

# : "${CKPT_DIR:="$HOME/hf/hub/models--mycollabllm--math-500--qwen2p5-7b-instruct--global_step_20"}"
# : "${CKPT_DIR:="$HOME/hf/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"}" # -- base model
: "${GLOBAL_STEP:=10}"
: "${CKPT_FORMAT:=models--mycollabllm--math-500--qwen2p5-0p5b-instruct--global_step_}"

CKPT_DIR="$HOME/hf/hub/${CKPT_FORMAT}${GLOBAL_STEP}"

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
  environment.skyrl_gym.collabllm_math_500_multiturn.user_simulator.tone="default" \
  generator.use_conversation_multi_turn=true \
  generator.max_turns=$MAX_TURNS \
  generator.max_input_length=8192 \
  generator.eval_n_samples_per_prompt=1 \
  trainer.project_name="collabllm-eval" \
  trainer.run_name="collabllm_eval_multiturn_global_${GLOBAL_STEP}" \
  generator.rollout_log_path="$HOME/ckpts/rollout_logs/test_rollouts_collabllm_multiturn_qwen2p5_0p5b_global_${GLOBAL_STEP}_eval.jsonl"
