#!/usr/bin/env bash
set -x

# Example colocated GRPO training on the Lost in Conversation sharded instructions.
# Requires OPENAI_API_KEY (or Azure OpenAI variables) because the environment delegates
# system/user turns to the upstream simulators.

DATASET_PATH=${DATASET_PATH:-$(dirname "$0")/lost_in_conversation.jsonl}

uv run --isolated --extra vllm -m skyrl_train.entrypoints.main_base \
  data.train_data="['${DATASET_PATH}']" \
  data.val_data="['${DATASET_PATH}']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="Qwen/Qwen2.5-0.5B-Instruct" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=1 \
  trainer.placement.ref_num_gpus_per_node=1 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.policy.optimizer_config.max_grad_norm=0.5 \
  trainer.train_batch_size=16 \
  trainer.policy_mini_batch_size=8 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.update_epochs_per_batch=1 \
  trainer.epochs=1 \
  trainer.max_prompt_length=2048 \
  generator.max_input_length=4096 \
  generator.sampling_params.max_generate_length=512 \
  generator.sampling_params.temperature=1.0 \
  generator.sampling_params.stop='[]' \
  generator.backend=vllm \
  generator.run_engines_locally=true \
  generator.batched=false \
  generator.use_conversation_multi_turn=true \
  generator.max_turns=6 \
  generator.n_samples_per_prompt=1 \
  environment.env_class="lost_in_conversation" \
  environment.skyrl_gym.max_env_workers=4 \
  environment.skyrl_gym.lost_in_conversation.dataset_path="lost_in_conversation/data/sharded_instructions_600.json" \
  environment.skyrl_gym.lost_in_conversation.assistant_model="gpt-4o-mini" \
  environment.skyrl_gym.lost_in_conversation.system_model="gpt-4o-mini" \
  environment.skyrl_gym.lost_in_conversation.user_model="gpt-4o-mini" \
  environment.skyrl_gym.lost_in_conversation.assistant_temperature=1.0 \
  environment.skyrl_gym.lost_in_conversation.user_temperature=1.0 \
  trainer.logger="wandb" \
  trainer.project_name="skyrl-lost-in-conversation" \
  trainer.run_name="lic-grpo" \
  "$@"