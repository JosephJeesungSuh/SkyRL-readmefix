# Tau2-Bench Integration with SkyRL

This directory contains the integration of [tau2-bench](https://github.com/sierra-research/tau2-bench) with SkyRL for training conversational agents using reinforcement learning.

## Overview

Tau2-bench is a benchmark for evaluating conversational agents in dual-control environments, focusing on customer service scenarios across multiple domains:
- **airline**: Customer service for airline booking and support
- **retail**: E-commerce and retail customer service
- **telecom**: Telecommunications customer support
- **mock**: Simple testing scenarios

This integration allows training RL agents on tau2-bench tasks using SkyRL's training pipeline.

## Installation

### Prerequisites

The tau2-bench integration is available as an optional dependency in SkyRL.

1. Install SkyRL with tau2bench and VLLM backend:
```bash
cd /nas/ucb/jjssuh/projs/external/SkyRL/skyrl-train
pip install -e ".[vllm,tau2bench]"
```

2. Verify tau2-bench is installed and data is available:
```bash
# Check installation
python -c "import tau2; print('tau2-bench installed at:', tau2.__file__)"

# Check data
tau2 check-data
```

**How it works**: The `tau2bench` extra in `pyproject.toml` references tau2-bench from `/nas/ucb/jjssuh/projs/tau2-bench` as a local editable dependency. This allows the training and evaluation scripts to use `uv run --isolated --extra tau2bench` to ensure tau2-bench is available in the isolated environment.

## Usage

### 1. Prepare Dataset

Generate datasets for specific domains:

```bash
# Airline domain
python examples/tau2bench/tau2bench_dataset.py \
  --domain airline \
  --env_class tau2bench_airline \
  --output_dir ~/data/tau2bench/airline

# Retail domain
python examples/tau2bench/tau2bench_dataset.py \
  --domain retail \
  --env_class tau2bench_retail \
  --output_dir ~/data/tau2bench/retail

# Telecom domain
python examples/tau2bench/tau2bench_dataset.py \
  --domain telecom \
  --env_class tau2bench_telecom \
  --output_dir ~/data/tau2bench/telecom

# Multi-domain (all domains combined)
python examples/tau2bench/tau2bench_dataset.py \
  --multidomain \
  --env_class tau2bench_multidomain \
  --output_dir ~/data/tau2bench/multidomain
```

**Task Splits:**
- `--train_split train`: Training set (used for RL training)
- `--val_split test`: Test set (held-out evaluation)
- `--train_split base`: Complete task set (for evaluation matching original tau2-bench)

### 2. Training

Train agents on specific domains:

```bash
# Airline domain
bash examples/tau2bench/run_tau2bench_airline.sh

# Multi-domain (recommended for generalization)
bash examples/tau2bench/run_tau2bench_multidomain.sh
```

**Configuration Options:**

Environment variables you can set:
- `DATA_DIR`: Path to dataset directory
- `NUM_GPUS`: Number of GPUs to use (default: 2)
- `LOGGER`: Logging backend (wandb/console, default: wandb)
- `INFERENCE_BACKEND`: vllm or other supported backend
- `MAX_TURNS`: Maximum conversation turns (default: 20)
- `MODEL_PATH`: Path to pre-trained model checkpoint

Example:
```bash
NUM_GPUS=4 MAX_TURNS=30 bash examples/tau2bench/run_tau2bench_airline.sh
```

### 3. Evaluation

Evaluate trained agents:

```bash
# Airline domain
GLOBAL_STEP=50 bash examples/tau2bench/eval_tau2bench_airline.sh

# Multi-domain
GLOBAL_STEP=50 bash examples/tau2bench/eval_tau2bench_multidomain.sh
```

## Environment Configuration

The tau2-bench environments have default configurations in [`skyrl_train/config/skyrl_gym_config/default.yaml`](../../skyrl_train/config/skyrl_gym_config/default.yaml#L48-L92).

Default configuration for each environment:
```yaml
tau2bench_airline:
  domain: "airline"
  task_split: "train"  # train, test, or base
  solo_mode: false  # true = agent works alone, false = interacts with user simulator
  user_llm: "gpt-4"  # LLM for user simulator (only used if solo_mode=false)
  user_llm_args:
    temperature: 0.7
  max_turns: 20
```

You can override these settings in your training scripts:

```yaml
environment:
  env_class: tau2bench_airline  # or tau2bench_retail, tau2bench_telecom, tau2bench_multidomain
  skyrl_gym:
    tau2bench_airline:
      task_split: test  # Override to use test split
      user_llm: "gpt-4o"  # Override user simulator LLM
      max_turns: 30  # Override max turns
```

**Configuration Options:**

- `domain`: Domain name (airline, retail, telecom, mock) - only for single-domain envs
- `task_split`: Task split to use
  - `"train"`: Training tasks (default for training)
  - `"test"`: Held-out test tasks (use for evaluation)
  - `"base"`: Complete task set (for comparison with original tau2-bench)
- `solo_mode`: Whether agent works independently
  - `false` (default): Agent interacts with user simulator
  - `true`: Agent works alone on task tickets (no user interaction)
- `user_llm`: LLM model for user simulator (e.g., "gpt-4", "gpt-4o", "claude-3-sonnet")
- `user_llm_args`: Additional arguments for user LLM (temperature, max_tokens, etc.)
- `max_turns`: Maximum conversation turns before episode terminates

## Architecture

### Environment Wrapper

[`skyrl_gym/envs/tau2bench/env.py`](../../skyrl-agent/skyrl-train/skyrl-gym/skyrl_gym/envs/tau2bench/env.py) contains:

- **`Tau2BenchEnv`**: Main wrapper adapting tau2-bench's `AgentGymEnv` to SkyRL's `BaseTextEnv`
- **`Tau2BenchMultiDomainEnv`**: Multi-domain variant that handles tasks from all domains

Key features:
- Wraps tau2-bench's Gymnasium interface
- Converts observations to/from SkyRL's conversation format
- Handles both single-turn and multi-turn interactions
- Supports solo mode (agent-only) and collaborative mode (with user simulator)
- Tracks metrics including task success, turns, and evaluation criteria

### Dataset Format

The dataset follows SkyRL's expected format:

```python
{
    "data_source": "tau2bench-airline",
    "prompt": [],  # Initial conversation (typically empty)
    "env_class": "tau2bench_airline",
    "task_id": "0",  # Task ID from tau2-bench
    "domain": "airline",
    "reward_spec": {
        "method": "tau2bench",  # Reward from tau2-bench evaluation
    },
    "extra_info": {
        "task_id": "0",
        "domain": "airline",
        "task_description": {...},
        "user_scenario": {...},
        "evaluation_criteria": {...},
    }
}
```

## Registered Environments

The following environments are registered in SkyRL:

- `tau2bench_airline`: Airline domain tasks
- `tau2bench_retail`: Retail domain tasks
- `tau2bench_telecom`: Telecom domain tasks
- `tau2bench_mock`: Mock domain tasks (for testing)
- `tau2bench_multidomain`: All domains combined

## Files

```
examples/tau2bench/
├── README.md                         # This file
├── tau2bench_dataset.py              # Dataset preparation script
├── run_tau2bench_airline.sh          # Training script for airline domain
├── run_tau2bench_multidomain.sh      # Training script for multi-domain
├── eval_tau2bench_airline.sh         # Evaluation script for airline
└── eval_tau2bench_multidomain.sh     # Evaluation script for multi-domain

skyrl-gym/skyrl_gym/envs/tau2bench/
└── env.py                            # Environment wrapper implementation
```

## Metrics and Rewards

The environment tracks the following metrics:
- **turns**: Number of conversation turns
- **task_passed**: Whether the task was completed successfully (from tau2-bench evaluation)
- **evaluation_type**: Type of evaluation performed
- **pass_rate**: Aggregated success rate across episodes

Rewards are computed by tau2-bench's built-in evaluation system, which checks if the agent:
1. Performed required actions
2. Communicated necessary information
3. Met natural language assertions defined in the task

## Example Workflow

```bash
# 1. Generate dataset
python examples/tau2bench/tau2bench_dataset.py \
  --domain airline \
  --env_class tau2bench_airline \
  --output_dir ~/data/tau2bench/airline

# 2. Train agent
bash examples/tau2bench/run_tau2bench_airline.sh

# 3. Evaluate at checkpoint 50
GLOBAL_STEP=50 bash examples/tau2bench/eval_tau2bench_airline.sh

# 4. View rollout logs
cat ~/ckpts/rollout_logs/test_rollouts_tau2bench_airline_*.jsonl
```

## Notes

- The user simulator is controlled by tau2-bench and uses the specified `user_llm` (e.g., GPT-4)
- Solo mode (`solo_mode=true`) disables user interaction - agent works independently on tickets
- Multi-turn training is enabled by default with `use_conversation_multi_turn=true`
- Checkpoints are saved to `~/ckpts/tau2bench_*` directories
- Rollout logs are saved to `~/ckpts/rollout_logs/` for analysis

## References

- [tau2-bench GitHub](https://github.com/sierra-research/tau2-bench)
- [tau2-bench Paper](https://arxiv.org/abs/2506.07982)
- [tau2-bench Leaderboard](https://taubench.com)
- [SkyRL Documentation](https://github.com/SkyRLTeam/SkyRL)
