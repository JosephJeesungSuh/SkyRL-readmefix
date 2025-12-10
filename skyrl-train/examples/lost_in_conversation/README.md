# Lost in Conversation Training

This example shows how to train with the `lost_in_conversation` SkyRL-Gym environment, which mirrors the sharded-conversation simulator from [microsoft/lost_in_conversation](https://github.com/microsoft/lost_in_conversation).

## Prepare data

Use the helper script to convert the sharded instructions shipped with the upstream repository into a SkyRL-compatible JSONL file:

```bash
# Creates skyrl-train/examples/lost_in_conversation/lost_in_conversation.jsonl by default
python build_dataset.py
```

You can override the input path, output location, and the default system/user/assistant model identifiers with CLI flags (see `--help`).

## Run training

The shell script below configures `skyrl_train.entrypoints.main_base` for colocated GRPO training. It assumes you have set `OPENAI_API_KEY` (or Azure equivalents) so the environment can call the Lost in Conversation system/user agents.

```bash
bash run_lost_in_conversation.sh
```

The script uses the generated JSONL file, points the environment at the upstream data, and keeps the exact system/user verification protocols from the original simulator. Adjust the model names, batch sizes, and sampling parameters as needed for your hardware.