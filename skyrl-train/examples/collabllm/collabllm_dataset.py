import argparse
import json
from pathlib import Path
from typing import Dict, Any

from datasets import load_dataset

DATA_SOURCE="huggingfaceh4--math-500"
ENV_CLASS="collabllm_math_500_multiturn" # ENV_CLASS="gsm8k"
DATASET_NAME = "HuggingFaceH4/MATH-500"
SEED = 42
TRAIN_RATIO = 0.85


def reformat_example(
    ex: Dict[str, Any],
    data_source: str,
    env_class: str,
) -> Dict[str, Any]:

    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": ex['problem']}],
        "env_class": env_class,
        "reward_spec": {
            "method": "rule",
            "ground_truth": ex.get("answer"),
            "initial_question": ex.get("problem"),
        },
        "extra_info": {
            "task_desc": "question answering",
            "math_subject": ex.get("subject"),
            "difficulty": ex.get("level"),
            "unique_id": ex.get("unique_id"),
            "golden_answer": ex.get("solution"),
        },
    }

def main(out_dir: Path):
    
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(DATASET_NAME, split="test")

    # deduplicate based on question string
    def _prompt_key(example):
        return json.dumps(example['problem'], sort_keys=True, ensure_ascii=True)
    seen = set()
    def _keep_unique_user_turn0_query(example):
        key = _prompt_key(example)
        if key in seen:
            return False
        seen.add(key)
        return True
    
    ds = ds.filter(_keep_unique_user_turn0_query)
    
    split = ds.train_test_split(
        test_size=1.0 - TRAIN_RATIO,
        seed=SEED,
        shuffle=True,
    )

    def _map_fn(example):
        return reformat_example(example,
                                data_source=DATA_SOURCE,
                                env_class=ENV_CLASS)
    
    original_cols = split["train"].column_names
    train_ds = split["train"].map(_map_fn, remove_columns=original_cols)
    val_ds = split["test"].map(_map_fn, remove_columns=original_cols)

    train_path = out_dir / "train.parquet"
    val_path = out_dir / "validation.parquet"

    train_ds.to_parquet(str(train_path))
    val_ds.to_parquet(str(val_path))

    print(f"Saved:\n  {train_path} (rows={len(train_ds)})\n  {val_path} (rows={len(val_ds)})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="~/data/collabllm")
    args = parser.parse_args()
    main(out_dir=Path(args.output_dir).expanduser())
