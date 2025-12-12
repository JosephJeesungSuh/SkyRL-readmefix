# last update: 2025-12-12 (Fri)

import argparse
import json
import os
import random
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List

from datasets import Dataset


REPO_ROOT = Path(__file__).resolve().parents[2].parent
LIC_ROOT = REPO_ROOT / "lost_in_conversation"
sys.path.insert(0, str(LIC_ROOT))


def build_initial_prompt(sample: Dict) -> List[Dict[str, str]]:
    """Create the initial system + user turn that the assistant sees.

    This mirrors the behavior in ``ConversationSimulatorSharded`` where the
    system prompt is emitted once and the first shard is provided as the first
    user turn before the assistant responds.
    """

    @contextmanager
    def _chdir(path: Path):
        prev = Path.cwd()
        os.chdir(path)
        try:
            yield
        finally:
            os.chdir(prev)

    with _chdir(LIC_ROOT):
        from tasks import get_task 
        task = get_task(sample["task"])
        system_prompt = task.generate_system_prompt(sample)
        first_shard = sample["shards"][0]["shard"]
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": first_shard},
        ]


def make_split(data: List[Dict], split_ratio: float) -> tuple[list[Dict], list[Dict]]:
    split = int(len(data) * split_ratio)
    return data[:split], data[split:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default=str(LIC_ROOT / "data/sharded_instructions_600.json"))
    parser.add_argument("--output_dir", default=os.path.expanduser("~/data/lost_in_conversation"))
    parser.add_argument("--train_ratio", type=float, default=0.85)
    args = parser.parse_args()

    with open(args.dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    random.Random(42).shuffle(data)
    train, validation = make_split(data, args.train_ratio)

    def to_row(sample: Dict, split: str):
        """
        sample has different nested structure based on the task (e.g., data-to-text, math, etc.)
        data-to-text keys: ['task_id', 'task', 'original_task_id', 'raw_table', 'fewshot_descriptions', 'table_html', 'metadata', 'table_highlighted_html', 'references', 'shards']
        math keys: ['question', 'answer', 'task_id', 'shards', 'task']
        """
        prompt = build_initial_prompt(sample)
        return {
            "prompt": prompt,
            "env_class": "lost_in_conversation",
            "dataset_fn": str(args.dataset_path),
            "task": str(sample.get("task") or ""),
            "task_id": str(sample.get("task_id") or ""),
            "split": split,
            "num_shards": int(len(sample.get("shards", []))),
            "sample_json": json.dumps(sample, ensure_ascii=False),
        }
        
    train_rows = [to_row(sample, "train") for sample in train]
    validation_rows = [to_row(sample, "validation") for sample in validation]

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    import pdb; pdb.set_trace()

    Dataset.from_list(train_rows).to_parquet(output_dir / "train.parquet")
    Dataset.from_list(validation_rows).to_parquet(output_dir / "validation.parquet")

    print(f"Wrote {len(train_rows)} training rows and {len(validation_rows)} validation rows to {output_dir}")


if __name__ == "__main__":
    print("current repo root: ", REPO_ROOT)
    main()