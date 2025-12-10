"""Convert the Lost in Conversation sharded instructions into a SkyRL dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_INPUT = Path(__file__).resolve().parents[3] / "lost_in_conversation" / "data" / "sharded_instructions_600.json"
DEFAULT_OUTPUT = Path(__file__).with_name("lost_in_conversation.jsonl")


def load_samples(input_path: Path) -> List[Dict[str, Any]]:
    with input_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_records(samples: List[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for sample in samples:
        records.append(
            {
                "prompt": [],
                "env_class": "lost_in_conversation",
                "env_extras": {
                    "sample": sample,
                    "assistant_model": args.assistant_model,
                    "system_model": args.system_model,
                    "user_model": args.user_model,
                    "assistant_temperature": args.assistant_temperature,
                    "user_temperature": args.user_temperature,
                    "dataset_fn": str(args.input_path),
                    "data_source": sample.get("task"),
                },
            }
        )
    return records


def write_records(records: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_path", type=Path, default=DEFAULT_INPUT, help="Path to sharded_instructions_600.json")
    parser.add_argument("--output_path", type=Path, default=DEFAULT_OUTPUT, help="Where to write the SkyRL dataset")
    parser.add_argument("--assistant_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--system_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--user_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--assistant_temperature", type=float, default=1.0)
    parser.add_argument("--user_temperature", type=float, default=1.0)

    args = parser.parse_args()
    samples = load_samples(args.input_path)
    records = build_records(samples, args)
    write_records(records, args.output_path)
    print(f"Wrote {len(records)} records to {args.output_path}")