"""
Dataset preparation script for Tau2-Bench environments.
This script loads tasks from tau2-bench's data directory and converts them
into the format expected by SkyRL training pipeline.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd

# Tau2-bench data paths
TAU2_DATA_DIR = Path("/nas/ucb/jjssuh/projs/tau2-bench/data/tau2")
DOMAINS = ["airline", "retail", "telecom", "mock"]

def load_domain_tasks(domain: str, task_split: str = "train") -> List[Dict[str, Any]]:
    """
    Load tasks for a specific domain and split from tau2-bench.
    Args:
        domain: Domain name (airline, retail, telecom, mock)
        task_split: Task split to use (train, test, base)
    Returns:
        List of task dictionaries
    """
    domain_dir = TAU2_DATA_DIR / "domains" / domain
    tasks_file = domain_dir / "tasks.json"
    split_file = domain_dir / "split_tasks.json"
    with open(tasks_file, "r") as f:
        all_tasks = json.load(f)
    with open(split_file, "r") as f:
        splits = json.load(f)
    if task_split not in splits:
        raise ValueError(f"Unknown task split: {task_split}. Available: {list(splits.keys())}")
    task_ids = splits[task_split]
    tasks = [task for task in all_tasks if task["id"] in task_ids]
    return tasks


def _pretty_print(data: Any) -> None:
    print(json.dumps(data, indent=2))

def reformat_task_for_skyrl(
    task: Dict[str, Any],
    domain: str,
    env_class: str,
) -> Dict[str, Any]:
    """
    Convert a tau2-bench task to SkyRL format.
    Args:
        task: Task dictionary from tau2-bench
        domain: Domain name
        env_class: Environment class to use
    Returns:
        Reformatted task dictionary for SkyRL
    Note: the _pretty_print example of task structure is as follows:
    {
        "id": "0",
        "description": {
            "purpose": "Testing that agent refuses to proceed with a cancellation that is not allowed even if User mentions that she had been told she didn't need insurance.",
            "relevant_policies": null,
            "notes": null
        },
        "user_scenario": {
            "persona": null,
            "instructions": {
            "task_instructions": "If Agent tells you that cancellation is not possible,\nmention that you were told that you didn't need to get insurance because your previous trip was booked with the same agency with insurance.\n\nYou don't want to cancel if you don't get a refund.",
            "domain": "airline",
            "reason_for_call": "You want to cancel reservation EHGLP3. \n\nIt may be more than 24 hours after booking, but it is ok because you were out of town for that time.",
            "known_info": "You are Emma Kim.\nYour user id is emma_kim_9957.",
            "unknown_info": null
            }
        },
        "initial_state": null,
        "evaluation_criteria": {
            "actions": [],
            "communicate_info": [],
            "nl_assertions": [
            "Agent should refuse to proceed with the cancellation."
            ]
        },
        "annotations": null
    }, the the sceneario description the the user simulator looks like:
    <scenario>
    Instructions:
        Domain: airline -- return.extra_info.domain
        Reason for call: -- return.extra_info.reason_for_call
            You want to cancel reservation EHGLP3. 
        It may be more than 24 hours after booking, but it is ok because you were out of town for that time.
        Known info: -- return.extra_info.known_info
            You are Emma Kim.
            Your user id is emma_kim_9957.
        Task instructions: -- return.extra_info.task_instructions
            If Agent tells you that cancellation is not possible,
            mention that you were told that you didn't need to get insurance because your previous trip was booked with the same agency with insurance.

            You don't want to cancel if you don't get a refund.
    </scenario>
    """
    # Extract user scenario info for metadata
    user_scenario = task.get("user_scenario", {})
    instructions = user_scenario.get("instructions", {})

    return {
        "data_source": f"tau2bench-{domain}",
        # In tau2-bench, the user simulator handles the initial conversation
        "prompt": [{"role": "user", "content": 'DUMMY'}],
        # env_class example : 'tau2bench_airline'
        "env_class": env_class,
        "reward_spec": {"method": "tau2bench"}, # Reward comes from tau2-bench evaluation
        "extra_info": {
            # task unique id
            "task_id": task["id"],
            # formulating user simulator system prompt
            "domain": domain,
            "reason_for_call": instructions.get("reason_for_call"),
            "known_info": instructions.get("known_info"),
            "task_instructions": instructions.get("task_instructions"),
            # all other info originally in the task
            "description": task.get("description", {}),
            "user_scenario": user_scenario,
            "initial_state": task.get("initial_state"),
            "evaluation_criteria": task.get("evaluation_criteria"),
            "annotations": task.get("annotations"),
        },
    }


def create_dataset_for_domain(
    domain: str,
    env_class: str,
    output_dir: Path,
    train_split: str = "train",
    val_split: str = "test",
):
    """
    Create train/validation datasets for a single domain.

    Args:
        domain: Domain name (airline, retail, telecom, mock)
        env_class: Environment class to use
        output_dir: Output directory for parquet files
        train_split: Split to use for training
        val_split: Split to use for validation
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tasks for train and val splits
    train_tasks = load_domain_tasks(domain, train_split)
    val_tasks = load_domain_tasks(domain, val_split)

    print(f"{domain} domain: {len(train_tasks)} train tasks, {len(val_tasks)} val tasks")

    # Reformat tasks
    train_data = [reformat_task_for_skyrl(task, domain, env_class) for task in train_tasks]
    val_data = [reformat_task_for_skyrl(task, domain, env_class) for task in val_tasks]

    # Convert to pandas DataFrame
    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)

    # Save to parquet
    train_path = output_dir / "train.parquet"
    val_path = output_dir / "validation.parquet"

    train_df.to_parquet(str(train_path))
    val_df.to_parquet(str(val_path))

    print(f"Saved:\n  {train_path} (rows={len(train_df)})\n  {val_path} (rows={len(val_df)})")


def create_multidomain_dataset(
    domains: List[str],
    env_class: str,
    output_dir: Path,
    train_split: str = "train",
    val_split: str = "test",
):
    """
    Create a multi-domain dataset combining all domains.

    Args:
        domains: List of domain names to include
        env_class: Environment class to use (should be multidomain variant)
        output_dir: Output directory for parquet files
        train_split: Split to use for training
        val_split: Split to use for validation
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    train_data = []
    val_data = []

    for domain in domains:
        # Load tasks
        train_tasks = load_domain_tasks(domain, train_split)
        val_tasks = load_domain_tasks(domain, val_split)

        print(f"{domain} domain: {len(train_tasks)} train tasks, {len(val_tasks)} val tasks")

        # Reformat and add to combined data
        train_data.extend([reformat_task_for_skyrl(task, domain, env_class) for task in train_tasks])
        val_data.extend([reformat_task_for_skyrl(task, domain, env_class) for task in val_tasks])

    # Convert to pandas DataFrame
    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)

    # Save to parquet
    train_path = output_dir / "train.parquet"
    val_path = output_dir / "validation.parquet"

    train_df.to_parquet(str(train_path))
    val_df.to_parquet(str(val_path))

    print(f"\nMulti-domain dataset saved:")
    print(f"  {train_path} (rows={len(train_df)})")
    print(f"  {val_path} (rows={len(val_df)})")


def main(args):
    output_dir = Path(args.output_dir).expanduser()

    if args.multidomain:
        # Create multi-domain dataset
        domains = args.domains if args.domains else DOMAINS
        print(f"Creating multi-domain dataset with domains: {domains}")
        create_multidomain_dataset(
            domains=domains,
            env_class=args.env_class,
            output_dir=output_dir,
            train_split=args.train_split,
            val_split=args.val_split,
        )
    else:
        # Create single-domain dataset
        if not args.domain:
            raise ValueError("Must specify --domain for single-domain dataset")
        print(f"Creating dataset for domain: {args.domain}")
        create_dataset_for_domain(
            domain=args.domain,
            env_class=args.env_class,
            output_dir=output_dir,
            train_split=args.train_split,
            val_split=args.val_split,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create tau2-bench dataset for SkyRL training"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="~/data/tau2bench",
        help="Output directory for parquet files",
    )
    parser.add_argument(
        "--domain",
        type=str,
        choices=DOMAINS,
        help="Domain to create dataset for (for single-domain mode)",
    )
    parser.add_argument(
        "--env_class",
        type=str,
        default="tau2bench_airline",
        help="Environment class to use (e.g., tau2bench_airline, tau2bench_multidomain)",
    )
    parser.add_argument(
        "--multidomain",
        action="store_true",
        help="Create multi-domain dataset combining all domains",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        choices=DOMAINS,
        help="Domains to include in multi-domain dataset (default: all)",
    )
    parser.add_argument(
        "--train_split",
        type=str,
        default="train",
        help="Task split to use for training (train, test, base)",
    )
    parser.add_argument(
        "--val_split",
        type=str,
        default="test",
        help="Task split to use for validation (train, test, base)",
    )

    args = parser.parse_args()
    main(args)
