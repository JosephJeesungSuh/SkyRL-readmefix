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

    # Load all tasks
    with open(tasks_file, "r") as f:
        all_tasks = json.load(f)

    # Load split information
    with open(split_file, "r") as f:
        splits = json.load(f)

    # Get task IDs for this split
    if task_split not in splits:
        raise ValueError(f"Unknown task split: {task_split}. Available: {list(splits.keys())}")

    task_ids = splits[task_split]

    # Filter tasks by ID
    tasks = [task for task in all_tasks if task["id"] in task_ids]

    return tasks


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
    """
    task_id = task["id"]

    # Create initial prompt based on user scenario
    # In tau2-bench, the user simulator handles the initial message
    # So we typically start with an empty prompt or system message
    initial_prompt = []

    # Extract user scenario info for metadata
    user_scenario = task.get("user_scenario", {})
    instructions = user_scenario.get("instructions", {})

    return {
        "data_source": f"tau2bench-{domain}",
        "prompt": initial_prompt,
        "env_class": env_class,
        "task_id": task_id,  # Critical: tau2-bench needs this to load the task
        "domain": domain,  # For multi-domain environments
        "reward_spec": {
            "method": "tau2bench",  # Reward comes from tau2-bench evaluation
        },
        "extra_info": {
            "task_id": task_id,
            "domain": domain,
            "task_description": task.get("description", {}),
            "user_scenario": user_scenario,
            "reason_for_call": instructions.get("reason_for_call", ""),
            "known_info": instructions.get("known_info", ""),
            "evaluation_criteria": task.get("evaluation_criteria", {}),
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
