from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from omegaconf import DictConfig

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, ConversationType

# Ensure the Lost in Conversation modules can be imported
_LOST_IN_CONVO_DIR = Path(__file__).resolve().parents[3] / "lost_in_conversation"
if str(_LOST_IN_CONVO_DIR) not in sys.path:
    sys.path.insert(0, str(_LOST_IN_CONVO_DIR))


def _import_lost_in_conversation():
    @contextlib.contextmanager
    def _cwd(path: Path):
        previous = Path.cwd()
        os.chdir(path)
        try:
            yield
        finally:
            os.chdir(previous)

    with _cwd(_LOST_IN_CONVO_DIR):
        from system_agent import SystemAgent  # type: ignore  # noqa: E402
        from tasks import get_task  # type: ignore  # noqa: E402
        from user_agent import UserAgent  # type: ignore  # noqa: E402
        from utils import date_str, extract_conversation  # type: ignore  # noqa: E402

    return SystemAgent, get_task, UserAgent, date_str, extract_conversation


SystemAgent, get_task, UserAgent, date_str, extract_conversation = _import_lost_in_conversation()


class LostInConversationEnv(BaseTextEnv):
    """Gym environment mirroring the sharded-conversation simulator.

    This environment wraps the logic in ``lost_in_conversation/simulator_sharded.py``
    so that SkyRL can train agents while using the exact user/system protocols
    from the original repository.
    """

    def __init__(self, env_config: DictConfig, extras: Dict[str, Any] = {}):
        super().__init__()

        self.env_config = env_config
        self.sample = extras.get("sample")
        self.dataset_fn: Optional[str] = extras.get(
            "dataset_fn", env_config.get("dataset_path", str(_LOST_IN_CONVO_DIR / "data" / "sharded_instructions_600.json"))
        )

        assert self.sample is not None, "A `sample` dict from the Lost in Conversation dataset is required."

        self.assistant_model: str = extras.get("assistant_model", env_config.get("assistant_model", "gpt-4o-mini"))
        self.system_model: str = extras.get("system_model", env_config.get("system_model", "gpt-4o-mini"))
        self.user_model: str = extras.get("user_model", env_config.get("user_model", "gpt-4o-mini"))
        self.assistant_temperature: float = float(
            extras.get("assistant_temperature", env_config.get("assistant_temperature", 1.0))
        )
        self.user_temperature: float = float(extras.get("user_temperature", env_config.get("user_temperature", 1.0)))

        # Keep execution aligned with the upstream simulator by working inside the repo directory
        self.repo_dir = _LOST_IN_CONVO_DIR
        self.task_name = self.sample["task"]

        with self._use_repo_dir():
            self.task = get_task(self.task_name)
            self.user_agent = UserAgent(self.task, self.user_model)
            self.system_agent = SystemAgent(self.task_name, self.system_model, self.sample)
            self.system_message = self.task.generate_system_prompt(self.sample)
            self.answer_description = self.task.get_answer_description()

        self.trace: List[Dict[str, Any]] = []
        self.chat_history: ConversationType = []
        self.last_score: Optional[float] = None
        self.is_correct: Optional[bool] = None

    @contextlib.contextmanager
    def _use_repo_dir(self):
        prev_dir = Path.cwd()
        os.chdir(self.repo_dir)
        try:
            yield
        finally:
            os.chdir(prev_dir)

    def _all_shards_revealed(self) -> bool:
        shards = self.sample.get("shards", [])
        shard_ids = {shard["shard_id"] for shard in shards}
        revealed = {
            msg["content"]["shard_id"]
            for msg in self.trace
            if msg.get("role") == "log"
            and isinstance(msg.get("content"), dict)
            and msg["content"].get("type") == "shard_revealed"
        }
        return shard_ids.issubset(revealed) and len(revealed) == len(shard_ids)

    def init(self, prompt: ConversationType) -> Tuple[ConversationType, Dict[str, Any]]:  # type: ignore[override]
        # Initialize the conversation with the system prompt and the first user shard
        self.trace = [{"role": "system", "content": self.system_message, "timestamp": date_str()}]
        self.chat_history = extract_conversation(self.trace, to_str=False)

        with self._use_repo_dir():
            user_response, shard_revealed_id, cost_usd = self.user_agent.generate_response(
                self.trace, self.sample, temperature=self.user_temperature
            )

        self.trace.append({"role": "user", "content": user_response, "timestamp": date_str(), "cost_usd": cost_usd})
        if shard_revealed_id != -1:
            self.trace.append(
                {"role": "log", "content": {"type": "shard_revealed", "shard_id": shard_revealed_id}, "timestamp": date_str()}
            )

        self.chat_history = extract_conversation(self.trace, to_str=False)
        metadata = {
            "task": self.task_name,
            "task_id": self.sample.get("task_id"),
            "dataset_fn": self.dataset_fn,
            "assistant_model": self.assistant_model,
        }
        return self.chat_history, metadata

    def _evaluate_answer(self) -> Tuple[float, bool]:
        with self._use_repo_dir():
            extracted_answer = self.system_agent.extract_answer(self.trace)
            evaluation_return = self.task.evaluator_function(extracted_answer, self.sample)

        score = evaluation_return.get("score", 1.0 if evaluation_return.get("is_correct") else 0.0)
        is_correct = evaluation_return.get("is_correct", score == 1.0)

        self.trace.append(
            {
                "role": "log",
                "content": {
                    "type": "answer-evaluation",
                    "exact_answer": extracted_answer,
                    "is_correct": is_correct,
                    "score": score,
                    "evaluation_return": evaluation_return,
                },
                "timestamp": date_str(),
            }
        )
        return score, is_correct

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turns += 1
        self.trace.append({"role": "assistant", "content": action, "timestamp": date_str()})

        reward = 0.0
        done = False
        new_observations: ConversationType = []
        metadata: Dict[str, Any] = {}

        # System verification mirrors the original simulator
        with self._use_repo_dir():
            system_verification_response, verification_cost_usd = self.system_agent.verify_system_response(self.trace)
        self.trace.append(
            {
                "role": "log",
                "content": {"type": "system-verification", "response": system_verification_response},
                "timestamp": date_str(),
                "cost_usd": verification_cost_usd,
            }
        )
        metadata["system_verification"] = system_verification_response

        if system_verification_response.get("response_type") == "answer_attempt":
            score, is_correct = self._evaluate_answer()
            self.last_score = score
            self.is_correct = is_correct
            reward = float(score)
            done = bool(is_correct)

        if not done and not self._all_shards_revealed():
            with self._use_repo_dir():
                user_response, shard_revealed_id, cost_usd = self.user_agent.generate_response(
                    self.trace, self.sample, temperature=self.user_temperature
                )
            self.trace.append(
                {"role": "user", "content": user_response, "timestamp": date_str(), "cost_usd": cost_usd}
            )
            if shard_revealed_id != -1:
                self.trace.append(
                    {
                        "role": "log",
                        "content": {"type": "shard_revealed", "shard_id": shard_revealed_id},
                        "timestamp": date_str(),
                    }
                )
            new_observations = extract_conversation(self.trace, to_str=False)
        else:
            done = done or self._all_shards_revealed()

        return BaseTextEnvStepOutput(
            observations=new_observations,
            reward=reward,
            done=done,
            metadata=metadata,
        )

    def get_metrics(self) -> Dict[str, Any]:
        return {"score": self.last_score, "is_correct": self.is_correct}

    @staticmethod
    def aggregate_metrics(metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not metrics:
            return {}
        valid = [m for m in metrics if m.get("score") is not None]
        if not valid:
            return {}
        avg_score = sum(float(m["score"]) for m in valid) / len(valid)
        accuracy = sum(1 for m in valid if m.get("is_correct")) / len(valid)
        return {"avg_score": avg_score, "accuracy": accuracy}