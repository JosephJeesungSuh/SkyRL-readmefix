import time
from typing import Any, Dict, Optional
from pathlib import Path

from omegaconf import OmegaConf, DictConfig
from openai import OpenAI
from loguru import logger

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from skyrl_gym.envs.collabllm.llm_as_a_judge_prompt import JUDGE_PROMPT
from skyrl_gym.envs.collabllm.llm_as_a_judge_utils import extract_outer_dict

CURRENT_DIR = Path(__file__).parent


def _format_conversation(conversation):
    """
    Format the historical conversation into a string for LLM judge prompt.
    """
    assert len(conversation) > 0 and conversation[-1]["role"] == "user", (
        "Conversation must be non-empty and the last message must be from the user."
    )
    formatted_turns = []
    for message in conversation:
        role = message.get("role", "")
        content = message.get("content", "")
        formatted_turns.append(f"{role.upper()}: {content}")
    return "\n\n".join(formatted_turns)


class CollabLLMLLMJudgeEnv(BaseTextEnv):
    """
    LLM-as-a-judge environment for collaborative LLM (first implementation: math500)
    The judge LLM inspects the entire conversation between the user simulator and the
    assistant, together with the ground-truth solution, and produces a binary score.
    """

    def __init__(self, env_config: DictConfig, extras: Dict[str, Any] = {}):
        """
        env_config (DictConfig): comes from skyrl-train/skyrl_train/config/skyrl_gym_config/default.yaml
        current structure:
          llm_judge:
            enabled: false
            model_name: "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
            base_url: "http://localhost:{port}/v1"
            is_local: true
            local_port: 8002
            temperature: 0.0
        """
        super().__init__()

        assert "reward_spec" in extras, "reward_spec field is required"
        assert "ground_truth" in extras["reward_spec"], "ground_truth is required in reward_spec field"
        self._ground_truth = extras["reward_spec"]["ground_truth"]
        self._original_query = extras["reward_spec"]["initial_question"]

        assert env_config.llm_judge.enabled, "llm_judge must be enabled in env_config"
        if env_config.llm_judge.is_local:
            base_url = env_config.llm_judge.base_url.format(port=env_config.llm_judge.local_port)
            self._judge_client = OpenAI(base_url=base_url)
            self._model = env_config.llm_judge.model_name
            self._temperature = env_config.llm_judge.temperature
        else:
            raise NotImplementedError("Only local vllm model is supported for Llm as a judge env.")
        self._max_retries = 8
        self._initial_conversation = None

    def init(self, prompt):
        self._initial_conversation = prompt
        return prompt, {}

    def _get_reward(self, action: str, debug: bool = False) -> float:
        conversation_text = _format_conversation(self._initial_conversation)
        message = JUDGE_PROMPT.format(
            question=self._original_query,
            chat_history=conversation_text,
            response=action,
        )

        if debug: # print messages sent to LLM judge
            logger.info(f"messages to LLM judge: {message}")

        for attempt in range(1, self._max_retries + 1):
            try:
                response: Optional[str] = None
                completion = self._judge_client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": message}],
                    temperature=self._temperature,
                )
                response = completion.choices[0].message.content.strip()
                if debug:
                    logger.info(f"LLM judge response: {response}")
                parsed_dict = extract_outer_dict(response)
                score = parsed_dict.get("score")
            except Exception:
                if response is None:
                    logger.exception("Attempt %d/%d: LLM judge failed.", attempt, self._max_retries)
                else:
                    logger.exception("Attempt %d/%d: extracting from LLM judge response failed.", attempt, self._max_retries)
                if attempt < self._max_retries:
                    time.sleep(2 ** (attempt - 1))
                else:
                    logger.error("LLM judge failed after %d attempts. Returning reward 0.0.", self._max_retries)
                    return 0.0
        
        return float(score)

    def step(self, action: str) -> BaseTextEnvStepOutput:
        done = True
        reward = self._get_reward(action, debug=True)
        self._initial_conversation.append({"role": "assistant", "content": action})
        return BaseTextEnvStepOutput(observations=[], reward=reward, done=done, metadata={})
