import copy
import time
from typing import Any, Dict, Optional, List

from omegaconf import DictConfig
from openai import OpenAI
from loguru import logger

from skyrl_gym.envs.base_text_env import (
    BaseTextEnv,
    BaseTextEnvStepOutput,
    ConversationType,
)
from skyrl_gym.envs.collabllm.llm_as_a_judge_prompt import JUDGE_PROMPT
from skyrl_gym.envs.collabllm.llm_as_a_judge_utils import extract_outer_dict
from skyrl_train.generators.user_simulator_prompt import USER_SIM_SYSPROMPT
from skyrl_train.generators.user_simulator import usersim_stringfy


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
            self._judge_model = env_config.llm_judge.model_name
            self._judge_temp = env_config.llm_judge.temperature
        else:
            raise NotImplementedError("Only local vllm model is supported for Llm as a judge env.")
        self._judge_max_retries = 8
        self._conversation: Optional[ConversationType] = None

    def init(self, prompt):
        self._conversation = prompt
        return prompt, {}

    def _get_reward(
        self,
        action: str,
        conversation: Optional[ConversationType] = None,
        debug: bool = False,
    ) -> float:
        conv = conversation if conversation is not None else self._conversation
        conversation_text = _format_conversation(conv)
        message = JUDGE_PROMPT.format(
            question=self._original_query,
            chat_history=conversation_text,
            response=action,
        )

        if debug: # print messages sent to LLM judge
            logger.info(f"messages to LLM judge: {message}")

        for attempt in range(1, self._judge_max_retries + 1):
            try:
                response: Optional[str] = None
                completion = self._judge_client.chat.completions.create(
                    model=self._judge_model,
                    messages=[{"role": "user", "content": message}],
                    temperature=self._judge_temp,
                )
                response = completion.choices[0].message.content.strip()
                if debug:
                    logger.info(f"LLM judge response: {response}")
                parsed_dict = extract_outer_dict(response)
                score = parsed_dict.get("score")
                return float(score)
            except Exception:
                if response is None:
                    logger.exception("Attempt %d/%d: LLM judge failed.", attempt, self._judge_max_retries)
                else:
                    logger.exception("Attempt %d/%d: extracting from LLM judge response failed.", attempt, self._judge_max_retries)
                if attempt < self._judge_max_retries:
                    time.sleep(2 ** (attempt - 1))
                else:
                    logger.error("LLM judge failed after %d attempts. Returning reward 0.0.", self._judge_max_retries)
                    return 0.0
        
        logger.error("LLM judge did not return a valid score. Returning reward 0.0.")
        return 0.0

    def step(self, action: str) -> BaseTextEnvStepOutput:
        done = True
        reward = self._get_reward(action)
        assert self._conversation is not None, "Environment not initialized with a prompt."
        self._conversation.append({"role": "assistant", "content": action})
        return BaseTextEnvStepOutput(observations=[], reward=reward, done=done, metadata={})


class CollabLLMLLMJudgeMultiTurnEnv(CollabLLMLLMJudgeEnv):
    """
    LLM-as-a-judge environment for collaborative LLM (first implementation: math500) + multi-turn
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
            user_simulator:
                enabled: false
                model_name: "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
                base_url: "http://localhost:{port}/v1"
                is_local: true
                local_port: 8002
                temperature: 0.7
                formatting:
                user_template: "User: "
                ai_template: "AI: "
                terminal_signal: "<END_OF_CONVERSATION>"
        """
        super().__init__(env_config, extras)

        user_simulator_cfg = env_config.get("user_simulator")
        assert user_simulator_cfg is not None and user_simulator_cfg.enabled, (
            "user_simulator must be enabled in env_config for multi-turn CollabLLM environment."
        )
        if user_simulator_cfg.is_local:
            base_url = user_simulator_cfg.base_url.format(port=user_simulator_cfg.local_port)
            self._user_simulator_client = OpenAI(base_url=base_url)
        else:
            raise NotImplementedError("Only local vllm model is supported for UserSimulator.")

        self._user_simulator_model = user_simulator_cfg.model_name
        self._user_simulator_temperature = user_simulator_cfg.temperature
        self._user_simulator_formatting = user_simulator_cfg.formatting
        self._user_terminal_signal = self._user_simulator_formatting.get(
            "terminal_signal", "<END_OF_CONVERSATION>"
        )
        self._user_simulator_max_retries = 8

        extra_info = extras.get("extra_info", {})
        self._task_desc = extra_info["task_desc"]
        # TODO : make max_turns synced with hydra config generator max_turns
        self.max_turns = env_config.get("max_turns", extra_info.get("max_turns", 4))

    def init(self, prompt: ConversationType):
        self.turns = 0
        return super().init(prompt)

    def _stringify_conversation_for_user_simulator(self) -> str:
        return usersim_stringfy(
            conversation=self._conversation or [],
            formatting_cfg=self._user_simulator_formatting,
        )

    def _simulate_user_action(self, debug: bool = False) -> str:
        system_prompt = USER_SIM_SYSPROMPT.format(
            task_desc=self._task_desc,
            single_turn_prompt=self._original_query,
            chat_history=self._stringify_conversation_for_user_simulator(),
            terminal_signal=self._user_terminal_signal,
        )

        for attempt in range(1, self._user_simulator_max_retries + 1):
            try:
                response_text: Optional[str] = None
                completion = self._user_simulator_client.chat.completions.create(
                    model=self._user_simulator_model,
                    messages=[{"role": "user", "content": system_prompt}],
                    temperature=self._user_simulator_temperature,
                )
                response_text = completion.choices[0].message.content.strip()
                if debug:
                    logger.info(f"user simulator raw response: {response_text}")
                parsed = extract_outer_dict(response_text)
                return parsed["response"]
            except Exception:
                if response_text is None:
                    logger.exception(
                        f"Attempt {attempt}/{self._user_simulator_max_retries}: user simulator failed."
                    )
                else:
                    logger.exception(
                        f"Attempt {attempt}/{self._user_simulator_max_retries}: extracting user simulator response failed "
                        f"with content: {response_text}"
                    )
                if attempt < self._user_simulator_max_retries:
                    time.sleep(2 ** (attempt - 1))
                else:
                    logger.error(f"User simulator failed after {self._user_simulator_max_retries} attempts.")
                    return self._user_terminal_signal

        return self._user_terminal_signal

    def step(self, action: str, debug: bool = True) -> BaseTextEnvStepOutput:
        assert self._conversation is not None, "Environment not initialized with a prompt."
        if debug:
            logger.info(f"Assistant action at turn {self.turns + 1}: {action}")
            logger.info(f"Conversation so far: {self._conversation}")

        self._conversation.append({"role": "assistant", "content": action})
        self.turns += 1
        done = self.turns >= self.max_turns
        observations: ConversationType = []
        reward = 0.0

        if not done: # has not reached max_turns
            user_reply = self._simulate_user_action(debug=debug)
            if user_reply.strip() == self._user_terminal_signal:
                done = True
            else:
                user_message = {"role": "user", "content": user_reply}
                self._conversation.append(user_message)
                observations = [user_message]

        if done:
            # NOTE: conversation_for_reward should end with assistant action
            # done when max_turn reached (even if no terminal_signal) or terminal_signal received
            conversation_for_reward: List[Dict[str, str]] = copy.deepcopy(self._conversation)
            reward = self._get_reward(action, conversation=conversation_for_reward, debug=debug)

        return BaseTextEnvStepOutput(
            observations=observations,
            reward=reward,
            done=done,
            metadata={},
        )