from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from skyrl_gym.envs.gsm8k import  utils
from typing import Dict, Any
from omegaconf import DictConfig


class CollabLLMEnv(BaseTextEnv):

    def __init__(self,
                 env_config: Dict[str, Any] = {},
                 extras: Dict[str, Any] = {}
    ):
        super().__init__()
        assert "reward_spec" in extras, "reward_spec field is required"
        assert "max_turns" in extras, "max_turns is required in extras field"
        self.max_turns = extras["max_turns"]

    def _get_reward(self, action: str) -> float:
        return utils.compute_score(action, self.ground_truth)

    def step(self, action: str) -> BaseTextEnvStepOutput:
        done = True  # always done after one step
        reward = self._get_reward(action)
        # No observation in gsm8k, and no tool call
        return BaseTextEnvStepOutput(observations=[], reward=reward, done=done, metadata={})
