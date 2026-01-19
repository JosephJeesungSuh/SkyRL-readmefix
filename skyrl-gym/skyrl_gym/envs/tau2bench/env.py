import copy
import time
from typing import Any, Dict, List, Optional

import gymnasium as gym
from omegaconf import DictConfig
from loguru import logger

from skyrl_gym.envs.base_text_env import (
    BaseTextEnv,
    BaseTextEnvStepOutput,
    ConversationType,
)
from tau2.gym import register_gym_agent, TAU_BENCH_ENV_ID

TAU2_DOMAINS = ["airline", "retail", "telecom", "mock"]

class Tau2BenchEnv(BaseTextEnv):
    """
    Wrapper for tau2-bench gymnasium environment to work with SkyRL's BaseTextEnv interface.
    This environment allows training RL agents on tau2-bench tasks including:
    - airline: Customer service scenarios for airline booking and support
    - retail: E-commerce and retail customer service scenarios
    - telecom: Telecommunications customer support scenarios
    - mock: Simple testing scenarios
    The environment wraps tau2-bench's AgentGymEnv and adapts it to work with
    SkyRL's text-based RL training pipeline.
    """

    def __init__(self, env_config: DictConfig, extras: Dict[str, Any] = {}):
        """
        Initialize the Tau2Bench environment.
        Args:
            env_config: Configuration from skyrl_train/config/skyrl_gym_config/default.yaml
                Expected structure:
                    tau2bench:
                        domain: "airline"  # or "retail", "telecom", "mock"
                        task_split: "train"  # or "test", "base", "val"
                        solo_mode: false  # whether agent works independently
                        user_llm: "gpt-4"  # LLM for user simulator
                        user_llm_args:
                            temperature: 0.7
                        max_turns: 20  # maximum conversation turns
            extras: Additional information including task_id from dataset
                has keys: dict_keys(['data_source', 'reward_spec', 'extra_info', 'max_turns'])
                Expected structure example:
                {'data_source': 'tau2bench-airline',
                 'reward_spec': {'method': 'tau2bench'},
                 'extra_info': {
                    'domain': 'airline',
                    'evaluation_criteria': {
                        'actions': [
                            {'action_id': '44_0',
                             'arguments': {
                                'amount': None,
                                'cabin': None,
                                'date': None,
                                'destination': None,
                                'flight_type': None,
                                'flights': None,
                                'insurance': None,
                                'nonfree_baggages': None,
                                'origin': None,
                                'passengers': None,
                                'payment_id': None,
                                'payment_methods': None,
                                'reservation_id': None,
                                'summary': None,
                                'total_baggages': None,
                                'user_id': 'sophia_silva_7557'},
                             'info': None,
                             'name': 'get_user_details'}, ... 
                        ],
                        'communicate_info': [],
                        'nl_assertions': [
                          'Agent cancels reservation S61CZX.',
                          'The total cost that the. agent mentions is between $1380 and $1390.',
                          'Agent upgrades NM1VX1 to business.',
                          'Agent upgrades H8Q05L to business.',
                          'Agent updates KC18K6 to business.']
                    },
                    'known_info': 'You are Sophia Silva.\nYour user id is sophia_silva_7557.',
                    'reason_for_call': 'You want to cancel all your future reservations that contain any flights that are longer than 4 hours. \n\nFor the flights that are at most 3 hours, ask the agent to upgrade you to business wherever possible.',
                    'task_description': {
                        'notes': None,
                        'purpose': 'Test that agent can collect information about reservation, reason about durations as well as cancellation and upgrades options.',
                        'relevant_policies': None
                    },
                    'task_id': '44',
                    'user_scenario': {
                        'instructions': {
                            'domain': 'airline',
                            'known_info': 'You are Sophia Silva.\nYour user id is sophia_silva_7557.',
                            'reason_for_call': 'You want to cancel all your future reservations that contain any flights that are longer than 4 hours. \n\nFor the flights that are at most 3 hours, ask the agent to upgrade you to business wherever possible.',
                            'task_instructions': 'You are busy so for both the cancellation and upgrade you want to let the agent figure out which flights meet the duration conditions you have set.\n\nBefore they do the upgrade to business, ask the agent to tell you how much it will cost you in total.',
                            'unknown_info': None},
                            'persona': None}
                        },
                 'max_turns': 30}            

        """
        super().__init__()

        # Register tau2-bench gym environments
        # NOTE: this is different from register at skyrl-gym/skyrl_gym/envs/__init__.py
        # This is called to ensure tau2-bench's AgentGymEnv is registered with Gymnasium
        # so we can use gym.make() or directly instantiate it.
        # SkyRL Training Pipeline --> Looks up "tau2bench_airline" in SkyRL registry
        # --> Creates Tau2BenchEnv (our wrapper) --> Tau2BenchEnv.__init__() calls register_gym_agent()
        # --> Creates tau2-bench's AgentGymEnv (internal) --> AgentGymEnv wraps tau2-bench's simulation
        register_gym_agent()

        # Extract configuration
        self.domain = env_config.get("domain")
        assert self.domain in TAU2_DOMAINS, f"Invalid domain: {self.domain}. Must be in {TAU2_DOMAINS}"
        self.task_split = env_config.get("task_split", "train")
        self.solo_mode = env_config.get("solo_mode", False)
        self.max_turns = env_config.get("max_turns", 30)

        # User simulator configuration (only for non-solo mode, which is default)
        user_llm = env_config.get("user_llm", "gpt-4.1-mini")
        user_llm_args = env_config.get("user_llm_args", {"temperature": 0.7})

        assert "task_id" in extras.get('extra_info'), "taubench:: task_id must be provided in extras"
        self.task_id = extras["extra_info"]["task_id"]
        self._extras = extras

        # Use gym.make() as per the official tau2-bench documentation
        # This properly uses the gymnasium registration system
        # Note: task_split is stored but not passed to gym.make() as AgentGymEnv doesn't accept it
        self._gym_env = gym.make(
            TAU_BENCH_ENV_ID,
            domain=self.domain,
            task_id=self.task_id,
            solo_mode=self.solo_mode,
            user_llm=user_llm if not self.solo_mode else None,
            user_llm_args=user_llm_args if not self.solo_mode else None,
        )

        self._conversation: Optional[ConversationType] = None
        self._gym_observation: Optional[str] = None
        self._gym_info: Optional[Dict[str, Any]] = None
        self.turns = 0
        self.max_turns = self.max_turns

    def init(self, prompt: ConversationType):
        """
        Initialize the environment with a prompt.
        Args:
            prompt: Initial conversation history (from dataset - typically a dummy placeholder)
                    The actual conversation is initiated by tau2-bench's user simulator
                    This parameter exists for BaseTextEnv interface compatibility but is not used.
        Returns:
            Tuple of (conversation, metadata)
        """
        # Reset the gym environment - this initializes the tau2-bench task
        # and the user simulator will generate the first message
        self._gym_observation, self._gym_info = self._gym_env.reset()
        self.turns = 0

        # Parse the observation from tau2-bench into conversation format
        # This will contain the user simulator's initial message
        self._conversation = self._parse_observation(self._gym_observation)
        # logger.info(f"Initial conversation from tau2-bench: {self._conversation}")
        ## Example self._conversation: [{
        ## 'role': 'user',
        ## 'content': "Hi! I'd like to book a flight from San Francisco to New York for three passengers. Can you help me with that?"}]
        
        # Note: The prompt from the dataset is just a dummy placeholder (e.g., [{"role": "user", "content": "DUMMY"}])
        # We ignore it and use the actual conversation from tau2-bench's gym environment.
        # The user simulator in tau2-bench has already been initialized with the task-specific
        # information (reason_for_call, known_info, task_instructions) via the task_id.
        # This is different from other SkyRL envs where the prompt provides initial context.
        metadata = {
            "domain": "tau2--" + self.domain,
            "task_id": self.task_id,
            "tools": self._gym_info.get("tools", []),
            "policy": self._gym_info.get("policy", ""), # policy is the <policy></policy> in agent sysprompt.
        }
        return self._conversation, metadata

    def _parse_observation(self, observation: Optional[str]) -> ConversationType:
        """
        Parse tau2-bench observation string into ConversationType format.
        Tau2-bench observations are formatted as:
        "role: content\nrole: content\n..."
        For example:
        'user: Actually, I’m looking to cancel an existing reservation, not book a new flight. Can you assist me with that?'
        We need to convert this to list of {"role": "...", "content": "..."}
        """
        if not observation or observation.strip() == "":
            return []

        messages = []
        current_role = None
        current_content = []
        for line in observation.strip().split("\n"):
            if ": " in line:
                # Check if this is a new message (starts with role:)
                parts = line.split(": ", 1)
                potential_role = parts[0].lower()
                if potential_role in ["user", "assistant", "tool", "system"]:
                    # Save previous message if exists
                    if current_role is not None:
                        messages.append({
                            "role": current_role,
                            "content": "\n".join(current_content).strip()
                        })
                    # Start new message
                    current_role = potential_role
                    current_content = [parts[1]]
                else: # potential_role not in available roles - continuation of previous message
                    if current_role is not None:
                        current_content.append(line)
            else: # also continuation of previous message
                if current_role is not None:
                    current_content.append(line)
        # Add final message
        if current_role is not None:
            messages.append({
                "role": current_role,
                "content": "\n".join(current_content).strip()
            })

        logger.info(f"Parsed observation into messages: {messages}")
        return messages

    def step(self, action: str) -> BaseTextEnvStepOutput:
        """
        Execute one step in the environment.
        Args:
            action: The action string from the agent (message or tool call)
        Returns:
            BaseTextEnvStepOutput with observations, reward, done, and metadata
        """
        assert self._conversation is not None, "Environment not initialized. Call init() first."

        # Add the action to our conversation history
        self._conversation.append({"role": "assistant", "content": action})

        # Step the gym environment
        self._gym_observation, reward, terminated, truncated, self._gym_info = self._gym_env.step(action)
        self.turns += 1

        # Parse new observations from gym
        new_messages = self._parse_observation(self._gym_observation)
        if new_messages:
            self._conversation.extend(new_messages)
        done = terminated or truncated or self.turns >= self.max_turns

        return BaseTextEnvStepOutput(
            observations=new_messages,
            reward=reward,
            done=done,
            metadata={
                "turns": self.turns,
                "terminated": terminated,
                "truncated": truncated,
                "gym_info": self._gym_info,
            }
        )

    def close(self):
        """
        Close the gym environment.
        """
        if hasattr(self, "_gym_env") and self._gym_env is not None:
            self._gym_env.close()

    def get_metrics(self) -> Dict[str, Any]:
        metrics = {"turns": self.turns}
        if self._gym_info and "simulation_run" in self._gym_info:
            sim_run = self._gym_info["simulation_run"]
            if isinstance(sim_run, dict):
                metrics["task_passed"] = sim_run.get("passed", False)
                metrics["evaluation_type"] = sim_run.get("evaluation_type", "unknown")
        return metrics

    @staticmethod
    def aggregate_metrics(metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate metrics across multiple episodes.
        Args:
            metrics: List of metric dictionaries from multiple episodes
        Returns:
            Aggregated metrics dictionary
        """
        if not metrics:
            return {}
        aggregated = {
            "avg_turns": sum(m.get("turns", 0) for m in metrics) / len(metrics),
        }
        # Calculate pass rate if available
        task_passed_list = [m.get("task_passed", False) for m in metrics if "task_passed" in m]
        if task_passed_list:
            aggregated["pass_rate"] = sum(task_passed_list) / len(task_passed_list)

        return aggregated


class Tau2BenchMultiDomainEnv(Tau2BenchEnv):
    """
    Multi-domain version of Tau2Bench environment.
    This variant can handle multiple domains and selects the domain
    based on information in the dataset extras field.
    """
    def __init__(self, env_config: DictConfig, extras: Dict[str, Any] = {}):
        """
        Initialize multi-domain environment.
        Expects extras to contain 'domain' field along with 'task_id'.
        """
        # Extract domain from extras if provided
        if "domain" in extras.get('extra_info', {}):
            env_config = copy.deepcopy(env_config)
            env_config.domain = extras["extra_info"]["domain"]
        super().__init__(env_config, extras)
