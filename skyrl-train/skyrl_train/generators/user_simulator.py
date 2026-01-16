from __future__ import annotations

import copy
import asyncio
import os
import runpy
from pathlib import Path
from typing import List, Optional, Dict

from loguru import logger
from omegaconf import DictConfig
from openai import AsyncOpenAI

from skyrl_train.inference_engines.base import ConversationType
from .user_simulator_custom_utils import extract_outer_dict
from .user_simulator_prompt import USER_SIM_SYSPROMPT, USER_SIM_SYSPROMPT_ANGRY


def usersim_stringfy(
    conversation: ConversationType,
    formatting_cfg: Dict
) -> Optional[str]:
    """
    Convert conversation messages to a formatted chat history string to be used in the system prompt.
    """
    user_tpl = formatting_cfg.get("user_template")
    ai_tpl = formatting_cfg.get("ai_template")
    chat_history_str = ""
    for message in conversation:
        if message["role"] not in ["user", "assistant"]:
            continue
        tpl = user_tpl if message["role"] == "user" else ai_tpl
        chat_history_str += tpl + message["content"] + "\n\n"
    return chat_history_str


class UserSimulator:
    
    def __init__(self, client: AsyncOpenAI, model_name: str, system_prompt: str, temperature: float) -> None:
        self._client = client
        self._model = model_name
        self._system_prompt = system_prompt
        self._temperature = temperature
        self._max_retries = 8

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "UserSimulator":
        # client setup
        if cfg.get("is_local"):
            base_url = cfg.get("base_url").format(port=cfg.get("local_port"))
            client = AsyncOpenAI(base_url=base_url)
        else:
            raise NotImplementedError("Only local vllm model is supported for UserSimulator.")
        # note that system_prompt has {} formatting with task_desc, single_turn_prompt, chat_history, terminal_signal
        if cfg.get("tone") == "default":
            _sysprompt = USER_SIM_SYSPROMPT
        elif cfg.get("tone") == "angry":
            _sysprompt = USER_SIM_SYSPROMPT_ANGRY
        else:
            raise ValueError(f"Unsupported tone {cfg.get('tone')} for UserSimulator."
                             f" Current config: {cfg}")
        return cls(
            client=client,
            model_name=cfg.model_name,
            system_prompt=_sysprompt,
            temperature=cfg.temperature,
        )

    async def rewrite(self,
                      chat_history: ConversationType,
                      task_desc: str,
                      single_turn_prompt: str,
                      formatting_cfg: Dict,
                      debug: bool = False,
                      **kwargs) -> Optional[str]:

        messages = [
            {"role": "user", "content": self._system_prompt.format(
                task_desc=task_desc,
                single_turn_prompt=single_turn_prompt,
                chat_history=usersim_stringfy(
                    conversation=chat_history,
                    formatting_cfg=formatting_cfg,
                ),
                terminal_signal=formatting_cfg.get("terminal_signal"),
            )}
        ]

        if debug: # print messages sent to user simulator
            logger.info(f"messages to user simulator: {messages}")
            if 'model_name' in kwargs:
                from transformers import AutoProcessor
                processor = AutoProcessor.from_pretrained(kwargs['model_name'])
                logger.info("apply_chat_template result: "
                            f"{processor.apply_chat_template(
                                messages, tokenize=False,
                                add_generation_prompt=True)}")

        for attempt in range(1, self._max_retries + 1):
            try:
                rewritten_content: Optional[str] = None
                completion = await self._client.chat.completions.create(
                    model=self._model, messages=messages, temperature=self._temperature
                )
                rewritten_content = completion.choices[0].message.content
                if debug:
                    logger.info(f"rewritten prompt from user simulator: {rewritten_content}")
                parsed_dict = extract_outer_dict(rewritten_content)
                assert isinstance(parsed_dict.get("response"), str), "extracted response not a string"
                return parsed_dict["response"]
            except Exception:
                if rewritten_content is None:
                    logger.exception(f"Attempt {attempt}/{self._max_retries}: rewriting failed.")
                else:
                    logger.exception(
                        f"Attempt {attempt}/{self._max_retries}: extracting from rewriting failed "
                        f"with content: {rewritten_content}"
                    )
                if attempt < self._max_retries:
                    await asyncio.sleep(2 ** (attempt - 1))
                else:
                    logger.error("All prompt rewriting attempts failed.")
                    return None
        return None

    def rewrite_sync(self, **kwargs) -> ConversationType:
        """
        Synchronously rewrite a prompt using an internal event loop.

        This is intended for call sites that cannot use ``await`` (e.g., quick
        smoke tests before launching Ray workers). If an event loop is already
        running in the current thread, callers should prefer the async
        ``rewrite`` API instead of blocking on the loop here.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.rewrite(**kwargs))

        raise RuntimeError("An event loop is already running; use `await rewrite(...)` instead.")