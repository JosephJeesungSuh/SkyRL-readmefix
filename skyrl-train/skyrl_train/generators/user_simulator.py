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
from .user_simulator_prompt import USER_SIM_SYSPROMPT


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
        return cls(
            client=client,
            model_name=cfg.model_name,
            system_prompt=USER_SIM_SYSPROMPT,
            temperature=cfg.temperature,
        )
    
    def _stringfy(self,
                  conversation: ConversationType,
                  formatting_cfg: Dict) -> Optional[str]:
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

    async def rewrite(self,
                      chat_history: ConversationType,
                      task_desc: str,
                      single_turn_prompt: str,
                      formatting_cfg: Dict,
                      debug: bool = False,
                      **kwargs) -> ConversationType:

        messages = [
            {"role": "user", "content": self._system_prompt.format(
                task_desc=task_desc,
                single_turn_prompt=single_turn_prompt,
                chat_history=self._stringfy(
                    conversation=chat_history,
                    formatting_cfg=formatting_cfg,
                ),
                terminal_signal=formatting_cfg.get("terminal_signal"),
            )}
        ]

        if debug:
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
                completion = await self._client.chat.completions.create(
                    model=self._model, messages=messages, temperature=self._temperature
                )
                break
            except Exception:
                logger.exception("Prompt rewriting failed on attempt %d/%d.", attempt, self._max_retries)
                if attempt < self._max_retries:
                    await asyncio.sleep(2 ** (attempt - 1))
                else:
                    logger.error("All prompt rewriting attempts failed.")
                    return None

        rewritten_content: Optional[str] = completion.choices[0].message.content
        if not rewritten_content:
            logger.warning("Prompt rewriter returned empty content; using original prompt.")
        return rewritten_content

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