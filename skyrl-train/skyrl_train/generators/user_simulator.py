from __future__ import annotations

import copy
import asyncio
from typing import List, Optional

from loguru import logger
from omegaconf import DictConfig
from openai import AsyncOpenAI

from skyrl_train.inference_engines.base import ConversationType


class UserSimulator:
    """Rewrites prompts with an external model before policy inference."""

    def __init__(self, client: AsyncOpenAI, model_name: str, system_prompt: str, temperature: float) -> None:
        self._client = client
        self._model = model_name
        self._system_prompt = system_prompt
        self._temperature = temperature

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "UserSimulator":
        client = AsyncOpenAI(base_url=cfg.get("base_url"))
        return cls(
            client=client,
            model_name=cfg.model_name,
            system_prompt=cfg.system_prompt,
            temperature=cfg.temperature,
        )

    async def rewrite(self, prompt: ConversationType) -> ConversationType:
        """Rewrite the last user prompt while keeping the conversation structure intact."""

        messages: List[dict] = [{"role": "system", "content": self._system_prompt}]
        messages.extend(prompt)
        assert messages[-1]["role"] == "user", "Last message in prompt must be from user"

        try:
            completion = await self._client.chat.completions.create(
                model=self._model, messages=messages, temperature=self._temperature
            )
        except Exception:
            logger.exception("Prompt rewriting failed; falling back to the original prompt.")
            return prompt

        rewritten_content: Optional[str] = completion.choices[0].message.content
        if not rewritten_content:
            logger.warning("Prompt rewriter returned empty content; using original prompt.")
            return prompt

        rewritten_prompt = copy.deepcopy(prompt)
        rewritten_prompt[-1]["content"] = rewritten_content
        return rewritten_prompt

    def rewrite_sync(self, prompt: ConversationType) -> ConversationType:
        """Synchronously rewrite a prompt using an internal event loop.

        This is intended for call sites that cannot use ``await`` (e.g., quick
        smoke tests before launching Ray workers). If an event loop is already
        running in the current thread, callers should prefer the async
        ``rewrite`` API instead of blocking on the loop here.
        """

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.rewrite(prompt))

        raise RuntimeError("An event loop is already running; use `await rewrite(...)` instead.")