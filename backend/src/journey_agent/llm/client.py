"""Chat model behind a minimal protocol.

We speak the OpenAI-compatible API, which Ollama, `mlx_lm.server`, LM Studio
and llama.cpp's server all expose — swapping runtimes is a config change.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass
from typing import Protocol

from openai import AsyncOpenAI

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


@dataclass(frozen=True)
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass(frozen=True)
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    model: str
    latency_ms: float
    tokens_estimated: bool = False


class ChatModel(Protocol):
    model: str

    async def complete(self, messages: list[Message]) -> Completion: ...

    async def health(self) -> bool: ...


def estimate_tokens(text: str) -> int:
    """~4 chars/token heuristic, used only when the server omits `usage`."""
    return max(1, len(text) // 4)


class OpenAICompatibleChat:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout_s: float,
        concurrency: int = 1,
    ) -> None:
        self.model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        # One retry: a local server is either up or not; long retry chains just add latency.
        self._client = AsyncOpenAI(
            base_url=base_url, api_key=api_key, timeout=timeout_s, max_retries=1
        )
        # Queue requests instead of letting parallel generations thrash unified memory.
        self._slots = asyncio.Semaphore(concurrency)

    async def complete(self, messages: list[Message]) -> Completion:
        async with self._slots:
            started = time.perf_counter()
            resp = await self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": m.role, "content": m.content} for m in messages],  # type: ignore[misc]
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
        latency = (time.perf_counter() - started) * 1000
        # Reasoning models (Qwen3, DeepSeek-R1) may emit <think> blocks; never show them.
        text = _THINK.sub("", resp.choices[0].message.content or "").strip()
        usage = resp.usage
        if usage is not None:
            return Completion(
                text, usage.prompt_tokens, usage.completion_tokens, resp.model, latency
            )
        prompt = "".join(m.content for m in messages)
        return Completion(
            text,
            estimate_tokens(prompt),
            estimate_tokens(text),
            resp.model,
            latency,
            tokens_estimated=True,
        )

    async def health(self) -> bool:
        try:
            await self._client.models.list()
        except Exception:
            return False
        return True
