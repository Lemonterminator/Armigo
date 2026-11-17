"""A small OpenAI-compatible chat client (local vLLM) with timeout and retries, and a scripted fake for tests."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

__all__ = ["ChatLLM", "ChatResult", "FakeLLM", "LLMError", "OpenAICompatLLM", "server_reachable"]

DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MODEL = "qwen3.8-vllm"


class LLMError(RuntimeError):
	"""The chat endpoint failed after all retries."""


@dataclass
class ChatResult:
	content: str
	latency_s: float
	prompt_tokens: int = 0
	completion_tokens: int = 0
	reasoning_tokens: int = 0
	reasoning: str | None = None
	finish_reason: str | None = None


class ChatLLM(Protocol):
	"""Anything that answers a list of chat messages."""

	def chat(self, messages: Sequence[dict[str, str]]) -> ChatResult: ...


class OpenAICompatLLM:
	"""``POST {base_url}/chat/completions`` with temperature 0 and (Qwen) thinking off by default.

	``enable_thinking`` is passed as ``chat_template_kwargs`` (vLLM); ``None`` leaves the server default."""

	def __init__(
		self,
		base_url: str = DEFAULT_BASE_URL,
		model: str = DEFAULT_MODEL,
		*,
		temperature: float = 0.0,
		max_tokens: int = 512,
		enable_thinking: bool | None = False,
		timeout_s: float = 120.0,
		retries: int = 2,
		backoff_s: float = 1.0,
		json_mode: bool = False,
	) -> None:
		import httpx

		self.base_url = base_url.rstrip("/")
		self.model = model
		self.temperature = float(temperature)
		self.max_tokens = int(max_tokens)
		self.enable_thinking = enable_thinking
		self.retries = int(retries)
		self.backoff_s = float(backoff_s)
		self.json_mode = bool(json_mode)
		self._client = httpx.Client(timeout=timeout_s)

	def _body(self, messages: Sequence[dict[str, str]]) -> dict[str, Any]:
		body: dict[str, Any] = {
			"model": self.model,
			"messages": list(messages),
			"temperature": self.temperature,
			"max_tokens": self.max_tokens,
		}
		if self.enable_thinking is not None:
			body["chat_template_kwargs"] = {"enable_thinking": bool(self.enable_thinking)}
		if self.json_mode:
			body["response_format"] = {"type": "json_object"}
		return body

	def chat(self, messages: Sequence[dict[str, str]]) -> ChatResult:
		import httpx

		body = self._body(messages)
		last: Exception | None = None
		for attempt in range(self.retries + 1):
			t0 = time.perf_counter()
			try:
				r = self._client.post(f"{self.base_url}/chat/completions", json=body)
				if r.status_code >= 500 or r.status_code == 429:
					raise LLMError(f"HTTP {r.status_code}: {r.text[:200]}")
				r.raise_for_status()
				data = r.json()
			except (httpx.HTTPError, LLMError, ValueError) as exc:
				last = exc
				if attempt < self.retries:
					time.sleep(self.backoff_s * (2**attempt))
				continue
			choice = data["choices"][0]
			msg = choice.get("message") or {}
			usage = data.get("usage") or {}
			details = usage.get("completion_tokens_details") or {}
			return ChatResult(
				content=msg.get("content") or "",
				latency_s=time.perf_counter() - t0,
				prompt_tokens=int(usage.get("prompt_tokens") or 0),
				completion_tokens=int(usage.get("completion_tokens") or 0),
				reasoning_tokens=int(details.get("reasoning_tokens") or 0),
				reasoning=msg.get("reasoning") or msg.get("reasoning_content"),
				finish_reason=choice.get("finish_reason"),
			)
		raise LLMError(f"chat failed after {self.retries + 1} attempts: {last!r}")

	def close(self) -> None:
		self._client.close()


@dataclass
class FakeLLM:
	"""Scripted answers in order (a string, or a callable of the messages); repeats the last one when exhausted."""

	responses: Sequence[str | Callable[[Sequence[dict[str, str]]], str]]
	calls: list[list[dict[str, str]]] = field(default_factory=list)

	def chat(self, messages: Sequence[dict[str, str]]) -> ChatResult:
		self.calls.append([dict(m) for m in messages])
		if not self.responses:
			raise LLMError("FakeLLM has no responses")
		r = self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]
		text = r(messages) if callable(r) else r
		return ChatResult(content=text, latency_s=0.0, prompt_tokens=sum(len(m["content"]) for m in messages) // 4)


def server_reachable(base_url: str = DEFAULT_BASE_URL, timeout_s: float = 2.0) -> bool:
	"""True if ``GET {base_url}/models`` answers."""
	import httpx

	try:
		return httpx.get(f"{base_url.rstrip('/')}/models", timeout=timeout_s).status_code == 200
	except httpx.HTTPError:
		return False
