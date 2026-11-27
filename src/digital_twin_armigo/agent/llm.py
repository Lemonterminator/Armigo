"""A small OpenAI-compatible chat client (local vLLM) with timeout and retries, and a scripted fake for tests."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

__all__ = [
	"ChatLLM",
	"ChatResult",
	"FakeLLM",
	"LLMError",
	"OpenAICompatLLM",
	"schema_response_format",
	"server_reachable",
]

DEFAULT_BASE_URL = "http://127.0.0.1:8038/v1"  # port of scripts/serve_armigo_qwen.sh
DEFAULT_MODEL = "qwen3.8-vllm"


class LLMError(RuntimeError):
	"""The chat endpoint failed after all retries. ``kind`` is ``timeout`` or ``http_error``."""

	def __init__(self, message: str = "", kind: str = "http_error", *, retryable: bool = True) -> None:
		super().__init__(message)
		self.kind = kind
		self.retryable = retryable


@dataclass
class ChatResult:
	content: str
	latency_s: float
	prompt_tokens: int = 0
	completion_tokens: int = 0
	reasoning_tokens: int = 0
	reasoning: str | None = None
	finish_reason: str | None = None
	response_format: str | None = None  # "json_schema", "json_object" or None, as actually sent
	cached_tokens: int | None = None


class ChatLLM(Protocol):
	"""Anything that answers a list of chat messages."""

	def chat(self, messages: Sequence[dict[str, str]]) -> ChatResult: ...


def schema_response_format(schema: dict[str, Any], name: str = "skill_choice") -> dict[str, Any]:
	"""The OpenAI ``response_format`` body for a strict JSON schema (vLLM 0.33 constrains decoding with it)."""
	return {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}}


class OpenAICompatLLM:
	"""``POST {base_url}/chat/completions`` with temperature 0 and (Qwen) thinking off by default.

	``enable_thinking`` is passed as ``chat_template_kwargs`` (vLLM); ``None`` leaves the server default.
	``chat(..., json_schema=...)`` sends a strict JSON-schema ``response_format``; if the server rejects it with a
	400/422 the client degrades to ``json_object`` for the rest of its life and sets :attr:`schema_degraded`.
	``max_model_len`` seeds the context limit; the server's ``/models`` entry overrides it when reachable."""

	structured = True

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
		max_model_len: int = 4096,
		tokenize_timeout_s: float = 2.0,
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
		self.max_model_len = int(max_model_len)
		self.tokenize_timeout_s = float(tokenize_timeout_s)
		self.schema_degraded = False
		self._tokenize_off_until = 0.0
		self._limit_checked = False
		self._client = httpx.Client(timeout=timeout_s)

	def _prompt_body(self, messages: Sequence[dict[str, str]]) -> dict[str, Any]:
		body: dict[str, Any] = {
			"model": self.model,
			"messages": list(messages),
		}
		if self.enable_thinking is not None:
			body["chat_template_kwargs"] = {"enable_thinking": bool(self.enable_thinking)}
		return body

	def _body(self, messages: Sequence[dict[str, str]], json_schema: dict[str, Any] | None = None) -> dict[str, Any]:
		body = self._prompt_body(messages)
		body.update(temperature=self.temperature, max_tokens=self.max_tokens)
		if json_schema is not None and not self.schema_degraded:
			body["response_format"] = schema_response_format(json_schema)
		elif self.json_mode or json_schema is not None:
			body["response_format"] = {"type": "json_object"}
		return body

	def _post(self, messages, json_schema, body):
		"""One POST; a 400/422 on a schema body degrades to ``json_object`` and resends at once."""
		r = self._client.post(f"{self.base_url}/chat/completions", json=body)
		if r.status_code in (400, 422) and body.get("response_format", {}).get("type") == "json_schema":
			self.schema_degraded = True
			body = self._body(messages, json_schema)
			r = self._client.post(f"{self.base_url}/chat/completions", json=body)
		if r.status_code >= 500 or r.status_code == 429:
			raise LLMError(f"HTTP {r.status_code}: {r.text[:200]}")
		if r.status_code >= 400:
			raise LLMError(f"HTTP {r.status_code}: {r.text[:200]}", retryable=False)
		return r.json(), body

	def chat(self, messages: Sequence[dict[str, str]], *, json_schema: dict[str, Any] | None = None) -> ChatResult:
		import httpx

		body = self._body(messages, json_schema)
		last: Exception | None = None
		kind = "http_error"
		for attempt in range(self.retries + 1):
			t0 = time.perf_counter()
			try:
				data, body = self._post(messages, json_schema, body)
			except httpx.TimeoutException as exc:
				last, kind = exc, "timeout"
			except LLMError as exc:
				last, kind = exc, exc.kind
				if not exc.retryable:
					break
			except (httpx.HTTPError, ValueError) as exc:
				last, kind = exc, "http_error"
			else:
				choice = data["choices"][0]
				msg = choice.get("message") or {}
				usage = data.get("usage") or {}
				details = usage.get("completion_tokens_details") or {}
				prompt_details = usage.get("prompt_tokens_details")
				cached_tokens = prompt_details.get("cached_tokens") if isinstance(prompt_details, dict) else None
				return ChatResult(
					content=msg.get("content") or "",
					latency_s=time.perf_counter() - t0,
					prompt_tokens=int(usage.get("prompt_tokens") or 0),
					completion_tokens=int(usage.get("completion_tokens") or 0),
					reasoning_tokens=int(details.get("reasoning_tokens") or 0),
					reasoning=msg.get("reasoning") or msg.get("reasoning_content"),
					finish_reason=choice.get("finish_reason"),
					response_format=(body.get("response_format") or {}).get("type"),
					cached_tokens=cached_tokens if type(cached_tokens) is int and cached_tokens >= 0 else None,
				)
			if attempt < self.retries:
				time.sleep(self.backoff_s * (2**attempt))
		raise LLMError(f"chat failed after {attempt + 1} attempts: {last!r}", kind)

	def context_limit(self) -> int:
		"""The configured model's advertised limit, or the constructor fallback."""
		import httpx

		if not self._limit_checked:
			self._limit_checked = True
			try:
				r = self._client.get(f"{self.base_url}/models", timeout=self.tokenize_timeout_s)
				r.raise_for_status()
				for model in r.json()["data"]:
					if isinstance(model, dict) and model.get("id") == self.model:
						limit = model.get("max_model_len")
						if type(limit) is int and limit > 0:
							self.max_model_len = limit
						break
			except (httpx.HTTPError, ValueError, LookupError, TypeError):
				pass
		return self.max_model_len

	def count_tokens(self, messages: Sequence[dict[str, str]]) -> int | None:
		"""Prompt tokens through vLLM's ``/tokenize`` (short timeout); ``None`` when unreachable.

		After a failure the endpoint is skipped for 30 s so a missing tokenizer costs one timeout, not one per call."""
		import httpx

		if time.monotonic() < self._tokenize_off_until:
			return None
		root = self.base_url.removesuffix("/v1")
		try:
			r = self._client.post(
				f"{root}/tokenize",
				json=self._prompt_body(messages),
				timeout=self.tokenize_timeout_s,
			)
			r.raise_for_status()
			return int(r.json()["count"])
		except (httpx.HTTPError, ValueError, LookupError, TypeError):
			self._tokenize_off_until = time.monotonic() + 30.0
			return None

	def close(self) -> None:
		self._client.close()


@dataclass
class FakeLLM:
	"""Scripted answers in order (a string, or a callable of the messages); repeats the last one when exhausted.

	A response may also be an exception instance, which is raised. ``schemas`` records each call's ``json_schema``."""

	responses: Sequence[str | Callable[[Sequence[dict[str, str]]], str] | Exception]
	calls: list[list[dict[str, str]]] = field(default_factory=list)
	schemas: list[dict[str, Any] | None] = field(default_factory=list)
	max_tokens: int = 96
	max_model_len: int = 4096
	structured = True

	def chat(self, messages: Sequence[dict[str, str]], *, json_schema: dict[str, Any] | None = None) -> ChatResult:
		self.calls.append([dict(m) for m in messages])
		self.schemas.append(json_schema)
		if not self.responses:
			raise LLMError("FakeLLM has no responses")
		r = self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]
		if isinstance(r, Exception):
			raise r
		text = r(messages) if callable(r) else r
		return ChatResult(content=text, latency_s=0.0, prompt_tokens=sum(len(m["content"]) for m in messages) // 4)

	def context_limit(self) -> int:
		return self.max_model_len


def server_reachable(base_url: str = DEFAULT_BASE_URL, timeout_s: float = 2.0) -> bool:
	"""True if ``GET {base_url}/models`` answers."""
	import httpx

	try:
		return httpx.get(f"{base_url.rstrip('/')}/models", timeout=timeout_s).status_code == 200
	except httpx.HTTPError:
		return False
