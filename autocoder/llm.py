"""
Model-agnostic adapter layer (Phase 2). The agent loop only ever talks to
this module's `LLMClient` interface and never knows or cares whether the
actual backend is Anthropic's API or your local llama-server.

Two backends are implemented:
  - OpenAICompatBackend: talks to any OpenAI-compatible /v1/chat/completions
    server -- this is what llama-server, LM Studio, vLLM, and Ollama's /v1
    endpoint all expose. This is the DEFAULT, pointed at your local
    llama-server running Ornith9B.
  - AnthropicBackend: talks to the real Anthropic API. Only imports the
    `anthropic` package if you actually select this provider, so it's not a
    hard dependency for the local-only path.

Both backends convert to/from a small backend-agnostic history format
(Turn / ToolCall / ToolResult) so the executor loop's logic doesn't change
when you flip `provider` in the config file.
"""
from __future__ import annotations

import json
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from .config import LLMBackendConfig


# ---------------------------------------------------------------------------
# Backend-agnostic conversation representation
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class ToolResult:
    tool_call_id: str
    content: str
    is_error: bool = False


@dataclass
class Turn:
    role: Literal["user", "assistant", "tool_results"]
    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)


@dataclass
class LLMResponse:
    text: str | None
    tool_calls: list[ToolCall]
    stop_reason: str
    input_tokens: int
    output_tokens: int


class LLMError(RuntimeError):
    pass


# A flat "tokenize the concatenated text" call (see OpenAICompatBackend.
# count_tokens) gets the REAL token count for actual message content, but
# can't see the role/special-token framing the chat template inserts
# BETWEEN messages (e.g. <|im_start|>user ... <|im_end|>). That's a small,
# bounded, well-understood category of overhead -- not a guess at how
# densely code tokenizes -- so it's a fixed per-message constant rather
# than a blanket safety margin. If a context overflow still happens after
# this, the fix is to raise this number, not to add another vague buffer.
CHAT_TEMPLATE_OVERHEAD_TOKENS_PER_MESSAGE = 12


class LLMClient:
    def complete(self, system: str, history: list[Turn], tools: list[dict], max_tokens: int) -> LLMResponse:
        raise NotImplementedError

    def count_tokens(self, system: str, history: list[Turn], tools: list[dict]) -> int:
        """The REAL prompt token count for this exact request, from this
        backend's own tokenizer -- not a character-based guess. Every
        backend must implement this for real; there is no default/fallback
        implementation here on purpose, so a backend that can't report a
        real count fails loudly instead of silently under-counting."""
        raise NotImplementedError


def trim_history_to_fit(history: list[Turn], system: str,
                         count_tokens_fn: "Callable[[str, list[Turn], list[dict] | None], int]",
                         context_window_tokens: int, reserve_output_tokens: int,
                         tools: list[dict] | None = None) -> tuple[list[Turn], bool]:
    """
    If the REAL prompt token count -- from count_tokens_fn, which calls the
    actual backend's own tokenizer (see LLMClient.count_tokens), not a
    character-based guess -- would exceed the model's real context window,
    replace the CONTENTS of the oldest tool_results with a short placeholder
    (keeping the turn structure and tool_call/result pairing intact, which
    most backends require) until it fits, or until nothing more can be
    trimmed.

    There is deliberately no character-based fallback here: guessing
    chars-per-token is what caused a real overflow in production (a request
    landed a few dozen tokens over a configured context window because the
    estimate didn't match how the real tokenizer actually counted
    code-heavy content). If count_tokens_fn can't produce a real count, it
    should raise -- see LLMClient.count_tokens -- and that failure surfaces
    through the normal LLMError retry/abort path instead of silently
    under-trimming.

    Nothing about the actual code is lost -- every accepted change is a git
    commit regardless of what's still in the model's short-term context.
    Returns (possibly-modified history, whether anything was trimmed).
    """
    budget_tokens = max(context_window_tokens - reserve_output_tokens, 1)

    if count_tokens_fn(system, history, tools) <= budget_tokens:
        return history, False

    trimmed = False
    # Walk oldest-to-newest, shrinking large tool_results first since those
    # are almost always the bulkiest (command output, file dumps) and the
    # least likely to matter once several turns have passed.
    for turn in history:
        if count_tokens_fn(system, history, tools) <= budget_tokens:
            break
        for tr in turn.tool_results:
            if len(tr.content) > 400:
                tr.content = tr.content[:200] + "\n...[older output trimmed to fit context window]...\n" + tr.content[-200:]
                trimmed = True
    return history, trimmed


# ---------------------------------------------------------------------------
# OpenAI-compatible backend (default: local llama-server running Ornith9B)
# ---------------------------------------------------------------------------

class OpenAICompatBackend(LLMClient):
    def __init__(self, cfg: LLMBackendConfig):
        self.cfg = cfg

    def complete(self, system: str, history: list[Turn], tools: list[dict], max_tokens: int) -> LLMResponse:
        messages = self._to_openai_messages(system, history)
        oa_tools = self._to_openai_tools(tools)
        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "tools": oa_tools,
            "tool_choice": "auto",
            "max_tokens": max_tokens,
            "stream": False,
        }
        body = self._post("/chat/completions", payload)

        choice = body["choices"][0]
        message = choice["message"]
        text = message.get("content")
        tool_calls: list[ToolCall] = []
        for tc in message.get("tool_calls") or []:
            fn = tc["function"]
            try:
                args = json.loads(fn["arguments"]) if fn.get("arguments") else {}
            except json.JSONDecodeError as e:
                # Surface as a normal empty-input call with the raw string kept in
                # 'raw' so the executor loop can turn this into a tool_result error
                # and let the model repair its own call -- no regex guessing here.
                args = {"_unparseable_arguments": fn.get("arguments", ""), "_error": str(e)}
            tool_calls.append(ToolCall(id=tc["id"], name=fn["name"], input=args))

        usage = body.get("usage", {}) or {}
        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            stop_reason=choice.get("finish_reason", "stop"),
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
        )

    def _to_openai_messages(self, system: str, history: list[Turn]) -> list[dict]:
        messages: list[dict] = [{"role": "system", "content": system}]
        for turn in history:
            if turn.role == "user":
                messages.append({"role": "user", "content": turn.text or ""})
            elif turn.role == "assistant":
                msg: dict[str, Any] = {"role": "assistant", "content": turn.text}
                if turn.tool_calls:
                    msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": json.dumps(tc.input)},
                        }
                        for tc in turn.tool_calls
                    ]
                messages.append(msg)
            elif turn.role == "tool_results":
                for tr in turn.tool_results:
                    messages.append({"role": "tool", "tool_call_id": tr.tool_call_id, "content": tr.content})
        return messages

    @staticmethod
    def _to_openai_tools(tools: list[dict]) -> list[dict]:
        return [
            {"type": "function", "function": {"name": t["name"], "description": t["description"],
                                               "parameters": t["input_schema"]}}
            for t in tools
        ]

    def _post(self, path: str, payload: dict) -> dict:
        return self._post_url(self.cfg.base_url.rstrip("/") + path, payload)

    def _server_root(self) -> str:
        """base_url is the OpenAI-compat mount point (e.g. .../v1).
        llama-server's native endpoints (/tokenize, /health, /props) live at
        the server root, one level up from that."""
        base = self.cfg.base_url.rstrip("/")
        if base.endswith("/v1"):
            base = base[: -len("/v1")]
        return base

    def count_tokens(self, system: str, history: list[Turn], tools: list[dict]) -> int:
        """Real token count from llama-server's own tokenizer via its
        native /tokenize endpoint -- not a character-based guess. Builds
        the same messages/tools structure the real request would send,
        concatenates their text content, and tokenizes that for real.

        The one gap a flat concatenated-text tokenize call can't see is the
        role/special-token framing the chat template inserts BETWEEN
        messages (e.g. <|im_start|>user ... <|im_end|>) -- that's a small,
        bounded, well-understood category of overhead, not a guess at
        content density, so it's covered by a fixed per-message constant
        rather than a blanket safety margin."""
        messages = self._to_openai_messages(system, history)
        oa_tools = self._to_openai_tools(tools)
        parts: list[str] = []
        for m in messages:
            parts.append(str(m.get("role", "")))
            if m.get("content"):
                parts.append(str(m["content"]))
            for tc in (m.get("tool_calls") or []):
                parts.append(tc["function"]["name"])
                parts.append(tc["function"]["arguments"])
        if oa_tools:
            parts.append(json.dumps(oa_tools))
        text = "\n".join(parts)

        url = self._server_root() + "/tokenize"
        body = self._post_url(url, {"content": text})
        tok = body.get("tokens")
        if not isinstance(tok, list):
            raise LLMError(
                f"local model server's native /tokenize endpoint at {url} returned an "
                f"unexpected response ({body!r}). This requires llama-server itself (or "
                f"another backend exposing the same native /tokenize endpoint), not just "
                f"any OpenAI-compat server -- some proxies/gateways only implement the "
                f"/v1/chat/completions route."
            )
        return len(tok) + len(messages) * CHAT_TEMPLATE_OVERHEAD_TOKENS_PER_MESSAGE

    def _post_url(self, url: str, payload: dict) -> dict:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.cfg.api_key}",
            },
        )
        try:
            # request_timeout_s is intentionally large by default (see config.py) --
            # local generation on modest hardware is not instant, and this timeout
            # exists to catch a genuinely hung server, not to rush the model.
            with urllib.request.urlopen(req, timeout=self.cfg.request_timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise LLMError(
                f"local model server at {url} returned HTTP {e.code}: {detail}\n"
                f"(check that llama-server is actually running and --port matches base_url)"
            ) from e
        except urllib.error.URLError as e:
            raise LLMError(
                f"could not reach local model server at {url}: {e.reason}\n"
                f"(is llama-server running? e.g. `llama-server -m <ornith9b.gguf> --port 8080`)"
            ) from e
        except OSError as e:
            # Broader than URLError on purpose: a read timeout on an already-
            # open connection (llama-server accepted the connection but never
            # sent a response) surfaces as a bare TimeoutError/socket.timeout,
            # which is NOT a subclass of URLError and was previously falling
            # through both except clauses entirely -- an unhandled crash deep
            # in urllib/http.client instead of a clean, retryable LLMError.
            reason = getattr(e, "reason", e)
            raise LLMError(
                f"local model server at {url} did not respond in time or the "
                f"connection failed ({type(e).__name__}: {reason}). If this happens on "
                f"long-running tasks, consider raising request_timeout_s in your config; "
                f"if it happens immediately, check that llama-server is actually running."
            ) from e


# ---------------------------------------------------------------------------
# Anthropic backend (optional -- only imported if selected)
# ---------------------------------------------------------------------------

class AnthropicBackend(LLMClient):
    def __init__(self, cfg: LLMBackendConfig):
        import os
        try:
            import anthropic
        except ImportError as e:
            raise LLMError(
                "provider is 'anthropic' but the `anthropic` package isn't installed. "
                "Run: pip install anthropic"
            ) from e
        api_key = os.environ.get(cfg.api_key_env)
        if not api_key:
            raise LLMError(f"provider is 'anthropic' but env var {cfg.api_key_env} is not set")
        self.cfg = cfg
        self._client = anthropic.Anthropic(api_key=api_key)

    def complete(self, system: str, history: list[Turn], tools: list[dict], max_tokens: int) -> LLMResponse:
        messages = self._to_anthropic_messages(history)
        response = self._client.messages.create(
            model=self.cfg.model,
            system=system,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
        )
        text_parts = [b.text for b in response.content if b.type == "text"]
        tool_calls = [ToolCall(id=b.id, name=b.name, input=b.input)
                      for b in response.content if b.type == "tool_use"]
        return LLMResponse(
            text="\n".join(text_parts) if text_parts else None,
            tool_calls=tool_calls,
            stop_reason=response.stop_reason,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    def count_tokens(self, system: str, history: list[Turn], tools: list[dict]) -> int:
        """Anthropic exposes a dedicated, exact token-counting endpoint --
        the same count the real API call would use, no estimation
        involved at all."""
        messages = self._to_anthropic_messages(history)
        try:
            result = self._client.messages.count_tokens(
                model=self.cfg.model, system=system, messages=messages, tools=tools,
            )
        except Exception as e:
            raise LLMError(f"Anthropic count_tokens call failed: {e}") from e
        return result.input_tokens

    @staticmethod
    def _to_anthropic_messages(history: list[Turn]) -> list[dict]:
        messages: list[dict] = []
        for turn in history:
            if turn.role == "user":
                messages.append({"role": "user", "content": turn.text or ""})
            elif turn.role == "assistant":
                content: list[dict] = []
                if turn.text:
                    content.append({"type": "text", "text": turn.text})
                for tc in turn.tool_calls:
                    content.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.input})
                messages.append({"role": "assistant", "content": content})
            elif turn.role == "tool_results":
                content = [
                    {"type": "tool_result", "tool_use_id": tr.tool_call_id,
                     "content": tr.content, "is_error": tr.is_error}
                    for tr in turn.tool_results
                ]
                messages.append({"role": "user", "content": content})
        return messages


def build_llm_client(cfg: LLMBackendConfig) -> LLMClient:
    if cfg.provider == "openai_compat":
        return OpenAICompatBackend(cfg)
    if cfg.provider == "anthropic":
        return AnthropicBackend(cfg)
    raise LLMError(f"unknown provider '{cfg.provider}' (expected 'openai_compat' or 'anthropic')")
