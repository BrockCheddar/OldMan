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
from typing import Any, Literal

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


class LLMClient:
    def complete(self, system: str, history: list[Turn], tools: list[dict], max_tokens: int) -> LLMResponse:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Rough, tokenizer-free context sizing (used only to decide when to trim old
# tool output before hitting the backend -- not for billing/accounting).
# ~4 chars/token is a reasonable average across code+English; err generous
# (undercount tokens) so we trim earlier rather than overflow the server.
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def history_char_len(system: str, history: list[Turn], tools: list[dict] | None = None) -> int:
    total = len(system)
    if tools:
        # Tool schema definitions (name + description + input_schema) are
        # sent with EVERY request -- this must be counted, not just the
        # conversation text.
        total += len(json.dumps(tools))
    for t in history:
        if t.text:
            total += len(t.text)
        for tc in t.tool_calls:
            total += len(tc.name) + len(json.dumps(tc.input))
        for tr in t.tool_results:
            total += len(tr.content)
    return total


def trim_history_to_fit(history: list[Turn], system: str, context_window_tokens: int,
                         reserve_output_tokens: int, tools: list[dict] | None = None) -> tuple[list[Turn], bool]:
    """
    If the (rough) estimated prompt size would exceed the model's real
    context window, replace the CONTENTS of the oldest tool_results with a
    short placeholder (keeping the turn structure and tool_call/result
    pairing intact, which most backends require) until it fits, or until
    nothing more can be trimmed.

    Nothing about the actual code is lost -- every accepted change is a git
    commit regardless of what's still in the model's short-term context.
    Returns (possibly-modified history, whether anything was trimmed).
    """
    budget_tokens = context_window_tokens - reserve_output_tokens
    if budget_tokens <= 0:
        budget_tokens = 1
    budget_chars = budget_tokens * 4  # inverse of the ~4 chars/token estimate

    if history_char_len(system, history, tools) <= budget_chars:
        return history, False

    trimmed = False
    # Walk oldest-to-newest, shrinking large tool_results first since those
    # are almost always the bulkiest (command output, file dumps) and the
    # least likely to matter once several turns have passed.
    for turn in history:
        if history_char_len(system, history, tools) <= budget_chars:
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
        url = self.cfg.base_url.rstrip("/") + path
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
                f"local model server at {self.cfg.base_url} returned HTTP {e.code}: {detail}\n"
                f"(check that llama-server is actually running and --port matches base_url)"
            ) from e
        except urllib.error.URLError as e:
            raise LLMError(
                f"could not reach local model server at {self.cfg.base_url}: {e.reason}\n"
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
                f"local model server at {self.cfg.base_url} did not respond in time or the "
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
