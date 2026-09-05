import sys
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from autocoder.llm import (
    OpenAICompatBackend, Turn, ToolCall, ToolResult, trim_history_to_fit, LLMError,
)
from autocoder.config import LLMBackendConfig


# ---------------------------------------------------------------------------
# A tiny fake OpenAI-compatible server, standing in for llama-server, so we
# can test the real HTTP request/response cycle without needing a GPU or a
# model actually loaded.
# ---------------------------------------------------------------------------

class _FakeServer(BaseHTTPRequestHandler):
    response_body: dict = {}
    last_request_body: dict | None = None
    last_request_path: str | None = None

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        _FakeServer.last_request_body = json.loads(body)
        _FakeServer.last_request_path = self.path
        payload = json.dumps(_FakeServer.response_body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass  # keep test output quiet


@pytest.fixture
def fake_server():
    server = HTTPServer(("127.0.0.1", 0), _FakeServer)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    yield port
    server.shutdown()


def test_openai_compat_backend_round_trip_text_response(fake_server):
    port = fake_server
    _FakeServer.response_body = {
        "choices": [{"message": {"role": "assistant", "content": "hello back"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 3},
    }
    cfg = LLMBackendConfig(provider="openai_compat", base_url=f"http://127.0.0.1:{port}/v1", model="test-model")
    backend = OpenAICompatBackend(cfg)
    resp = backend.complete(system="sys prompt", history=[Turn(role="user", text="hi")], tools=[], max_tokens=100)

    assert resp.text == "hello back"
    assert resp.tool_calls == []
    assert resp.input_tokens == 12
    assert resp.output_tokens == 3
    # verify what we actually sent
    sent = _FakeServer.last_request_body
    assert sent["model"] == "test-model"
    assert sent["messages"][0]["role"] == "system"
    assert sent["messages"][-1] == {"role": "user", "content": "hi"}


def test_openai_compat_backend_round_trip_tool_call(fake_server):
    port = fake_server
    _FakeServer.response_body = {
        "choices": [{
            "message": {
                "role": "assistant", "content": None,
                "tool_calls": [{"id": "call_abc", "type": "function",
                                 "function": {"name": "read_file", "arguments": json.dumps({"path": "a.py"})}}],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {"prompt_tokens": 20, "completion_tokens": 5},
    }
    cfg = LLMBackendConfig(provider="openai_compat", base_url=f"http://127.0.0.1:{port}/v1", model="test-model")
    backend = OpenAICompatBackend(cfg)
    tools = [{"name": "read_file", "description": "reads a file", "input_schema": {"type": "object", "properties": {}}}]
    resp = backend.complete(system="sys", history=[Turn(role="user", text="read a.py")], tools=tools, max_tokens=100)

    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "read_file"
    assert resp.tool_calls[0].input == {"path": "a.py"}

    sent = _FakeServer.last_request_body
    assert sent["tools"][0]["function"]["name"] == "read_file"


def test_openai_compat_backend_serializes_full_conversation_shape(fake_server):
    port = fake_server
    _FakeServer.response_body = {
        "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        "usage": {},
    }
    cfg = LLMBackendConfig(provider="openai_compat", base_url=f"http://127.0.0.1:{port}/v1", model="m")
    backend = OpenAICompatBackend(cfg)
    history = [
        Turn(role="user", text="do a thing"),
        Turn(role="assistant", text=None, tool_calls=[ToolCall(id="c1", name="run_command", input={"command": "ls"})]),
        Turn(role="tool_results", tool_results=[ToolResult(tool_call_id="c1", content="file1\nfile2")]),
    ]
    backend.complete(system="sys", history=history, tools=[], max_tokens=10)
    sent = _FakeServer.last_request_body["messages"]
    assert sent[0]["role"] == "system"
    assert sent[1] == {"role": "user", "content": "do a thing"}
    assert sent[2]["role"] == "assistant"
    assert sent[2]["tool_calls"][0]["function"]["name"] == "run_command"
    assert sent[3] == {"role": "tool", "tool_call_id": "c1", "content": "file1\nfile2"}


def test_backend_raises_llmerror_on_connection_failure():
    cfg = LLMBackendConfig(provider="openai_compat", base_url="http://127.0.0.1:1/v1", model="m", request_timeout_s=2)
    backend = OpenAICompatBackend(cfg)
    with pytest.raises(LLMError, match="could not reach local model server"):
        backend.complete(system="sys", history=[Turn(role="user", text="hi")], tools=[], max_tokens=10)


def test_backend_raises_llmerror_on_bare_timeout_error(monkeypatch):
    """
    Regression: a read timeout on an already-open connection (server accepted
    the connection but never sent a response -- e.g. llama-server generating
    for a long time) surfaces from urllib as a bare TimeoutError, which is
    NOT a subclass of urllib.error.URLError. Catching only URLError let this
    fall through as an unhandled crash deep in http.client/socket, all the
    way up through the whole CLI, instead of a clean, retryable LLMError.
    """
    import urllib.request

    def _raise_timeout(*args, **kwargs):
        raise TimeoutError("timed out")

    monkeypatch.setattr(urllib.request, "urlopen", _raise_timeout)

    cfg = LLMBackendConfig(provider="openai_compat", base_url="http://127.0.0.1:8080/v1", model="m")
    backend = OpenAICompatBackend(cfg)
    with pytest.raises(LLMError, match="did not respond in time"):
        backend.complete(system="sys", history=[Turn(role="user", text="hi")], tools=[], max_tokens=10)


def _char_over_4(system, history, tools):
    """A simple deterministic stand-in tokenizer for testing generic trim
    mechanics -- NOT what production uses (see llm.py's
    OpenAICompatBackend.count_tokens / AnthropicBackend.count_tokens,
    which always call the real backend's real tokenizer)."""
    total = len(system or "")
    if tools:
        total += len(json.dumps(tools))
    for t in history:
        if t.text:
            total += len(t.text)
        for tc in t.tool_calls:
            total += len(tc.name) + len(json.dumps(tc.input))
        for tr in t.tool_results:
            total += len(tr.content)
    return max(1, total // 4)


def test_trim_history_leaves_small_history_untouched():
    history = [Turn(role="user", text="short")]
    result, trimmed = trim_history_to_fit(
        history, "sys", _char_over_4, context_window_tokens=32768, reserve_output_tokens=8000)
    assert not trimmed
    assert result[0].text == "short"


def test_trim_history_shrinks_old_large_tool_output_when_over_budget():
    big_output = "x" * 50000
    history = [
        Turn(role="user", text="do stuff"),
        Turn(role="tool_results", tool_results=[ToolResult(tool_call_id="c1", content=big_output)]),
    ]
    # tiny context window forces trimming
    result, trimmed = trim_history_to_fit(
        history, "sys", _char_over_4, context_window_tokens=1000, reserve_output_tokens=200)
    assert trimmed
    assert len(result[1].tool_results[0].content) < len(big_output)
    assert "trimmed to fit context window" in result[1].tool_results[0].content


def test_trim_history_responds_to_a_real_tokenizer_denser_than_char_guess():
    """Real-world regression: a request landed at 32800 tokens against a
    32768-token configured window -- a naive chars/4 guess would have said
    it was comfortably under budget, but the REAL tokenizer (code-heavy
    content tokenizes tighter than the 4-char average) disagreed. The old
    implementation re-derived its own char-based estimate internally, so
    it could never see this gap no matter what. The fix (see llm.py) is to
    trust whatever count_tokens_fn reports -- normally a real call to the
    backend's own tokenizer -- rather than re-guessing. This test proves
    that mechanism directly: a count_tokens_fn that's deliberately denser
    than chars/4 (simulating a real tokenizer on code-heavy content) must
    still force a trim, even on content a chars/4 guess would call fine."""
    context_window_tokens = 32768
    reserve_output_tokens = 8000
    budget_tokens = context_window_tokens - reserve_output_tokens  # 24768

    # sized so a chars/4 guess says "fine" but a denser chars/3 "real"
    # tokenizer says "over budget" -- exactly the gap that caused the
    # production incident
    content_len = budget_tokens * 4 - 500
    assert content_len // 4 < budget_tokens   # sanity: old heuristic would NOT have trimmed
    assert content_len // 3 > budget_tokens   # sanity: denser real tokenizer says over budget

    def denser_real_tokenizer(system, history, tools):
        return _char_over_4(system, history, tools) * 4 // 3  # ~33% denser than chars/4

    history = [
        Turn(role="user", text="do stuff"),
        Turn(role="tool_results", tool_results=[ToolResult(tool_call_id="c1", content="x" * content_len)]),
    ]
    result, trimmed = trim_history_to_fit(
        history, "sys", denser_real_tokenizer,
        context_window_tokens=context_window_tokens, reserve_output_tokens=reserve_output_tokens,
    )
    assert trimmed, "a denser real-tokenizer count should force a trim even though a chars/4 guess would not"


def test_count_tokens_hits_native_tokenize_endpoint_at_server_root(fake_server):
    """The real fix for a live overflow: count_tokens must call
    llama-server's NATIVE /tokenize endpoint, using the model's real
    tokenizer -- not guess from character count. That endpoint lives at
    the server root, not under /v1 (where /chat/completions lives)."""
    port = fake_server
    _FakeServer.response_body = {"tokens": list(range(37))}  # 37 real tokens
    cfg = LLMBackendConfig(provider="openai_compat", base_url=f"http://127.0.0.1:{port}/v1", model="test-model")
    backend = OpenAICompatBackend(cfg)

    count = backend.count_tokens(system="sys", history=[Turn(role="user", text="hi")], tools=[])

    assert _FakeServer.last_request_path == "/tokenize"  # NOT /v1/tokenize
    # 37 real tokens + per-message chat-template overhead (system + 1 user message = 2 messages)
    assert count == 37 + 2 * 12


def test_count_tokens_includes_tool_schemas(fake_server):
    """Tool schema definitions are sent with every request -- they must
    count toward the token total, not just conversation text."""
    port = fake_server
    _FakeServer.response_body = {"tokens": list(range(100))}
    cfg = LLMBackendConfig(provider="openai_compat", base_url=f"http://127.0.0.1:{port}/v1", model="m")
    backend = OpenAICompatBackend(cfg)
    tools = [{"name": "read_file", "description": "reads a file", "input_schema": {"type": "object", "properties": {}}}]

    backend.count_tokens(system="sys", history=[Turn(role="user", text="hi")], tools=tools)

    sent_content = _FakeServer.last_request_body["content"]
    assert "read_file" in sent_content


def test_count_tokens_raises_llmerror_on_unexpected_response(fake_server):
    """If something answers on the configured port but isn't actually
    llama-server's native /tokenize endpoint (e.g. a plain OpenAI-compat
    proxy that only implements /v1/chat/completions), this must fail
    loudly through the normal LLMError path -- not silently return 0 and
    let an oversized prompt through unchecked."""
    port = fake_server
    _FakeServer.response_body = {"error": "not found"}  # no "tokens" key
    cfg = LLMBackendConfig(provider="openai_compat", base_url=f"http://127.0.0.1:{port}/v1", model="m")
    backend = OpenAICompatBackend(cfg)
    with pytest.raises(LLMError, match="tokenize"):
        backend.count_tokens(system="sys", history=[Turn(role="user", text="hi")], tools=[])


def test_server_root_strips_v1_suffix():
    cfg = LLMBackendConfig(provider="openai_compat", base_url="http://127.0.0.1:8080/v1", model="m")
    backend = OpenAICompatBackend(cfg)
    assert backend._server_root() == "http://127.0.0.1:8080"
