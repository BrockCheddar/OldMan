"""
End-to-end smoke test. Run manually: python e2e_smoke_test.py
Drives the real CLI against a fake OpenAI-compatible HTTP server (same wire
format as llama-server) to catch wiring bugs that unit tests skip.
"""
import json, shutil, subprocess, sys, tempfile, threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

SCRIPT = [
    # outer turn 1: propose a step
    {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
        {"id": "p1", "type": "function", "function": {"name": "propose_step", "arguments": json.dumps({
            "title": "Create hello.py",
            "acceptance_command": "python -m py_compile hello.py",
        })}}
    ]}, "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}},
    # inner turn 1: write the file
    {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
        {"id": "i1", "type": "function", "function": {"name": "write_file", "arguments": json.dumps({
            "path": "hello.py", "content": "print('hi')\n"
        })}}
    ]}, "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}},
    # inner turn 2: mark step done
    {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
        {"id": "i2", "type": "function", "function": {"name": "mark_step_done", "arguments": json.dumps({
            "summary": "created hello.py"
        })}}
    ]}, "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}},
    # outer turn 2: declare overall done
    {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
        {"id": "o1", "type": "function", "function": {"name": "declare_done", "arguments": json.dumps({
            "summary": "all done"
        })}}
    ]}, "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}},
]

_call_index = [0]

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        i = _call_index[0]
        _call_index[0] += 1
        resp = SCRIPT[min(i, len(SCRIPT) - 1)]
        payload = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
    def log_message(self, *a):
        pass

def main():
    server = HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()

    tmp = Path(tempfile.mkdtemp())
    workspace = tmp / "ws"
    config_path = tmp / "autocoder.config.json"
    config_path.write_text(json.dumps({
        "llm": {"provider": "openai_compat", "base_url": f"http://127.0.0.1:{port}/v1",
                "model": "fake", "context_window_tokens": 32768, "request_timeout_s": 30},
        "approval": {"mode": "auto"},
        "budget": {"default_command_timeout": 30},
    }))

    repo_root = Path(__file__).resolve().parent
    result = subprocess.run(
        [sys.executable, "-m", "autocoder", "start", "make a hello world script",
         "--workspace", str(workspace),
         "--config", str(config_path),
         "--acceptance", "python -m py_compile hello.py"],
        cwd=repo_root, capture_output=True, text=True, timeout=120,
    )
    print(result.stdout)
    if result.stderr.strip():
        print(result.stderr, file=sys.stderr)

    assert result.returncode == 0, f"CLI exited {result.returncode}\n{result.stdout}\n{result.stderr}"
    assert (workspace / "hello.py").exists(), "hello.py not created"
    state_file = workspace / ".autocoder" / "session.json"
    assert state_file.exists()
    state = json.loads(state_file.read_text(encoding="utf-8"))
    assert state["status"] == "done"
    assert len(state["completed_steps"]) == 1

    shutil.rmtree(tmp)
    print("\nE2E SMOKE TEST PASSED")

if __name__ == "__main__":
    main()
