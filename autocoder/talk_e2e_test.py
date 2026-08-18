"""Manual e2e test for talk.py -- feeds real stdin answers through the real
script against a fake local server, exactly like a user typing at the prompt."""
import json, shutil, subprocess, sys, tempfile, threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

SCRIPT = [
    {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
        {"id": "p1", "type": "function", "function": {"name": "propose_step", "arguments": json.dumps({
            "title": "Create hello.py", "acceptance_command": "python -m py_compile hello.py"})}}
    ]}, "finish_reason": "tool_calls"}], "usage": {}},
    {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
        {"id": "i1", "type": "function", "function": {"name": "write_file", "arguments": json.dumps({
            "path": "hello.py", "content": "print('hi')\n"})}}
    ]}, "finish_reason": "tool_calls"}], "usage": {}},
    {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
        {"id": "i2", "type": "function", "function": {"name": "mark_step_done", "arguments": json.dumps({
            "summary": "created hello.py"})}}
    ]}, "finish_reason": "tool_calls"}], "usage": {}},
    {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
        {"id": "o1", "type": "function", "function": {"name": "declare_done", "arguments": json.dumps({
            "summary": "done"})}}
    ]}, "finish_reason": "tool_calls"}], "usage": {}},
]
_i = [0]

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        resp = SCRIPT[min(_i[0], len(SCRIPT) - 1)]
        _i[0] += 1
        payload = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
    def log_message(self, *a): pass

def main():
    server = HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()

    tmp = Path(tempfile.mkdtemp())
    (tmp / "autocoder.config.json").write_text(json.dumps({
        "llm": {"provider": "openai_compat", "base_url": f"http://127.0.0.1:{port}/v1",
                "model": "fake", "context_window_tokens": 32768, "request_timeout_s": 30},
        "approval": {"mode": "auto"},
        "budget": {"default_command_timeout": 30},
    }))

    stdin_answers = "\n".join([
        "ws",                                  # workspace folder
        "make a hello world script",           # goal
        "",                                     # source repo -> blank
        "python -m py_compile hello.py",        # acceptance command
        "n",                                    # anything else? -> no
    ]) + "\n"

    repo_root = Path(__file__).resolve().parent
    result = subprocess.run(
        [sys.executable, str(repo_root / "talk.py")],
        cwd=tmp, input=stdin_answers, capture_output=True, text=True, timeout=60,
    )
    print(result.stdout)
    if result.stderr.strip():
        print(result.stderr, file=sys.stderr)

    assert result.returncode == 0, f"exited {result.returncode}"
    assert (tmp / "ws" / "hello.py").exists(), "hello.py not created"
    state = json.loads((tmp / "ws" / ".autocoder" / "session.json").read_text(encoding="utf-8"))
    assert state["status"] == "done"

    shutil.rmtree(tmp)
    print("\nTALK.PY E2E TEST PASSED")

if __name__ == "__main__":
    main()
