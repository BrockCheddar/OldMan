import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autocoder import cli


class _FakeLLMCfg:
    provider = "fake"
    model = "fake-model"
    base_url = "http://localhost"


class _FakeConfig:
    llm = _FakeLLMCfg()


class _FakeAgent:
    calls = []

    def __init__(self, config):
        _FakeAgent.calls.append(("__init__", config))

    def reopen(self, instructions):
        _FakeAgent.calls.append(("reopen", instructions))

    def run(self, goal, resume, final_acceptance_command=None):
        _FakeAgent.calls.append(("run", goal, resume, final_acceptance_command))


def test_reopen_subcommand_calls_reopen_then_resumes(tmp_path, monkeypatch):
    _FakeAgent.calls = []
    monkeypatch.setattr(cli, "Agent", _FakeAgent)
    monkeypatch.setattr(cli, "load_config", lambda **kw: _FakeConfig())
    workspace = tmp_path / "ws"
    workspace.mkdir()

    rc = cli.main(["reopen", "add a --version flag", "--workspace", str(workspace)])

    assert rc == 0
    kinds = [c[0] for c in _FakeAgent.calls]
    assert kinds == ["__init__", "reopen", "run"]
    assert _FakeAgent.calls[1] == ("reopen", "add a --version flag")
    assert _FakeAgent.calls[2][1] is None  # goal=None
    assert _FakeAgent.calls[2][2] is True  # resume=True


def test_reopen_subcommand_reports_error_without_calling_run(tmp_path, monkeypatch, capsys):
    class _RefusingAgent(_FakeAgent):
        def reopen(self, instructions):
            _FakeAgent.calls.append(("reopen", instructions))
            raise ValueError("no existing session to reopen")

    _FakeAgent.calls = []
    monkeypatch.setattr(cli, "Agent", _RefusingAgent)
    monkeypatch.setattr(cli, "load_config", lambda **kw: _FakeConfig())
    workspace = tmp_path / "ws"
    workspace.mkdir()

    rc = cli.main(["reopen", "more work", "--workspace", str(workspace)])

    assert rc == 1
    kinds = [c[0] for c in _FakeAgent.calls]
    assert kinds == ["__init__", "reopen"]  # run() never called
    assert "no existing session" in capsys.readouterr().err


def test_export_subcommand_calls_export_session(tmp_path, monkeypatch):
    _FakeAgent.calls = []

    class _ExportAgent(_FakeAgent):
        def export_session(self, out_path):
            _FakeAgent.calls.append(("export_session", out_path))

    monkeypatch.setattr(cli, "Agent", _ExportAgent)
    monkeypatch.setattr(cli, "load_config", lambda **kw: _FakeConfig())
    workspace = tmp_path / "ws"
    workspace.mkdir()
    out = tmp_path / "bundle.json"

    rc = cli.main(["export", "--workspace", str(workspace), "--out", str(out)])

    assert rc == 0
    kinds = [c[0] for c in _FakeAgent.calls]
    assert kinds == ["__init__", "export_session"]


def test_import_subcommand_passes_force_flag(tmp_path, monkeypatch):
    _FakeAgent.calls = []

    class _ImportAgent(_FakeAgent):
        def import_session(self, bundle_path, force=False):
            _FakeAgent.calls.append(("import_session", bundle_path, force))

    monkeypatch.setattr(cli, "Agent", _ImportAgent)
    monkeypatch.setattr(cli, "load_config", lambda **kw: _FakeConfig())
    workspace = tmp_path / "ws"
    workspace.mkdir()
    bundle = tmp_path / "bundle.json"
    bundle.write_text("{}")

    rc = cli.main(["import", str(bundle), "--workspace", str(workspace), "--force"])

    assert rc == 0
    assert _FakeAgent.calls[1] == ("import_session", bundle.resolve(), True)


def test_import_lessons_subcommand(tmp_path, monkeypatch):
    _FakeAgent.calls = []

    class _LessonsAgent(_FakeAgent):
        def import_lessons(self, path):
            _FakeAgent.calls.append(("import_lessons", path))
            return 3

    monkeypatch.setattr(cli, "Agent", _LessonsAgent)
    monkeypatch.setattr(cli, "load_config", lambda **kw: _FakeConfig())
    workspace = tmp_path / "ws"
    workspace.mkdir()
    lessons = tmp_path / "lessons.json"
    lessons.write_text("{}")

    rc = cli.main(["import-lessons", str(lessons), "--workspace", str(workspace)])

    assert rc == 0
    assert _FakeAgent.calls[1] == ("import_lessons", lessons.resolve())
