import shutil
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autocoder.config import Config
from autocoder.session import SessionStore
from autocoder.planner import RunState


def make_session(tmp_path):
    cfg = Config(workspace_root=tmp_path / "ws")
    return SessionStore(cfg), cfg


def test_session_dir_recreated_after_being_deleted_mid_run(tmp_path):
    """
    Reproduces the real crash: something (git clean -fdx, a human, disk
    cleanup) deletes .autocoder/ out from under a running process. The
    very next save_state/log_event used to raise FileNotFoundError and
    crash the whole run. It must now silently recreate the directory and
    keep going -- the in-memory state being written IS the source of
    truth at that point, recreating the directory loses nothing that
    wasn't already gone.
    """
    store, cfg = make_session(tmp_path)
    state = RunState(goal="g")
    store.save_state(state)
    assert cfg.session_file.exists()

    shutil.rmtree(cfg.session_dir)  # simulate git clean -fdx wiping it
    assert not cfg.session_dir.exists()

    store.save_state(state)  # must not raise
    assert cfg.session_file.exists()

    shutil.rmtree(cfg.session_dir)
    store.log_event("test_event", {})  # must not raise either
    assert cfg.log_file.exists()
