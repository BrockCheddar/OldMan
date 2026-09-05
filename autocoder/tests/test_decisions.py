import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autocoder.decisions import DecisionsStore, MAX_DECISIONS_STORED


def test_empty_store_returns_none_yet(tmp_path):
    store = DecisionsStore(tmp_path / ".autocoder" / "decisions.json")
    assert store.load() == []
    assert store.summary_text() == "(none yet)"


def test_add_and_load_roundtrip(tmp_path):
    store = DecisionsStore(tmp_path / ".autocoder" / "decisions.json")
    store.add(decision="Use SQLAlchemy ORM, not raw sqlite3", originating_step="step: db setup")
    decisions = store.load()
    assert len(decisions) == 1
    assert decisions[0].decision == "Use SQLAlchemy ORM, not raw sqlite3"
    assert decisions[0].originating_step == "step: db setup"
    assert decisions[0].active is True
    assert decisions[0].id  # non-empty


def test_add_ignores_blank_decision(tmp_path):
    store = DecisionsStore(tmp_path / ".autocoder" / "decisions.json")
    assert store.add(decision="   ") is None
    assert store.load() == []


def test_summary_text_shows_id_and_text(tmp_path):
    store = DecisionsStore(tmp_path / ".autocoder" / "decisions.json")
    result = store.add(decision="REST API, JSON bodies")
    text = store.summary_text()
    assert f"[{result.id}]" in text
    assert "REST API, JSON bodies" in text


def test_supersede_by_id_deactivates_old_and_hides_from_summary(tmp_path):
    store = DecisionsStore(tmp_path / ".autocoder" / "decisions.json")
    first = store.add(decision="Use raw sqlite3 directly")
    second = store.add(decision="Use SQLAlchemy ORM, not raw sqlite3", supersedes=first.id)

    assert second.superseded_found is True
    decisions = {d.id: d for d in store.load()}
    assert decisions[first.id].active is False
    assert decisions[second.id].active is True

    text = store.summary_text()
    assert "SQLAlchemy" in text
    assert "raw sqlite3 directly" not in text


def test_supersede_with_unknown_id_does_not_deactivate_anything(tmp_path):
    store = DecisionsStore(tmp_path / ".autocoder" / "decisions.json")
    store.add(decision="REST API, JSON bodies")
    result = store.add(decision="something else", supersedes="nonexistent-id")

    assert result.superseded_found is False
    # nothing was wrongly deactivated, and the new entry doesn't record a
    # supersedes link to an id that never matched
    decisions = store.load()
    assert all(d.active for d in decisions)
    assert decisions[-1].supersedes is None


def test_superseded_entries_stay_on_disk(tmp_path):
    store = DecisionsStore(tmp_path / ".autocoder" / "decisions.json")
    first = store.add(decision="old choice")
    store.add(decision="new choice", supersedes=first.id)
    assert len(store.load()) == 2  # full history preserved


def test_cap_evicts_inactive_before_touching_active(tmp_path):
    store = DecisionsStore(tmp_path / ".autocoder" / "decisions.json")
    # Fill with superseded (inactive) pairs so the store is well over cap,
    # all as inactive/active chains, then add one more active decision.
    prev = None
    for i in range(MAX_DECISIONS_STORED + 5):
        result = store.add(decision=f"choice {i}", supersedes=prev.id if prev else None)
        prev = result
    decisions = store.load()
    assert len(decisions) == MAX_DECISIONS_STORED
    # the single most recent decision is still active and present
    assert any(d.active and d.decision == f"choice {MAX_DECISIONS_STORED + 4}" for d in decisions)


def test_cap_never_drops_active_entries_while_inactive_ones_remain(tmp_path):
    store = DecisionsStore(tmp_path / ".autocoder" / "decisions.json")
    # One decision that stays active and is never superseded...
    kept = store.add(decision="load-bearing decision, never superseded")
    # ...then pile on unrelated superseded pairs past the cap.
    prev = None
    for i in range(MAX_DECISIONS_STORED + 5):
        prev = store.add(decision=f"noise {i}", supersedes=prev.id if prev else None)
    decisions = {d.id: d for d in store.load()}
    assert kept.id in decisions
    assert decisions[kept.id].active is True


def test_summary_text_respects_max_decisions_param(tmp_path):
    store = DecisionsStore(tmp_path / ".autocoder" / "decisions.json")
    for i in range(5):
        store.add(decision=f"choice {i}")
    text = store.summary_text(max_decisions=2)
    assert "choice 3" in text and "choice 4" in text
    assert "choice 0" not in text


def test_load_survives_corrupted_file(tmp_path):
    path = tmp_path / ".autocoder" / "decisions.json"
    path.parent.mkdir(parents=True)
    path.write_text("not valid json{{{", encoding="utf-8")
    store = DecisionsStore(path)
    assert store.load() == []
