import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autocoder.lessons import LessonsStore, MAX_LESSONS_STORED


def test_empty_store_returns_none_yet(tmp_path):
    store = LessonsStore(tmp_path / ".autocoder" / "lessons.json")
    assert store.load() == []
    assert store.summary_text() == "(none yet)"


def test_add_and_load_roundtrip(tmp_path):
    store = LessonsStore(tmp_path / ".autocoder" / "lessons.json")
    store.add(context="step: build thing", symptom="it crashed", fix="added missing import")
    lessons = store.load()
    assert len(lessons) == 1
    assert lessons[0].symptom == "it crashed"
    assert lessons[0].fix == "added missing import"


def test_add_ignores_empty_symptom_or_fix(tmp_path):
    store = LessonsStore(tmp_path / ".autocoder" / "lessons.json")
    store.add(context="x", symptom="", fix="something")
    store.add(context="x", symptom="something", fix="")
    assert store.load() == []


def test_summary_text_includes_context_symptom_and_fix(tmp_path):
    store = LessonsStore(tmp_path / ".autocoder" / "lessons.json")
    store.add(context="step: create file", symptom="test -f failed on Windows", fix="used python -c check instead")
    text = store.summary_text()
    assert "create file" in text
    assert "test -f failed on Windows" in text
    assert "used python -c check instead" in text


def test_store_caps_at_max_lessons_dropping_oldest(tmp_path):
    store = LessonsStore(tmp_path / ".autocoder" / "lessons.json")
    for i in range(MAX_LESSONS_STORED + 5):
        store.add(context=f"ctx{i}", symptom=f"symptom{i}", fix=f"fix{i}")
    lessons = store.load()
    assert len(lessons) == MAX_LESSONS_STORED
    # oldest ones (0-4) should have been dropped; newest should remain
    assert lessons[0].symptom == "symptom5"
    assert lessons[-1].symptom == f"symptom{MAX_LESSONS_STORED + 4}"


def test_summary_text_respects_max_lessons_param(tmp_path):
    store = LessonsStore(tmp_path / ".autocoder" / "lessons.json")
    for i in range(5):
        store.add(context=f"ctx{i}", symptom=f"symptom{i}", fix=f"fix{i}")
    text = store.summary_text(max_lessons=2)
    assert "symptom3" in text and "symptom4" in text
    assert "symptom0" not in text


def test_load_survives_corrupted_file(tmp_path):
    path = tmp_path / ".autocoder" / "lessons.json"
    path.parent.mkdir(parents=True)
    path.write_text("not valid json{{{", encoding="utf-8")
    store = LessonsStore(path)
    assert store.load() == []  # doesn't crash, just treats as empty
