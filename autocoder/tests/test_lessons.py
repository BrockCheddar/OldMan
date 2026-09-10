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


def test_new_lesson_gets_an_id_and_is_active(tmp_path):
    store = LessonsStore(tmp_path / ".autocoder" / "lessons.json")
    store.add(context="x", symptom="crashed", fix="added import")
    lessons = store.load()
    assert lessons[0].active is True
    assert lessons[0].id  # non-empty


def test_supersede_by_id_deactivates_old_and_hides_from_summary(tmp_path):
    store = LessonsStore(tmp_path / ".autocoder" / "lessons.json")
    store.add(context="x", symptom="old bug", fix="workaround")
    first_id = store.load()[0].id
    store.add(context="x", symptom="old bug", fix="real fix", supersedes=first_id)

    lessons = {l.id: l for l in store.load()}
    assert lessons[first_id].active is False
    assert len(lessons) == 2  # full history preserved on disk

    text = store.summary_text()
    assert "real fix" in text
    assert "workaround" not in text


def test_supersede_with_unknown_id_does_not_deactivate_anything(tmp_path):
    store = LessonsStore(tmp_path / ".autocoder" / "lessons.json")
    store.add(context="x", symptom="a", fix="b")
    store.add(context="x", symptom="c", fix="d", supersedes="nonexistent-id")
    lessons = store.load()
    assert all(l.active for l in lessons)
    assert lessons[-1].supersedes is None


def test_cap_evicts_inactive_before_touching_active(tmp_path):
    store = LessonsStore(tmp_path / ".autocoder" / "lessons.json")
    kept_symptom = "load-bearing lesson, never superseded"
    store.add(context="x", symptom=kept_symptom, fix="fix")
    prev_id = None
    for i in range(MAX_LESSONS_STORED + 5):
        store.add(context="x", symptom=f"noise{i}", fix=f"fix{i}", supersedes=prev_id)
        prev_id = store.load()[-1].id
    lessons = store.load()
    assert len(lessons) == MAX_LESSONS_STORED
    assert any(l.active and l.symptom == kept_symptom for l in lessons)


def test_add_stores_intent_and_render_includes_it(tmp_path):
    store = LessonsStore(tmp_path / ".autocoder" / "lessons.json")
    store.add(context="x", symptom="import error", fix="added __init__.py",
              intent="building the contact-book package layout")
    lesson = store.load()[0]
    assert lesson.intent == "building the contact-book package layout"
    text = store.summary_text()
    assert "building the contact-book package layout" in text


def test_find_relevant_matches_on_keyword_overlap(tmp_path):
    store = LessonsStore(tmp_path / ".autocoder" / "lessons.json")
    store.add(context="step: parse CSV files", symptom="pandas raised a UnicodeDecodeError",
              fix="opened with encoding='utf-8-sig'", intent="importing legacy CSV exports")
    store.add(context="step: send emails", symptom="SMTP auth failed",
              fix="used an app password instead", intent="notifying users")

    matches = store.find_relevant("parsing a CSV file that raises decode errors")
    assert len(matches) == 1
    assert matches[0].symptom == "pandas raised a UnicodeDecodeError"


def test_find_relevant_returns_empty_when_nothing_matches(tmp_path):
    store = LessonsStore(tmp_path / ".autocoder" / "lessons.json")
    store.add(context="x", symptom="totally unrelated database migration issue", fix="ran alembic upgrade")
    assert store.find_relevant("completely different topic about image resizing") == []


def test_find_relevant_ignores_inactive_lessons(tmp_path):
    store = LessonsStore(tmp_path / ".autocoder" / "lessons.json")
    store.add(context="x", symptom="csv parsing encoding error", fix="old fix")
    old_id = store.load()[0].id
    store.add(context="x", symptom="csv parsing encoding error", fix="better fix", supersedes=old_id)

    matches = store.find_relevant("csv parsing encoding error")
    assert len(matches) == 1
    assert matches[0].fix == "better fix"


def test_find_relevant_ranks_higher_overlap_first(tmp_path):
    store = LessonsStore(tmp_path / ".autocoder" / "lessons.json")
    store.add(context="x", symptom="socket timeout connecting to redis cache", fix="fix a",
              intent="caching layer")
    store.add(context="x", symptom="redis cache eviction policy misconfigured causing timeout errors",
              fix="fix b", intent="redis cache tuning")

    matches = store.find_relevant("redis cache timeout errors during eviction")
    assert len(matches) == 2
    assert matches[0].fix == "fix b"  # more overlapping keywords
