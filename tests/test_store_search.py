from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from infolang_langgraph import InfoLangStore, _envelope
from tests.conftest import FakeInfoLang


def make_store(fake: FakeInfoLang, **kwargs: object) -> InfoLangStore:
    return InfoLangStore(client=fake, **kwargs)  # type: ignore[arg-type]


def test_search_with_query_ranks_and_scores(fake: FakeInfoLang) -> None:
    store = make_store(fake)
    ns = ("memories", "u1")
    store.put(ns, "a", {"text": "the cat sat on the mat"})
    store.put(ns, "b", {"text": "python programming language"})
    results = store.search(ns, query="python language")
    assert results[0].key == "b"
    assert results[0].score is not None and results[0].score > 0
    assert fake.recall_calls == 1


def test_search_respects_limit(fake: FakeInfoLang) -> None:
    store = make_store(fake)
    ns = ("m",)
    for i in range(5):
        store.put(ns, f"k{i}", {"text": f"memory number {i}"})
    results = store.search(ns, query="memory", limit=2)
    assert len(results) == 2


def test_search_offset_paging(fake: FakeInfoLang) -> None:
    store = make_store(fake)
    ns = ("m",)
    for i in range(5):
        store.put(ns, f"k{i}", {"text": f"shared token item {i}"})
    page1 = store.search(ns, query="shared token", limit=2, offset=0)
    page2 = store.search(ns, query="shared token", limit=2, offset=2)
    keys1 = {r.key for r in page1}
    keys2 = {r.key for r in page2}
    assert keys1.isdisjoint(keys2)
    assert len(page1) == 2 and len(page2) == 2


def test_search_without_query_sorts_newest_first(
    fake: FakeInfoLang, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Deterministic, strictly-increasing timestamps.
    base = datetime(2026, 1, 1, tzinfo=UTC)
    ticks = iter(base + timedelta(seconds=i) for i in range(100))
    monkeypatch.setattr(_envelope, "utcnow", lambda: next(ticks))

    store = make_store(fake)
    ns = ("m",)
    store.put(ns, "old", {"text": "first"})
    store.put(ns, "new", {"text": "second"})
    results = store.search(ns)
    assert [r.key for r in results] == ["new", "old"]
    assert all(r.score is None for r in results)


def test_search_filter_equality(fake: FakeInfoLang) -> None:
    store = make_store(fake)
    ns = ("m",)
    store.put(ns, "a", {"text": "one", "kind": "note"})
    store.put(ns, "b", {"text": "two", "kind": "task"})
    results = store.search(ns, filter={"kind": "task"})
    assert [r.key for r in results] == ["b"]


def test_search_filter_operator(fake: FakeInfoLang) -> None:
    store = make_store(fake)
    ns = ("m",)
    store.put(ns, "a", {"text": "one", "score": 1})
    store.put(ns, "b", {"text": "two", "score": 9})
    results = store.search(ns, filter={"score": {"$gt": 5}})
    assert [r.key for r in results] == ["b"]


def test_search_filter_with_query_overfetches(fake: FakeInfoLang) -> None:
    store = make_store(fake)
    ns = ("m",)
    store.put(ns, "a", {"text": "alpha keyword", "kind": "keep"})
    store.put(ns, "b", {"text": "beta keyword", "kind": "drop"})
    results = store.search(ns, query="keyword", filter={"kind": "keep"})
    assert [r.key for r in results] == ["a"]


def test_search_empty_bank_returns_empty(fake: FakeInfoLang) -> None:
    store = make_store(fake)
    assert store.search(("never",), query="anything") == []
    assert store.search(("never",)) == []


def test_search_query_degrades_on_not_found(fake: FakeInfoLang) -> None:
    from infolang.errors import NotFoundError

    store = make_store(fake)
    fake.recall_error = NotFoundError("no bank", status=404)
    assert store.search(("m",), query="anything") == []


def test_search_skips_foreign_chunks(fake: FakeInfoLang) -> None:
    store = make_store(fake)
    store.put(("m",), "k", {"text": "genuine keyword item"})
    # A foreign (non-envelope) record shares the bank; recall returns it too,
    # but it must be skipped rather than surfaced as a result.
    fake.banks["lg.m"].append(
        {"id": "foreign-1", "text": "foreign keyword text", "tags": None, "source": None}
    )
    results = store.search(("m",), query="keyword")
    assert [r.key for r in results] == ["k"]
