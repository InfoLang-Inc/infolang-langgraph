from __future__ import annotations

from infolang_langgraph import InfoLangStore
from tests.conftest import FakeInfoLang


def make_store(fake: FakeInfoLang, **kwargs: object) -> InfoLangStore:
    return InfoLangStore(client=fake, **kwargs)  # type: ignore[arg-type]


def seed(store: InfoLangStore) -> None:
    store.put(("memories", "u1"), "k", {"text": "a"})
    store.put(("memories", "u2"), "k", {"text": "b"})
    store.put(("tasks", "u1"), "k", {"text": "c"})


def test_list_namespaces_returns_tuples(fake: FakeInfoLang) -> None:
    store = make_store(fake)
    seed(store)
    namespaces = store.list_namespaces()
    assert ("memories", "u1") in namespaces
    assert ("memories", "u2") in namespaces
    assert ("tasks", "u1") in namespaces


def test_list_namespaces_filters_foreign_banks(fake: FakeInfoLang) -> None:
    store = make_store(fake)
    seed(store)
    # A bank written by another tool (no "lg." prefix) must be ignored.
    fake.banks["oa-session-xyz"] = [{"id": "x", "text": "foreign", "tags": None, "source": None}]
    namespaces = store.list_namespaces()
    assert all(ns[0] in {"memories", "tasks"} for ns in namespaces)


def test_list_namespaces_max_depth_dedupes(fake: FakeInfoLang) -> None:
    store = make_store(fake)
    seed(store)
    namespaces = store.list_namespaces(max_depth=1)
    assert ("memories",) in namespaces
    assert ("tasks",) in namespaces
    assert namespaces.count(("memories",)) == 1


def test_list_namespaces_prefix_condition(fake: FakeInfoLang) -> None:
    store = make_store(fake)
    seed(store)
    namespaces = store.list_namespaces(prefix=("memories",))
    assert set(namespaces) == {("memories", "u1"), ("memories", "u2")}


def test_list_namespaces_suffix_condition(fake: FakeInfoLang) -> None:
    store = make_store(fake)
    seed(store)
    namespaces = store.list_namespaces(suffix=("u1",))
    assert set(namespaces) == {("memories", "u1"), ("tasks", "u1")}


def test_list_namespaces_limit_and_offset(fake: FakeInfoLang) -> None:
    store = make_store(fake)
    seed(store)
    all_ns = store.list_namespaces()
    assert all_ns == sorted(all_ns)
    first = store.list_namespaces(limit=1, offset=0)
    second = store.list_namespaces(limit=1, offset=1)
    assert len(first) == 1 and len(second) == 1
    assert first != second


def test_list_namespaces_disabled_with_custom_mapping(fake: FakeInfoLang) -> None:
    store = make_store(fake, namespace_for=lambda ns: "custom." + ".".join(ns))
    store.put(("memories", "u1"), "k", {"text": "a"})
    assert store.list_namespaces() == []


def test_list_namespaces_degrades_on_error(fake: FakeInfoLang) -> None:
    from infolang.errors import NotFoundError

    store = make_store(fake)
    seed(store)
    fake.list_banks_error = NotFoundError("banks route unavailable", status=404)
    assert store.list_namespaces() == []
