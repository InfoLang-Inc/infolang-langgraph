from __future__ import annotations

import pytest

from infolang_langgraph import InfoLangStore
from tests.conftest import FakeInfoLang


def make_store(fake: FakeInfoLang, **kwargs: object) -> InfoLangStore:
    return InfoLangStore(client=fake, **kwargs)  # type: ignore[arg-type]


def test_put_then_get_roundtrip(fake: FakeInfoLang) -> None:
    store = make_store(fake)
    store.put(("memories", "u1"), "k1", {"text": "Ada prefers dark mode"})
    item = store.get(("memories", "u1"), "k1")
    assert item is not None
    assert item.key == "k1"
    assert item.namespace == ("memories", "u1")
    assert item.value == {"text": "Ada prefers dark mode"}


def test_get_missing_returns_none(fake: FakeInfoLang) -> None:
    store = make_store(fake)
    assert store.get(("memories", "u1"), "missing") is None


def test_get_on_unknown_bank_returns_none(fake: FakeInfoLang) -> None:
    store = make_store(fake)
    assert store.get(("never", "written"), "k") is None


def test_put_writes_to_mapped_bank(fake: FakeInfoLang) -> None:
    store = make_store(fake)
    store.put(("memories", "u1"), "k1", {"text": "hi"})
    assert "lg.memories.u1" in fake.banks


def test_overwrite_keeps_single_record_and_created_at(fake: FakeInfoLang) -> None:
    store = make_store(fake)
    store.put(("m",), "k", {"text": "v1"})
    first = store.get(("m",), "k")
    assert first is not None
    store.put(("m",), "k", {"text": "v2"})
    # Old memory forgotten, exactly one remains.
    assert len(fake.banks["lg.m"]) == 1
    second = store.get(("m",), "k")
    assert second is not None
    assert second.value == {"text": "v2"}
    assert second.created_at == first.created_at
    assert second.updated_at >= first.updated_at


def test_delete_removes_item(fake: FakeInfoLang) -> None:
    store = make_store(fake)
    store.put(("m",), "k", {"text": "v"})
    store.delete(("m",), "k")
    assert store.get(("m",), "k") is None
    assert fake.forget_calls == 1


def test_delete_missing_is_noop(fake: FakeInfoLang) -> None:
    store = make_store(fake)
    store.delete(("m",), "nope")  # no raise
    assert fake.forget_calls == 0


def test_scan_degrades_on_not_found(fake: FakeInfoLang) -> None:
    from infolang.errors import NotFoundError

    store = make_store(fake)
    fake.list_recent_error = NotFoundError("no bank", status=404)
    assert store.get(("m",), "k") is None


def test_record_without_id_is_skipped(fake: FakeInfoLang) -> None:
    store = make_store(fake)
    store.put(("m",), "k", {"text": "real"})
    # A record missing an id cannot be managed; it must be ignored.
    fake.banks["lg.m"].append({"text": "no id here", "tags": None, "source": None})
    item = store.get(("m",), "k")
    assert item is not None
    assert item.value == {"text": "real"}


def test_foreign_record_is_skipped(fake: FakeInfoLang) -> None:
    store = make_store(fake)
    # Seed a foreign (non-envelope) record directly into the mapped bank.
    fake.banks.setdefault("lg.m", []).append(
        {"id": "foreign-1", "text": "not an envelope", "tags": None, "source": None}
    )
    store.put(("m",), "k", {"text": "mine"})
    item = store.get(("m",), "k")
    assert item is not None
    assert item.value == {"text": "mine"}


def test_forget_race_is_swallowed(fake: FakeInfoLang) -> None:
    store = make_store(fake)
    # Directly exercise the NotFoundError branch in _forget.
    store._forget("lg.m", "does-not-exist")  # no raise


def test_client_and_kwargs_conflict_raises(fake: FakeInfoLang) -> None:
    with pytest.raises(ValueError, match="not both"):
        InfoLangStore(client=fake, api_key="x")  # type: ignore[arg-type]


def test_max_scan_must_be_positive(fake: FakeInfoLang) -> None:
    with pytest.raises(ValueError, match="max_scan must be positive"):
        make_store(fake, max_scan=0)


def test_borrowed_client_not_closed(fake: FakeInfoLang) -> None:
    store = make_store(fake)
    store.close()
    assert fake.closed is False


def test_owned_client_is_closed() -> None:
    # Constructing from kwargs builds (and owns) a real InfoLang client; close()
    # tears down its httpx transport without any network call.
    store = InfoLangStore(api_key="il_test", base_url="http://127.0.0.1:9")
    store.close()


def test_context_manager_closes(fake: FakeInfoLang) -> None:
    with make_store(fake) as store:
        store.put(("m",), "k", {"text": "v"})
    assert fake.closed is False  # borrowed client not closed


def test_custom_namespace_prefix(fake: FakeInfoLang) -> None:
    store = make_store(fake, namespace_prefix="app", separator="_")
    store.put(("m", "u"), "k", {"text": "v"})
    assert "app_m_u" in fake.banks


def test_source_tag_recorded(fake: FakeInfoLang) -> None:
    store = make_store(fake, source="my-store")
    store.put(("m",), "k", {"text": "v"})
    assert fake.banks["lg.m"][0]["source"] == "my-store"
