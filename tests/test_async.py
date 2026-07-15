from __future__ import annotations

from infolang_langgraph import InfoLangStore
from tests.conftest import FakeInfoLang


async def test_aput_aget_roundtrip(fake: FakeInfoLang) -> None:
    store = InfoLangStore(client=fake)  # type: ignore[arg-type]
    await store.aput(("m",), "k", {"text": "async value"})
    item = await store.aget(("m",), "k")
    assert item is not None
    assert item.value == {"text": "async value"}


async def test_asearch(fake: FakeInfoLang) -> None:
    store = InfoLangStore(client=fake)  # type: ignore[arg-type]
    await store.aput(("m",), "k", {"text": "python rocks"})
    results = await store.asearch(("m",), query="python")
    assert results[0].key == "k"


async def test_adelete(fake: FakeInfoLang) -> None:
    store = InfoLangStore(client=fake)  # type: ignore[arg-type]
    await store.aput(("m",), "k", {"text": "v"})
    await store.adelete(("m",), "k")
    assert await store.aget(("m",), "k") is None


async def test_alist_namespaces(fake: FakeInfoLang) -> None:
    store = InfoLangStore(client=fake)  # type: ignore[arg-type]
    await store.aput(("memories", "u1"), "k", {"text": "v"})
    namespaces = await store.alist_namespaces()
    assert ("memories", "u1") in namespaces


async def test_aclose_borrowed_client(fake: FakeInfoLang) -> None:
    store = InfoLangStore(client=fake)  # type: ignore[arg-type]
    await store.aclose()
    assert fake.closed is False


async def test_async_context_manager(fake: FakeInfoLang) -> None:
    async with InfoLangStore(client=fake) as store:  # type: ignore[arg-type]
        await store.aput(("m",), "k", {"text": "v"})
    assert fake.closed is False


async def test_aclose_owned_client() -> None:
    store = InfoLangStore(api_key="il_test", base_url="http://127.0.0.1:9")
    await store.aclose()
