"""Contract checks against the *installed* frameworks.

Keeps the fake client and this integration honest: if the real ``infolang`` SDK
or LangGraph ``BaseStore`` surface drifts from what this package targets, these
fail fast instead of the fake silently masking the change.
"""

from __future__ import annotations

import inspect

from infolang import InfoLang
from langgraph.store.base import BaseStore

from infolang_langgraph import InfoLangStore
from tests.conftest import FakeInfoLang


def test_infolang_client_exposes_methods_we_use() -> None:
    for name in ("recall", "remember", "remember_batch", "forget", "list_recent", "list_banks"):
        assert callable(getattr(InfoLang, name)), name


def test_fake_matches_real_signatures() -> None:
    # The fake must accept the same keyword arguments this package passes.
    for name in ("recall", "remember", "remember_batch", "forget", "list_recent"):
        real = set(inspect.signature(getattr(InfoLang, name)).parameters)
        fake = set(inspect.signature(getattr(FakeInfoLang, name)).parameters)
        # Real client uses **kwargs pass-through; assert the fake covers the
        # keyword names we actually rely on.
        for kwarg in ("namespace",):
            if kwarg in real or "kwargs" in real:
                assert kwarg in fake, f"{name} missing {kwarg}"


def test_basestore_abstract_methods_are_batch_only() -> None:
    # This package implements exactly these; if LangGraph adds more, we must too.
    assert BaseStore.__abstractmethods__ == frozenset({"batch", "abatch"})


def test_infolangstore_is_a_basestore() -> None:
    assert issubclass(InfoLangStore, BaseStore)


def test_infolangstore_implements_abstract_methods(fake: FakeInfoLang) -> None:
    store = InfoLangStore(client=fake)  # type: ignore[arg-type]
    assert not getattr(store, "__abstractmethods__", frozenset())
    # instantiation would have raised if abstract methods were missing
    assert isinstance(store, BaseStore)


def test_public_helpers_route_through_batch(fake: FakeInfoLang) -> None:
    # get/put/search/delete/list_namespaces are inherited and must work.
    store = InfoLangStore(client=fake)  # type: ignore[arg-type]
    store.put(("m",), "k", {"text": "v"})
    assert store.get(("m",), "k") is not None
    assert store.search(("m",), query="v")
    store.delete(("m",), "k")
    assert store.get(("m",), "k") is None
