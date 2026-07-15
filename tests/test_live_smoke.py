"""Optional live smoke test against the real InfoLang API.

Skipped unless ``INFOLANG_API_KEY`` is set, and additionally marked ``live`` so
the default ``pytest`` run (``-m 'not live'``) never collects it. It is excluded
from the coverage gate. Only ever touches namespaces prefixed
``ittest-langgraph-`` and clears them in a ``finally`` block, so it is safe to
run against a shared account.

Run it with::

    INFOLANG_API_KEY=il_live_... pytest -m live -v
"""

from __future__ import annotations

import os
import uuid

import pytest

from infolang_langgraph import InfoLangStore

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not os.environ.get("INFOLANG_API_KEY"),
        reason="live smoke test requires INFOLANG_API_KEY",
    ),
]


def test_live_round_trip() -> None:
    prefix = f"ittest-langgraph-{uuid.uuid4().hex[:8]}"
    store = InfoLangStore(namespace_prefix=prefix)
    namespace = ("memories", "user-1")
    key = "fact-1"
    try:
        store.put(namespace, key, {"text": "InfoLang live smoke fact: sky is teal"})

        got = store.get(namespace, key)
        assert got is not None
        assert got.value["text"].endswith("sky is teal")

        hits = store.search(namespace, query="what color is the sky")
        assert any(h.key == key for h in hits)

        store.delete(namespace, key)
        assert store.get(namespace, key) is None
    finally:
        store.delete(namespace, key)
        store.close()
