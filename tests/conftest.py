"""Shared test fixtures.

``FakeInfoLang`` is an in-memory stand-in for ``infolang.InfoLang`` that
implements exactly the public methods ``infolang-langgraph`` calls
(``recall``/``remember``/``remember_batch``/``forget``/``list_recent``/
``list_banks``/``close``) with the same signatures and return types as the real
SDK. Unit tests inject it via ``InfoLangStore(client=fake)`` so no test touches
the network. A ``test_contract.py`` check keeps these signatures honest against
the installed ``infolang`` package.
"""

from __future__ import annotations

import itertools
import re
from typing import Any

import pytest
from infolang import Bank, RecallResult, RememberResult
from infolang.errors import NotFoundError
from infolang.types import Chunk

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


class FakeInfoLang:
    """Minimal in-memory fake of the InfoLang sync client."""

    def __init__(self, *, default_namespace: str = "default") -> None:
        self._default_ns = default_namespace
        self._ids = itertools.count(1)
        # bank -> list of record dicts, in insertion order
        self.banks: dict[str, list[dict[str, Any]]] = {}
        self.closed = False
        self.remember_calls = 0
        self.remember_batch_calls = 0
        self.forget_calls = 0
        self.recall_calls = 0
        self.list_banks_error: Exception | None = None
        self.recall_error: Exception | None = None
        self.list_recent_error: Exception | None = None

    # -- helpers --------------------------------------------------------------

    def _bank(self, namespace: str | None) -> str:
        return namespace if namespace is not None else self._default_ns

    def _new_record(
        self, text: str, source: str | None, tags: Any
    ) -> dict[str, Any]:
        return {
            "id": f"mem-{next(self._ids)}",
            "text": text,
            "source": source,
            "tags": tags,
        }

    # -- public SDK surface ---------------------------------------------------

    def remember(
        self,
        text: str,
        *,
        namespace: str | None = None,
        source: str | None = None,
        tags: str | None = None,
    ) -> RememberResult:
        self.remember_calls += 1
        record = self._new_record(text, source, tags)
        self.banks.setdefault(self._bank(namespace), []).append(record)
        return RememberResult(id=record["id"])

    def remember_batch(
        self,
        items: list[Any],
        *,
        namespace: str | None = None,
        source: str | None = None,
    ) -> list[RememberResult]:
        self.remember_batch_calls += 1
        results: list[RememberResult] = []
        for item in items:
            if isinstance(item, str):
                text, item_source, tags = item, source, None
            else:
                text = item["text"]
                item_source = item.get("source", source)
                tags = item.get("tags")
            record = self._new_record(text, item_source, tags)
            self.banks.setdefault(self._bank(namespace), []).append(record)
            results.append(RememberResult(id=record["id"]))
        return results

    def forget(self, memory_id: str, *, namespace: str | None = None) -> None:
        self.forget_calls += 1
        records = self.banks.get(self._bank(namespace), [])
        for index, record in enumerate(records):
            if record["id"] == memory_id:
                del records[index]
                return
        raise NotFoundError(f"no such memory: {memory_id}", status=404)

    def list_recent(
        self, *, namespace: str | None = None, n: int | None = None
    ) -> list[dict[str, Any]]:
        if self.list_recent_error is not None:
            raise self.list_recent_error
        records = self.banks.get(self._bank(namespace), [])
        # Reverse insertion order to prove callers do not rely on server order.
        ordered = list(reversed(records))
        return ordered[:n] if n is not None else ordered

    def recall(
        self,
        query: str,
        *,
        namespace: str | None = None,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        verbose: bool | None = None,
    ) -> RecallResult:
        self.recall_calls += 1
        if self.recall_error is not None:
            raise self.recall_error
        records = self.banks.get(self._bank(namespace), [])
        query_tokens = _tokens(query)
        scored: list[tuple[float, str, dict[str, Any]]] = []
        for record in records:
            text = record["text"]
            overlap = query_tokens & _tokens(text)
            score = len(overlap) / max(1, len(query_tokens))
            scored.append((score, record["id"], record))
        scored.sort(key=lambda triple: (-triple[0], triple[1]))
        if top_k is not None:
            scored = scored[:top_k]
        chunks = [
            Chunk(id=record["id"], text=record["text"], score=score, tags=record["tags"])
            for score, _, record in scored
        ]
        return RecallResult(chunks=chunks)

    def list_banks(self) -> list[Bank]:
        if self.list_banks_error is not None:
            raise self.list_banks_error
        return [Bank(namespace=name, count=len(records)) for name, records in self.banks.items()]

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake() -> FakeInfoLang:
    return FakeInfoLang()
