"""``InfoLangStore`` -- a LangGraph :class:`BaseStore` backed by InfoLang memory.

LangGraph's ``BaseStore`` is the sanctioned long-term-memory interface for
LangGraph agents: ``put``/``get``/``search``/``delete`` over hierarchical
``tuple[str, ...]`` namespaces. This class implements it on top of the public
``infolang`` Python SDK (``remember``/``recall``/``forget``/``list_recent``), so
anything that speaks ``BaseStore`` -- ``create_react_agent(store=...)``, the
functional API, custom graphs -- gets InfoLang semantic memory for free.

In current LangGraph the only abstract methods on ``BaseStore`` are ``batch``
and ``abatch``; the public ``get``/``put``/``search``/``delete``/
``list_namespaces`` helpers are concrete and route through ``batch``. So this
class implements ``batch`` (translating each :class:`Op` into InfoLang calls)
and ``abatch`` (running ``batch`` in a worker thread), and the ergonomic public
methods come for free from the base class.

Design notes
------------
* **Namespaces.** Each LangGraph namespace tuple maps to exactly one InfoLang
  bank via :func:`infolang_langgraph._scoping.to_infolang_namespace` (a stable
  ``prefix + separator``-joined string). ``search`` therefore operates on the
  bank equal to the given ``namespace_prefix``; it does not fan out across
  deeper sub-namespaces, because InfoLang ``recall`` is single-bank. Query with
  the full namespace (the common LangGraph memory pattern).
* **Keys / overwrite.** InfoLang has no key column and no update-in-place, so
  ``put`` finds and ``forget``s any prior memory carrying the same key in that
  bank, then ``remember``s the new envelope. ``get``/``delete`` scan the bank
  (bounded by ``max_scan``) and match on the decoded envelope key.
* **TTL / index.** InfoLang exposes no TTL or per-field index knobs, so
  ``PutOp.ttl``/``PutOp.index`` and ``refresh_ttl`` are accepted and ignored.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable, Iterable
from typing import Any

from infolang import InfoLang
from infolang.errors import InfoLangAPIError, NotFoundError
from langgraph.store.base import (
    BaseStore,
    GetOp,
    Item,
    ListNamespacesOp,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
)

from . import _envelope
from ._matching import matches_condition, matches_filter
from ._scoping import (
    DEFAULT_NAMESPACE_PREFIX,
    DEFAULT_SEPARATOR,
    from_infolang_namespace,
    to_infolang_namespace,
)

DEFAULT_SOURCE = "langgraph"
DEFAULT_MAX_SCAN = 1000

NamespaceForFn = Callable[[tuple[str, ...]], str]


class InfoLangStore(BaseStore):
    """A LangGraph ``BaseStore`` implementation backed by InfoLang memory.

    Args:
        client: An existing :class:`infolang.InfoLang` to reuse. Its lifecycle
            then belongs to the caller. Mutually exclusive with ``client_kwargs``.
        namespace_prefix: Prefix applied to every InfoLang bank this store
            writes, so :meth:`list_namespaces` can distinguish this store's
            banks from banks written by other InfoLang integrations sharing the
            workspace. Defaults to ``"lg"``.
        separator: Character joining the prefix and namespace segments.
        source: ``source`` tag written on every stored memory, for provenance.
        max_scan: Upper bound on records fetched from one bank for key lookups
            (``get``/``delete``/``put`` overwrite) and non-semantic ``search``.
        namespace_for: Optional override mapping a namespace tuple to an InfoLang
            bank string. When supplied, ``list_namespaces`` reverse-mapping is
            disabled (the default join is reversible; an arbitrary mapping may
            not be), so ``list_namespaces`` returns ``[]``.
        **client_kwargs: Forwarded to ``InfoLang(...)`` when ``client`` is not
            given (e.g. ``api_key=``, ``base_url=``, ``namespace=``,
            ``workspace=``).

    Raises:
        ValueError: If both ``client`` and ``client_kwargs`` are supplied.
    """

    __slots__ = (
        "_client",
        "_owns_client",
        "_prefix",
        "_separator",
        "_source",
        "_max_scan",
        "_namespace_for",
    )

    supports_ttl = False

    def __init__(
        self,
        *,
        client: InfoLang | None = None,
        namespace_prefix: str = DEFAULT_NAMESPACE_PREFIX,
        separator: str = DEFAULT_SEPARATOR,
        source: str = DEFAULT_SOURCE,
        max_scan: int = DEFAULT_MAX_SCAN,
        namespace_for: NamespaceForFn | None = None,
        **client_kwargs: Any,
    ) -> None:
        if client is not None and client_kwargs:
            raise ValueError(
                "Pass either client=<InfoLang instance> or client construction "
                "kwargs (api_key=, base_url=, ...), not both."
            )
        if max_scan <= 0:
            raise ValueError("max_scan must be positive")
        self._client: InfoLang = client if client is not None else InfoLang(**client_kwargs)
        self._owns_client = client is None
        self._prefix = namespace_prefix
        self._separator = separator
        self._source = source
        self._max_scan = max_scan
        self._namespace_for = namespace_for

    # -- namespace mapping ----------------------------------------------------

    def _bank(self, namespace: tuple[str, ...]) -> str:
        if self._namespace_for is not None:
            return self._namespace_for(namespace)
        return to_infolang_namespace(
            namespace, prefix=self._prefix, separator=self._separator
        )

    # -- batch dispatch (the only abstract methods) ---------------------------

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        results: list[Result] = []
        for op in ops:
            if isinstance(op, GetOp):
                results.append(self._get(op))
            elif isinstance(op, SearchOp):
                results.append(self._search(op))
            elif isinstance(op, PutOp):
                self._put(op)
                results.append(None)
            elif isinstance(op, ListNamespacesOp):
                results.append(self._list_namespaces(op))
            else:  # pragma: no cover - defensive; langgraph passes only the 4 ops
                raise TypeError(f"Unsupported store op: {type(op)!r}")
        return results

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        return await asyncio.to_thread(self.batch, list(ops))

    # -- op handlers ----------------------------------------------------------

    def _scan(self, bank: str) -> list[tuple[str, _envelope.DecodedItem]]:
        """Returns ``(memory_id, decoded)`` for every decodable record in bank.

        Foreign / corrupt records (not written by this library) and records
        missing a memory id are skipped. A bank that does not exist yet returns
        an empty list rather than raising.
        """

        try:
            raw = self._client.list_recent(namespace=bank, n=self._max_scan)
        except NotFoundError:
            return []
        decoded: list[tuple[str, _envelope.DecodedItem]] = []
        for record in raw:
            memory_id = _envelope.record_id(record)
            if memory_id is None:
                continue
            item = _envelope.decode(_envelope.record_text(record))
            if item is None:
                continue
            decoded.append((memory_id, item))
        return decoded

    def _find_by_key(
        self, bank: str, key: str
    ) -> list[tuple[str, _envelope.DecodedItem]]:
        return [pair for pair in self._scan(bank) if pair[1].key == key]

    def _get(self, op: GetOp) -> Item | None:
        bank = self._bank(op.namespace)
        matches = self._find_by_key(bank, op.key)
        if not matches:
            return None
        # Overwrite keeps one record per key, but if duplicates ever exist
        # (e.g. a concurrent writer), return the most recently updated one.
        _, decoded = max(matches, key=lambda pair: pair[1].updated_at)
        return Item(
            value=decoded.value,
            key=decoded.key,
            namespace=op.namespace,
            created_at=decoded.created_at,
            updated_at=decoded.updated_at,
        )

    def _put(self, op: PutOp) -> None:
        bank = self._bank(op.namespace)
        existing = self._find_by_key(bank, op.key)

        if op.value is None:
            # delete
            for memory_id, _ in existing:
                self._forget(bank, memory_id)
            return None

        now = _envelope.utcnow()
        created_at = min((d.created_at for _, d in existing), default=now)
        for memory_id, _ in existing:
            self._forget(bank, memory_id)
        text = _envelope.encode(
            namespace=op.namespace,
            key=op.key,
            value=op.value,
            created_at=created_at,
            updated_at=now,
        )
        self._client.remember(
            text, namespace=bank, source=self._source, tags=f"lg-key:{op.key}"
        )
        return None

    def _forget(self, bank: str, memory_id: str) -> None:
        # A concurrent delete may have already removed it; treat as success.
        with contextlib.suppress(NotFoundError):
            self._client.forget(memory_id, namespace=bank)

    def _search(self, op: SearchOp) -> list[SearchItem]:
        bank = self._bank(op.namespace_prefix)
        window = op.offset + op.limit

        candidates: list[tuple[_envelope.DecodedItem, float | None]]
        if op.query:
            candidates = self._semantic_candidates(bank, op, window)
        else:
            candidates = [
                (decoded, None)
                for _, decoded in sorted(
                    self._scan(bank), key=lambda pair: pair[1].updated_at, reverse=True
                )
            ]

        filtered = [
            (decoded, score)
            for decoded, score in candidates
            if matches_filter(decoded.value, op.filter)
        ]
        page = filtered[op.offset : op.offset + op.limit]
        return [
            SearchItem(
                namespace=op.namespace_prefix,
                key=decoded.key,
                value=decoded.value,
                created_at=decoded.created_at,
                updated_at=decoded.updated_at,
                score=score,
            )
            for decoded, score in page
        ]

    def _semantic_candidates(
        self, bank: str, op: SearchOp, window: int
    ) -> list[tuple[_envelope.DecodedItem, float | None]]:
        # Over-fetch when a structured filter is present, since InfoLang ranks
        # semantically and cannot pre-apply the filter server-side.
        pool = min(self._max_scan, max(window * 4, 50)) if op.filter else max(window, 1)
        assert op.query is not None  # only called from the op.query branch
        try:
            result = self._client.recall(op.query, namespace=bank, top_k=pool)
        except NotFoundError:
            return []
        out: list[tuple[_envelope.DecodedItem, float | None]] = []
        for chunk in result.chunks:
            decoded = _envelope.decode(chunk.text)
            if decoded is not None:
                out.append((decoded, chunk.score))
        return out

    def _list_namespaces(self, op: ListNamespacesOp) -> list[tuple[str, ...]]:
        # Reverse-mapping only works for the default join, not an arbitrary
        # namespace_for override.
        if self._namespace_for is not None:
            return []
        try:
            banks = self._client.list_banks()
        except (AttributeError, NotFoundError, InfoLangAPIError):
            return []

        seen: set[tuple[str, ...]] = set()
        namespaces: list[tuple[str, ...]] = []
        for bank in banks:
            ns = from_infolang_namespace(
                bank.namespace, prefix=self._prefix, separator=self._separator
            )
            if ns is None or ns == ():
                continue
            if op.max_depth is not None:
                ns = ns[: op.max_depth]
            if ns in seen:
                continue
            if op.match_conditions and not all(
                matches_condition(condition, ns) for condition in op.match_conditions
            ):
                continue
            seen.add(ns)
            namespaces.append(ns)

        namespaces.sort()
        return namespaces[op.offset : op.offset + op.limit]

    # -- lifecycle ------------------------------------------------------------

    def close(self) -> None:
        """Closes the underlying InfoLang client, if this store owns it."""

        if self._owns_client:
            self._client.close()

    async def aclose(self) -> None:
        """Async variant of :meth:`close`."""

        if self._owns_client:
            await asyncio.to_thread(self._client.close)

    def __enter__(self) -> InfoLangStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    async def __aenter__(self) -> InfoLangStore:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()
