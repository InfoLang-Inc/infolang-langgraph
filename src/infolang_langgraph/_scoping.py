"""Maps LangGraph ``BaseStore`` namespace tuples onto InfoLang namespaces.

LangGraph scopes stored items with a hierarchical ``tuple[str, ...]`` namespace
(e.g. ``("memories", "user-123")``). InfoLang memory banks are keyed by a single
flat namespace string. This module collapses the tuple into one deterministic,
reversible-enough namespace string, following the frozen-contract rule
"namespace tuples map to InfoLang namespaces (join with a stable separator)".

The mapping is:

    ("memories", "user-123")  ->  "lg.memories.user-123"

with a configurable ``prefix`` ("lg" by default) and ``separator`` (".").

The prefix lets :meth:`~infolang_langgraph.store.InfoLangStore.list_namespaces`
tell this store's banks apart from banks written by other InfoLang integrations
sharing the same workspace, and lets us reverse-map a bank name back to a
namespace tuple.

Each segment is sanitized to ``[a-zA-Z0-9_-]`` (the separator and any other
character become ``-``) so the joined string round-trips through a split on the
separator and stays URL-query-safe (recall/list_recent send the namespace as a
query parameter). Two tuples whose *sanitized* segments are identical therefore
collide onto one bank -- e.g. ``("a.b",)`` and ``("a-b",)`` both map to
``"lg.a-b"``. Supply a custom ``namespace_for`` to
:class:`~infolang_langgraph.store.InfoLangStore` if that matters for your keys.
"""

from __future__ import annotations

import re

DEFAULT_NAMESPACE_PREFIX = "lg"
DEFAULT_SEPARATOR = "."

# Everything outside this class (including the separator ".") is replaced with
# "-" so the joined namespace splits back into segments unambiguously.
_UNSAFE_SEGMENT_CHARS = re.compile(r"[^a-zA-Z0-9_-]+")


def _sanitize_segment(segment: str) -> str:
    cleaned = _UNSAFE_SEGMENT_CHARS.sub("-", segment.strip()).strip("-")
    return cleaned or "unknown"


def to_infolang_namespace(
    namespace: tuple[str, ...],
    *,
    prefix: str = DEFAULT_NAMESPACE_PREFIX,
    separator: str = DEFAULT_SEPARATOR,
) -> str:
    """Collapses a LangGraph namespace tuple into an InfoLang namespace string.

    Args:
        namespace: The LangGraph namespace (or namespace prefix) tuple.
        prefix: Bank-name prefix identifying this store's namespaces.
        separator: Character joining the prefix and sanitized segments.

    Returns:
        A flat InfoLang namespace string. An empty ``namespace`` maps to the
        bare ``prefix`` (or ``"root"`` when ``prefix`` is also empty).
    """

    parts = [prefix] if prefix else []
    parts.extend(_sanitize_segment(seg) for seg in namespace)
    return separator.join(parts) if parts else "root"


def from_infolang_namespace(
    bank: str,
    *,
    prefix: str = DEFAULT_NAMESPACE_PREFIX,
    separator: str = DEFAULT_SEPARATOR,
) -> tuple[str, ...] | None:
    """Reverses :func:`to_infolang_namespace` for banks owned by this store.

    Returns ``None`` for any bank name that does not carry this store's
    ``prefix`` -- i.e. a bank written by some other tool -- so callers can
    filter foreign banks out of ``list_namespaces`` results.

    The returned tuple is built from *sanitized* segments, so it may differ
    from the exact tuple originally passed to ``put`` when that tuple contained
    characters outside ``[a-zA-Z0-9_-]``. It always forward-maps back to the
    same bank, so it remains a valid handle for ``get``/``search``/``delete``.
    """

    if prefix:
        head = f"{prefix}{separator}"
        if bank == prefix:
            return ()
        if not bank.startswith(head):
            return None
        remainder = bank[len(head) :]
    else:
        remainder = bank
    if not remainder:
        return ()
    return tuple(remainder.split(separator))
