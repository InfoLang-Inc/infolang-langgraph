"""Wire format for LangGraph store items persisted in InfoLang memory.

A LangGraph store item is a ``(namespace, key, value, created_at, updated_at)``
record where ``value`` is an arbitrary JSON-able dict. InfoLang's ``remember``
only stores a text blob (plus string tags) and returns an opaque memory id --
it has no key column, no structured-value column, and no update-in-place.

Each item is therefore wrapped in an envelope written as the memory's text:

    <human-readable content for semantic recall>
    <<infolang-langgraph:v1>><base64(json(metadata))>

The first line(s) are the value's natural-language content, so InfoLang embeds
something meaningful and ``recall`` ranks it sensibly. The trailing marker plus
base64-encoded JSON carries the exact ``key``/``namespace``/``value``/timestamps
for a lossless round-trip. base64's alphabet (``A-Za-z0-9+/=``) can never
contain the marker, so splitting on the marker is unambiguous.

Records that are not well-formed envelopes written by this library (foreign data
already in a bank, corruption) decode to ``None`` and are skipped by callers
rather than failing the whole read.
"""

from __future__ import annotations

import base64
import json
from datetime import UTC, datetime
from typing import Any

ENVELOPE_VERSION = 1
_MARKER = "<<infolang-langgraph:v1>>"


def utcnow() -> datetime:
    """Timezone-aware current UTC time (used for created_at/updated_at)."""

    return datetime.now(UTC)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _parse_iso(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def content_for_embedding(value: dict[str, Any]) -> str:
    """Derives human-readable text from a value dict for semantic embedding.

    Collects the value's string (and scalar) leaves so InfoLang embeds real
    content rather than JSON punctuation. Falls back to a compact JSON dump when
    the value has no usable text leaves (e.g. all-numeric values).
    """

    parts: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, str):
            text = node.strip()
            if text:
                parts.append(text)
        elif isinstance(node, bool):
            return
        elif isinstance(node, (int, float)):
            parts.append(str(node))
        elif isinstance(node, dict):
            for item in node.values():
                walk(item)
        elif isinstance(node, (list, tuple)):
            for item in node:
                walk(item)

    walk(value)
    if parts:
        return "\n".join(parts)
    return json.dumps(value, separators=(",", ":"), sort_keys=True, default=str)


def encode(
    *,
    namespace: tuple[str, ...],
    key: str,
    value: dict[str, Any],
    created_at: datetime,
    updated_at: datetime,
) -> str:
    """Serializes a store item into the text stored in one InfoLang memory."""

    metadata = {
        "v": ENVELOPE_VERSION,
        "ns": list(namespace),
        "key": key,
        "value": value,
        "created_at": _iso(created_at),
        "updated_at": _iso(updated_at),
    }
    raw = json.dumps(metadata, separators=(",", ":"), sort_keys=True, default=str)
    blob = base64.b64encode(raw.encode("utf-8")).decode("ascii")
    return f"{content_for_embedding(value)}\n{_MARKER}{blob}"


class DecodedItem:
    """A decoded envelope: the fields needed to rebuild a LangGraph ``Item``."""

    __slots__ = ("namespace", "key", "value", "created_at", "updated_at")

    def __init__(
        self,
        *,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        created_at: datetime,
        updated_at: datetime,
    ) -> None:
        self.namespace = namespace
        self.key = key
        self.value = value
        self.created_at = created_at
        self.updated_at = updated_at


def decode(text: str | None) -> DecodedItem | None:
    """Parses stored memory text back into a :class:`DecodedItem`, or ``None``.

    ``None`` is returned for anything that is not a well-formed envelope written
    by this library, so foreign/corrupt records can be skipped.
    """

    if not isinstance(text, str) or _MARKER not in text:
        return None
    blob = text.rsplit(_MARKER, 1)[1].strip()
    try:
        raw = base64.b64decode(blob.encode("ascii"), validate=True)
        payload = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    key = payload.get("key")
    value = payload.get("value")
    ns = payload.get("ns")
    if not isinstance(key, str) or not isinstance(value, dict) or not isinstance(ns, list):
        return None
    if not all(isinstance(segment, str) for segment in ns):
        return None
    created = _parse_iso(payload.get("created_at")) or utcnow()
    updated = _parse_iso(payload.get("updated_at")) or created
    return DecodedItem(
        namespace=tuple(ns),
        key=key,
        value=value,
        created_at=created,
        updated_at=updated,
    )


def display_text(text: str) -> str:
    """Returns the human-readable portion of a memory's text.

    For an envelope written by this library, that is the leading content lines
    (the base64 metadata trailer is dropped). For any other memory (e.g. one
    written by ``create_remember_tool`` as plain text), the text is returned
    unchanged.
    """

    if not isinstance(text, str):
        return ""
    if _MARKER in text:
        return text.split(_MARKER, 1)[0].strip()
    return text


def record_id(record: Any) -> str | None:
    """Extracts a memory id from a raw ``list_recent`` record.

    ``list_recent`` records are untyped dicts; mirror the id-key fallback the
    ``infolang`` SDK itself uses internally (``id`` / ``memory_id`` / ``i``).
    """

    if not isinstance(record, dict):
        return None
    for candidate in ("id", "memory_id", "i"):
        value = record.get(candidate)
        if isinstance(value, str) and value:
            return value
    return None


def record_text(record: Any) -> str | None:
    """Extracts the stored text from a raw ``list_recent`` record."""

    if not isinstance(record, dict):
        return None
    for candidate in ("text", "t", "content"):
        value = record.get(candidate)
        if isinstance(value, str) and value:
            return value
    return None
