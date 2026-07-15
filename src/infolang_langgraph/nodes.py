"""Prebuilt LangGraph nodes for automatic InfoLang memory (no tool calls).

Two factories build plain graph-node callables ``(state) -> state_update``:

* :func:`create_recall_node` -- **recall node**: before the LLM runs, recall
  memories relevant to the latest user turn and inject them into the state.
* :func:`create_retain_node` -- **retain node**: after a turn, persist the
  latest exchange to InfoLang so a future run can recall it.

The returned callables are synchronous; LangGraph runs sync nodes in its own
worker threadpool during async graph execution, so they are safe in both sync
and async graphs without blocking the event loop.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from infolang import InfoLang
from infolang.errors import NotFoundError
from langchain_core.messages import SystemMessage

from . import _envelope

NodeFn = Callable[[Mapping[str, Any]], dict[str, Any]]

_HUMAN_ROLES = frozenset({"human", "user"})
_AI_ROLES = frozenset({"ai", "assistant"})

DEFAULT_CONTEXT_KEY = "recalled_memories"
DEFAULT_SYSTEM_TEMPLATE = "Relevant memories from earlier:\n{context}"
DEFAULT_RETAIN_SOURCE = "langgraph-node"


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, (list, tuple)):
        pieces: list[str] = []
        for part in content:
            if isinstance(part, str):
                pieces.append(part)
            elif isinstance(part, Mapping):
                text = part.get("text")
                if isinstance(text, str):
                    pieces.append(text)
        return "\n".join(piece for piece in pieces if piece.strip()).strip()
    return ""


def _message_role_text(message: Any) -> tuple[str | None, str]:
    """Extracts ``(role, text)`` from a LangChain/dict/tuple message."""

    role = getattr(message, "type", None)
    if role is not None:
        return str(role), _content_to_text(getattr(message, "content", ""))
    if isinstance(message, Mapping):
        raw_role = message.get("role") or message.get("type")
        return (
            str(raw_role) if raw_role is not None else None,
            _content_to_text(message.get("content", "")),
        )
    if isinstance(message, (list, tuple)) and len(message) == 2:
        return str(message[0]), _content_to_text(message[1])
    if isinstance(message, str):
        return None, message.strip()
    return None, ""


def _messages(state: Mapping[str, Any], messages_key: str) -> list[Any]:
    raw = state.get(messages_key)
    return list(raw) if isinstance(raw, (list, tuple)) else []


def _latest_role_text(
    state: Mapping[str, Any], messages_key: str, roles: frozenset[str]
) -> str:
    for message in reversed(_messages(state, messages_key)):
        role, text = _message_role_text(message)
        if role in roles and text:
            return text
    return ""


def _as_tag_list(tags: str | Sequence[str] | None) -> list[str]:
    if tags is None:
        return []
    if isinstance(tags, str):
        return [tags]
    return [str(tag) for tag in tags]


def create_recall_node(
    client: InfoLang,
    *,
    namespace: str | None = None,
    top_k: int = 5,
    min_score: float | None = None,
    messages_key: str = "messages",
    context_key: str = DEFAULT_CONTEXT_KEY,
    inject_system_message: bool = False,
    system_template: str = DEFAULT_SYSTEM_TEMPLATE,
    query_fn: Callable[[Mapping[str, Any]], str] | None = None,
) -> NodeFn:
    """Builds a recall node that injects relevant memories before the LLM runs.

    The node derives a query (by default, the latest human message in
    ``state[messages_key]``; override with ``query_fn``), recalls from InfoLang,
    and either:

    * writes the formatted memories to ``state[context_key]`` (default), or
    * appends a :class:`~langchain_core.messages.SystemMessage` to
      ``state[messages_key]`` when ``inject_system_message=True`` (requires the
      messages channel to use an append reducer, e.g. ``add_messages``).

    When there is no query, or nothing is recalled, the node is a no-op
    (returns ``{}`` in system-message mode, or an empty context string).
    """

    def node(state: Mapping[str, Any]) -> dict[str, Any]:
        query = query_fn(state) if query_fn is not None else _latest_role_text(
            state, messages_key, _HUMAN_ROLES
        )
        if not query:
            return {}
        try:
            result = client.recall(query, namespace=namespace, top_k=top_k)
        except NotFoundError:
            context = ""
        else:
            chunks = [
                chunk
                for chunk in result.chunks
                if min_score is None or chunk.score is None or chunk.score >= min_score
            ]
            context = "\n".join(
                f"- {_envelope.display_text(chunk.text)}" for chunk in chunks
            )
        if inject_system_message:
            if not context:
                return {}
            return {"messages": [SystemMessage(content=system_template.format(context=context))]}
        return {context_key: context}

    return node


def create_retain_node(
    client: InfoLang,
    *,
    namespace: str | None = None,
    source: str = DEFAULT_RETAIN_SOURCE,
    tags: str | Sequence[str] | None = None,
    messages_key: str = "messages",
    content_fn: Callable[[Mapping[str, Any]], Sequence[str]] | None = None,
) -> NodeFn:
    """Builds a retain node that persists the latest turn to InfoLang.

    By default the node stores the most recent human message and the most recent
    AI message from ``state[messages_key]`` (each tagged with its role) in one
    ``remember_batch`` call. Override extraction with ``content_fn``, which
    receives the state and returns the list of strings to store.

    The node is a no-op (returns ``{}``) when there is nothing to store, and it
    never mutates the graph state.
    """

    def node(state: Mapping[str, Any]) -> dict[str, Any]:
        base_tags = _as_tag_list(tags)
        items: list[dict[str, Any]] = []
        if content_fn is not None:
            for text in content_fn(state):
                if text and text.strip():
                    items.append({"text": text, "source": source, "tags": list(base_tags)})
        else:
            human = _latest_role_text(state, messages_key, _HUMAN_ROLES)
            ai = _latest_role_text(state, messages_key, _AI_ROLES)
            if human:
                items.append(
                    {"text": human, "source": source, "tags": [*base_tags, "role:human"]}
                )
            if ai:
                items.append(
                    {"text": ai, "source": source, "tags": [*base_tags, "role:ai"]}
                )
        if not items:
            return {}
        client.remember_batch(items, namespace=namespace, source=source)
        return {}

    return node
