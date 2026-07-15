"""LangChain tools that let a tool-calling agent read and write InfoLang memory.

Two factories return ready-to-bind :class:`langchain_core.tools.BaseTool`
instances:

* :func:`create_recall_tool` -- the agent searches long-term memory.
* :func:`create_remember_tool` -- the agent saves a fact to long-term memory.

Both wrap the public ``infolang`` SDK. Each tool exposes a sync ``func`` and an
async ``coroutine`` (the async path runs the sync client in a worker thread) so
the tools work in both ``AgentExecutor`` and async LangGraph runs.
"""

from __future__ import annotations

import asyncio

from infolang import InfoLang, RecallResult
from infolang.errors import NotFoundError
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from . import _envelope

DEFAULT_RECALL_NAME = "infolang_recall"
DEFAULT_RECALL_DESCRIPTION = (
    "Search your long-term memory for facts, preferences, or context relevant "
    "to a question. Use this before answering when the user may have told you "
    "something earlier. Input is a natural-language search query."
)
DEFAULT_REMEMBER_NAME = "infolang_remember"
DEFAULT_REMEMBER_DESCRIPTION = (
    "Save a fact to your long-term memory so you can recall it in future "
    "conversations. Input is the single self-contained statement to remember."
)
_NO_RESULTS = "No relevant memories found."


class _RecallInput(BaseModel):
    query: str = Field(description="Natural-language description of what to look up.")


class _RememberInput(BaseModel):
    text: str = Field(description="A self-contained fact to store for later recall.")


def _format_recall(result: RecallResult, min_score: float | None) -> str:
    chunks = [
        chunk
        for chunk in result.chunks
        if min_score is None or chunk.score is None or chunk.score >= min_score
    ]
    if not chunks:
        return _NO_RESULTS
    return "\n".join(
        f"{index}. {_envelope.display_text(chunk.text)}"
        for index, chunk in enumerate(chunks, start=1)
    )


def create_recall_tool(
    client: InfoLang,
    *,
    namespace: str | None = None,
    top_k: int = 5,
    min_score: float | None = None,
    name: str = DEFAULT_RECALL_NAME,
    description: str = DEFAULT_RECALL_DESCRIPTION,
) -> BaseTool:
    """Builds a tool that recalls memories from InfoLang.

    Args:
        client: The :class:`infolang.InfoLang` client to query.
        namespace: InfoLang namespace to search. ``None`` uses the client's
            default namespace.
        top_k: Maximum number of memories to return.
        min_score: Drop chunks scoring below this similarity threshold (InfoLang
            treats scores below ``0.85`` as weak). ``None`` keeps all chunks.
        name: Tool name exposed to the LLM.
        description: Tool description exposed to the LLM.

    Returns:
        A :class:`langchain_core.tools.BaseTool` returning a formatted,
        newline-numbered list of memory texts (or a "no results" message).
    """

    def _recall(query: str) -> str:
        try:
            result = client.recall(query, namespace=namespace, top_k=top_k)
        except NotFoundError:
            return _NO_RESULTS
        return _format_recall(result, min_score)

    async def _arecall(query: str) -> str:
        return await asyncio.to_thread(_recall, query)

    return StructuredTool.from_function(
        func=_recall,
        coroutine=_arecall,
        name=name,
        description=description,
        args_schema=_RecallInput,
    )


def create_remember_tool(
    client: InfoLang,
    *,
    namespace: str | None = None,
    source: str = "langgraph-tool",
    tags: str | None = None,
    name: str = DEFAULT_REMEMBER_NAME,
    description: str = DEFAULT_REMEMBER_DESCRIPTION,
) -> BaseTool:
    """Builds a tool that stores a memory in InfoLang.

    Args:
        client: The :class:`infolang.InfoLang` client to write with.
        namespace: InfoLang namespace to write to. ``None`` uses the client's
            default namespace.
        source: ``source`` tag recorded on every stored memory.
        tags: Optional comma-separated tag string recorded on stored memories.
        name: Tool name exposed to the LLM.
        description: Tool description exposed to the LLM.

    Returns:
        A :class:`langchain_core.tools.BaseTool` returning a short confirmation
        string (including the new memory id when the runtime reports one).
    """

    def _remember(text: str) -> str:
        result = client.remember(text, namespace=namespace, source=source, tags=tags)
        if result.memory_id:
            return f"Stored memory {result.memory_id}."
        return "Stored memory."

    async def _aremember(text: str) -> str:
        return await asyncio.to_thread(_remember, text)

    return StructuredTool.from_function(
        func=_remember,
        coroutine=_aremember,
        name=name,
        description=description,
        args_schema=_RememberInput,
    )
