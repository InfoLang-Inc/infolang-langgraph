from __future__ import annotations

from infolang import RememberResult
from langchain_core.tools import BaseTool

from infolang_langgraph import InfoLangStore, create_recall_tool, create_remember_tool
from tests.conftest import FakeInfoLang


def test_recall_tool_is_basetool_with_defaults(fake: FakeInfoLang) -> None:
    tool = create_recall_tool(fake)  # type: ignore[arg-type]
    assert isinstance(tool, BaseTool)
    assert tool.name == "infolang_recall"
    assert "memory" in tool.description.lower()


def test_recall_tool_formats_results(fake: FakeInfoLang) -> None:
    fake.remember("Ada likes tea", namespace="notes")
    fake.remember("Grace likes coffee", namespace="notes")
    tool = create_recall_tool(fake, namespace="notes")  # type: ignore[arg-type]
    out = tool.invoke({"query": "tea"})
    assert "1. Ada likes tea" in out


def test_recall_tool_empty_message(fake: FakeInfoLang) -> None:
    tool = create_recall_tool(fake, namespace="empty")  # type: ignore[arg-type]
    assert tool.invoke({"query": "anything"}) == "No relevant memories found."


def test_recall_tool_min_score_filters(fake: FakeInfoLang) -> None:
    fake.remember("apple banana", namespace="n")
    tool = create_recall_tool(fake, namespace="n", min_score=0.99)  # type: ignore[arg-type]
    # Query overlaps only partially -> score below threshold -> filtered out.
    assert tool.invoke({"query": "apple orange kiwi"}) == "No relevant memories found."


def test_recall_tool_handles_not_found(fake: FakeInfoLang) -> None:
    from infolang.errors import NotFoundError

    fake.recall_error = NotFoundError("no bank", status=404)
    tool = create_recall_tool(fake, namespace="n")  # type: ignore[arg-type]
    assert tool.invoke({"query": "x"}) == "No relevant memories found."


def test_recall_tool_strips_envelope(fake: FakeInfoLang) -> None:
    store = InfoLangStore(client=fake)  # type: ignore[arg-type]
    store.put(("notes",), "k", {"text": "clean visible content"})
    tool = create_recall_tool(fake, namespace="lg.notes")  # type: ignore[arg-type]
    out = tool.invoke({"query": "content"})
    assert "clean visible content" in out
    assert "infolang-langgraph" not in out


async def test_recall_tool_async(fake: FakeInfoLang) -> None:
    fake.remember("async memory", namespace="n")
    tool = create_recall_tool(fake, namespace="n")  # type: ignore[arg-type]
    out = await tool.ainvoke({"query": "async"})
    assert "async memory" in out


def test_remember_tool_stores_and_reports_id(fake: FakeInfoLang) -> None:
    tool = create_remember_tool(fake, namespace="n")  # type: ignore[arg-type]
    out = tool.invoke({"text": "remember this"})
    assert out.startswith("Stored memory mem-")
    assert fake.banks["n"][0]["text"] == "remember this"


def test_remember_tool_records_source_and_tags(fake: FakeInfoLang) -> None:
    tool = create_remember_tool(fake, namespace="n", source="tool-src", tags="a,b")  # type: ignore[arg-type]
    tool.invoke({"text": "x"})
    record = fake.banks["n"][0]
    assert record["source"] == "tool-src"
    assert record["tags"] == "a,b"


def test_remember_tool_without_id(fake: FakeInfoLang, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(fake, "remember", lambda *a, **k: RememberResult())
    tool = create_remember_tool(fake, namespace="n")  # type: ignore[arg-type]
    assert tool.invoke({"text": "x"}) == "Stored memory."


async def test_remember_tool_async(fake: FakeInfoLang) -> None:
    tool = create_remember_tool(fake, namespace="n")  # type: ignore[arg-type]
    out = await tool.ainvoke({"text": "async store"})
    assert out.startswith("Stored memory")
    assert fake.banks["n"][0]["text"] == "async store"


def test_remember_tool_custom_name(fake: FakeInfoLang) -> None:
    tool = create_remember_tool(fake, name="save_fact")  # type: ignore[arg-type]
    assert tool.name == "save_fact"
