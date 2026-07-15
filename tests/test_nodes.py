from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from infolang_langgraph import create_recall_node, create_retain_node
from tests.conftest import FakeInfoLang


def test_recall_node_writes_context_key(fake: FakeInfoLang) -> None:
    fake.remember("Ada prefers dark mode", namespace="mem")
    node = create_recall_node(fake, namespace="mem")  # type: ignore[arg-type]
    update = node({"messages": [HumanMessage("what mode does Ada prefer")]})
    assert update["recalled_memories"] == "- Ada prefers dark mode"


def test_recall_node_system_message_mode(fake: FakeInfoLang) -> None:
    fake.remember("Ada prefers dark mode", namespace="mem")
    node = create_recall_node(fake, namespace="mem", inject_system_message=True)  # type: ignore[arg-type]
    update = node({"messages": [HumanMessage("mode preference")]})
    messages = update["messages"]
    assert isinstance(messages[0], SystemMessage)
    assert "Ada prefers dark mode" in messages[0].content


def test_recall_node_no_query_is_noop(fake: FakeInfoLang) -> None:
    node = create_recall_node(fake, namespace="mem")  # type: ignore[arg-type]
    assert node({"messages": []}) == {}
    assert node({"messages": [AIMessage("only assistant text")]}) == {}


def test_recall_node_system_mode_empty_is_noop(fake: FakeInfoLang) -> None:
    node = create_recall_node(fake, namespace="empty", inject_system_message=True)  # type: ignore[arg-type]
    assert node({"messages": [HumanMessage("nothing stored")]}) == {}


def test_recall_node_query_fn_override(fake: FakeInfoLang) -> None:
    fake.remember("special topic", namespace="mem")
    node = create_recall_node(  # type: ignore[arg-type]
        fake, namespace="mem", query_fn=lambda state: "special"
    )
    update = node({"anything": True})
    assert "special topic" in update["recalled_memories"]


def test_recall_node_min_score(fake: FakeInfoLang) -> None:
    fake.remember("apple banana", namespace="mem")
    node = create_recall_node(fake, namespace="mem", min_score=0.99)  # type: ignore[arg-type]
    update = node({"messages": [HumanMessage("apple orange kiwi")]})
    assert update["recalled_memories"] == ""


def test_recall_node_handles_not_found(fake: FakeInfoLang) -> None:
    from infolang.errors import NotFoundError

    fake.recall_error = NotFoundError("no bank", status=404)
    node = create_recall_node(fake, namespace="mem")  # type: ignore[arg-type]
    assert node({"messages": [HumanMessage("x")]}) == {"recalled_memories": ""}


def test_retain_node_persists_human_and_ai(fake: FakeInfoLang) -> None:
    node = create_retain_node(fake, namespace="mem")  # type: ignore[arg-type]
    state = {
        "messages": [
            HumanMessage("my name is Ada"),
            AIMessage("nice to meet you Ada"),
        ]
    }
    assert node(state) == {}
    records = fake.banks["mem"]
    assert len(records) == 2
    texts = {r["text"] for r in records}
    assert texts == {"my name is Ada", "nice to meet you Ada"}
    tags = [r["tags"] for r in records]
    assert ["role:human"] in tags
    assert ["role:ai"] in tags


def test_retain_node_merges_extra_tags(fake: FakeInfoLang) -> None:
    node = create_retain_node(fake, namespace="mem", tags="session:1")  # type: ignore[arg-type]
    node({"messages": [HumanMessage("hi")]})
    assert fake.banks["mem"][0]["tags"] == ["session:1", "role:human"]


def test_retain_node_content_fn(fake: FakeInfoLang) -> None:
    node = create_retain_node(  # type: ignore[arg-type]
        fake, namespace="mem", content_fn=lambda state: ["fact one", "  ", "fact two"]
    )
    node({})
    texts = [r["text"] for r in fake.banks["mem"]]
    assert texts == ["fact one", "fact two"]  # blank entry skipped


def test_retain_node_no_messages_is_noop(fake: FakeInfoLang) -> None:
    node = create_retain_node(fake, namespace="mem")  # type: ignore[arg-type]
    assert node({"messages": []}) == {}
    assert fake.remember_batch_calls == 0


def test_retain_node_accepts_dict_messages(fake: FakeInfoLang) -> None:
    node = create_retain_node(fake, namespace="mem")  # type: ignore[arg-type]
    node(
        {
            "messages": [
                {"role": "user", "content": "dict human"},
                {"role": "assistant", "content": "dict ai"},
            ]
        }
    )
    texts = {r["text"] for r in fake.banks["mem"]}
    assert texts == {"dict human", "dict ai"}


def test_retain_node_accepts_tuple_and_multimodal(fake: FakeInfoLang) -> None:
    node = create_retain_node(fake, namespace="mem")  # type: ignore[arg-type]
    multimodal: list[Any] = [
        ("human", "tuple human"),
        {"role": "assistant", "content": [{"type": "text", "text": "block ai"}, "tail"]},
    ]
    node({"messages": multimodal})
    texts = {r["text"] for r in fake.banks["mem"]}
    assert texts == {"tuple human", "block ai\ntail"}


def test_retain_node_ignores_unknown_message_shapes(fake: FakeInfoLang) -> None:
    node = create_retain_node(fake, namespace="mem")  # type: ignore[arg-type]
    # An int is not a recognizable message; treated as empty and skipped.
    assert node({"messages": [12345]}) == {}
    assert fake.remember_batch_calls == 0


def test_retain_node_list_tags(fake: FakeInfoLang) -> None:
    node = create_retain_node(fake, namespace="mem", tags=["a", "b"])  # type: ignore[arg-type]
    node({"messages": [HumanMessage("hi")]})
    assert fake.banks["mem"][0]["tags"] == ["a", "b", "role:human"]


def test_retain_node_bare_string_message_has_no_role(fake: FakeInfoLang) -> None:
    node = create_retain_node(fake, namespace="mem")  # type: ignore[arg-type]
    # A bare string message carries no role, so it is not stored as human/ai.
    assert node({"messages": ["just a floating string"]}) == {}


def test_retain_node_non_text_content_is_empty(fake: FakeInfoLang) -> None:
    node = create_retain_node(fake, namespace="mem")  # type: ignore[arg-type]
    state = {
        "messages": [
            {"role": "user", "content": 123},  # non-text content -> ""
            {"role": "assistant", "content": [7, {"type": "image", "url": "x"}, "kept"]},
        ]
    }
    node(state)
    # Only the assistant's usable text leaf survives.
    texts = {r["text"] for r in fake.banks["mem"]}
    assert texts == {"kept"}
