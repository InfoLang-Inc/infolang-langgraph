"""End-to-end example: a LangGraph app that recalls a fact from a prior run.

Uses the prebuilt ``recall`` and ``retain`` nodes so memory is automatic (no
tool calls, no LLM required to see the effect). The "respond" node here just
echoes the recalled context so you can watch memory flow across separate
``invoke`` calls; swap it for a real LLM call in your app.

Run it against production InfoLang::

    INFOLANG_API_KEY=il_live_... python examples/quickstart.py

It only writes to a throwaway namespace and clears it on exit.
"""

from __future__ import annotations

import os
import uuid
from typing import Annotated, TypedDict

from infolang import InfoLang
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from infolang_langgraph import create_recall_node, create_retain_node


class State(TypedDict):
    messages: Annotated[list, add_messages]
    recalled_memories: str


def main() -> None:
    if not os.environ.get("INFOLANG_API_KEY"):
        raise SystemExit("Set INFOLANG_API_KEY to run this example.")

    namespace = f"example-langgraph-{uuid.uuid4().hex[:8]}"
    client = InfoLang()  # reads INFOLANG_API_KEY

    recall = create_recall_node(client, namespace=namespace)
    retain = create_retain_node(client, namespace=namespace)

    def respond(state: State) -> dict:
        context = state.get("recalled_memories", "")
        note = f"(recalled: {context})" if context else "(no memories yet)"
        return {"messages": [("ai", f"Got it. {note}")]}

    graph = StateGraph(State)
    graph.add_node("recall", recall)
    graph.add_node("respond", respond)
    graph.add_node("retain", retain)
    graph.add_edge(START, "recall")
    graph.add_edge("recall", "respond")
    graph.add_edge("respond", "retain")
    graph.add_edge("retain", END)
    app = graph.compile()

    try:
        print("Run 1 — teach a fact:")
        first = app.invoke({"messages": [("user", "My name is Ada and I love sailing.")]})
        print(" ", first["messages"][-1].content)

        print("Run 2 — recall it in a fresh run:")
        second = app.invoke({"messages": [("user", "What hobby did I mention?")]})
        print(" ", second["messages"][-1].content)
    finally:
        # Clean up the throwaway namespace's memories.
        for record in client.list_recent(namespace=f"lg.{namespace}", n=1000):
            memory_id = record.get("id")
            if memory_id:
                client.forget(memory_id, namespace=f"lg.{namespace}")
        client.close()


if __name__ == "__main__":
    main()
