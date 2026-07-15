# infolang-langgraph

InfoLang semantic memory for [LangGraph](https://langchain-ai.github.io/langgraph/)
and [LangChain](https://python.langchain.com/): a `BaseStore` implementation,
memory tools, and prebuilt graph nodes, all backed by the
[InfoLang](https://infolang.ai) memory API.

Three ways to give an agent long-term memory, from most to least automatic:

| API | What it is | When to use |
|-----|------------|-------------|
| `InfoLangStore` | A LangGraph `BaseStore` (`put`/`get`/`search`/`delete`) | Wire memory into `create_react_agent(store=...)`, the functional API, or any graph that reads/writes the store |
| `create_recall_node` / `create_retain_node` | Prebuilt graph nodes | Automatic recall-before / retain-after a turn, no tool calls |
| `create_recall_tool` / `create_remember_tool` | LangChain tools | Let the LLM decide when to read/write memory |

## Install

```bash
pip install infolang-langgraph
```

Requires Python 3.11+, `langgraph>=1.0`, `langchain-core>=1.0`, and an InfoLang
API key. It depends only on the **published** `infolang` Python SDK
(`>=0.2,<0.3`) — no engine internals.

## Quickstart: `InfoLangStore`

```python
from infolang import InfoLang
from infolang_langgraph import InfoLangStore

# Reuse one InfoLang client for the store's lifetime.
client = InfoLang(api_key="il_live_...")          # or set INFOLANG_API_KEY
store = InfoLangStore(client=client)

# LangGraph namespaces are tuples; values are JSON-able dicts.
store.put(("memories", "user-123"), "pref-1", {"text": "Ada prefers dark mode"})

# Exact lookup by key.
item = store.get(("memories", "user-123"), "pref-1")
print(item.value["text"])                          # -> "Ada prefers dark mode"

# Semantic search within a namespace.
hits = store.search(("memories", "user-123"), query="what UI theme does the user like?")
print(hits[0].value["text"], hits[0].score)

store.delete(("memories", "user-123"), "pref-1")
```

`InfoLangStore()` also accepts the same construction kwargs as the `infolang`
SDK when you don't pass `client=` (they are forwarded verbatim):

```python
store = InfoLangStore(api_key="il_live_...", workspace="acme")
# store.close() when done — this only closes clients the store constructed itself.
```

## Wire it into a prebuilt agent

`InfoLangStore` is a drop-in `BaseStore`, so it works anywhere LangGraph accepts
a store:

```python
from langgraph.prebuilt import create_react_agent
from infolang_langgraph import InfoLangStore

agent = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=[...],
    store=InfoLangStore(api_key="il_live_..."),
)
```

## Prebuilt nodes: automatic memory

`create_recall_node` injects relevant memories before the model runs;
`create_retain_node` persists the turn afterward. This is how an agent
"demonstrably recalls a fact learned in a previous run":

```python
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict
from infolang import InfoLang
from infolang_langgraph import create_recall_node, create_retain_node

client = InfoLang(api_key="il_live_...")
NS = "demo-user-1"

class State(TypedDict):
    messages: Annotated[list, add_messages]
    recalled_memories: str

recall = create_recall_node(client, namespace=NS)     # writes state["recalled_memories"]
retain = create_retain_node(client, namespace=NS)     # persists the latest human + AI turn

def respond(state: State) -> dict:
    context = state.get("recalled_memories", "")
    # ... call your LLM here, passing `context` into the prompt ...
    return {"messages": [("ai", f"(context seen: {context!r})")]}

graph = StateGraph(State)
graph.add_node("recall", recall)
graph.add_node("respond", respond)
graph.add_node("retain", retain)
graph.add_edge(START, "recall")
graph.add_edge("recall", "respond")
graph.add_edge("respond", "retain")
graph.add_edge("retain", END)
app = graph.compile()

# Run 1 teaches a fact; a later run recalls it.
app.invoke({"messages": [("user", "My name is Ada and I love sailing.")]})
app.invoke({"messages": [("user", "What hobby did I mention?")]})
```

Pass `inject_system_message=True` to `create_recall_node` to append a
`SystemMessage` with the recalled context to `state["messages"]` instead of
writing a separate state key (requires an append reducer such as
`add_messages`).

## Memory tools: let the LLM decide

```python
from infolang import InfoLang
from infolang_langgraph import create_recall_tool, create_remember_tool

client = InfoLang(api_key="il_live_...")
tools = [
    create_recall_tool(client, namespace="user-1", top_k=5),
    create_remember_tool(client, namespace="user-1"),
]
# Bind `tools` to your model / pass to create_react_agent as usual.
```

Both tools expose a sync and an async implementation, so they work in
`AgentExecutor` and in async LangGraph runs.

## How namespaces map

InfoLang memory banks are keyed by a single flat string; LangGraph namespaces
are tuples. `infolang-langgraph` joins the tuple with a stable separator and a
prefix:

```
("memories", "user-123")   ->   "lg.memories.user-123"
```

- The `namespace_prefix` (default `"lg"`) lets `list_namespaces` tell this
  store's banks apart from banks written by other InfoLang integrations sharing
  a workspace, and makes the mapping reversible.
- Each segment is sanitized to `[a-zA-Z0-9_-]` (the separator and other
  characters become `-`). Two tuples whose *sanitized* segments are identical
  therefore collide onto one bank; pass `namespace_for=` to override the mapping
  if that matters for your keys.
- **`search` is single-bank**: `search(namespace_prefix, ...)` queries the bank
  equal to `namespace_prefix` (InfoLang `recall` is single-bank), so query with
  the full namespace — the usual LangGraph memory pattern. It does not fan out
  across deeper sub-namespaces.

## Semantics & limitations

- **Overwrite**: InfoLang has no update-in-place, so `put` finds and `forget`s
  any prior memory carrying the same key before writing the new one, preserving
  the original `created_at`.
- **TTL / indexing**: InfoLang exposes no TTL or per-field index knobs, so
  `put(..., ttl=...)`, `PutOp.index`, and `refresh_ttl` are accepted and
  ignored (`supports_ttl` is `False`).
- **`filter`**: applied client-side, matching LangGraph's reference
  `InMemoryStore` (exact match plus `$eq`/`$ne`/`$gt`/`$gte`/`$lt`/`$lte`).
- **`list_namespaces`**: derived from the public `list_banks()` and filtered to
  this store's `namespace_prefix`; returns `[]` if the runtime does not expose
  banks or when a custom `namespace_for` is set.
- **Async**: `InfoLangStore` uses a sync `infolang.InfoLang` client; `abatch`
  and `aclose` run the sync path in a worker thread (`asyncio.to_thread`).

## Development

```bash
pip install -e ".[dev]"
ruff check .
mypy
pytest
```

Unit tests inject an in-memory fake of the InfoLang client (no network). An
optional live smoke test is gated behind the `live` marker and
`INFOLANG_API_KEY`, and is excluded from the default `pytest` run:

```bash
INFOLANG_API_KEY=il_live_... pytest -m live -v
```

## License

Apache-2.0
