# infolang-langgraph — agent instructions

InfoLang semantic memory for **LangGraph / LangChain**. Package name:
`infolang-langgraph`, import path `infolang_langgraph`.

## Architecture

- `src/infolang_langgraph/store.py` — `InfoLangStore`, a LangGraph
  `BaseStore`. Implements the two abstract methods (`batch`/`abatch`); the
  public `get`/`put`/`search`/`delete`/`list_namespaces` are inherited and
  route through `batch`. Each `Op` is translated into `infolang` SDK calls
  (`remember`/`recall`/`forget`/`list_recent`/`list_banks`).
- `src/infolang_langgraph/_scoping.py` — namespace tuple ⇄ InfoLang bank string
  (stable `prefix + separator` join; reversible for `list_namespaces`).
- `src/infolang_langgraph/_envelope.py` — the stored-item wire format: a
  human-readable content prefix (for good semantic recall) plus a base64 JSON
  trailer carrying the exact key/namespace/value/timestamps.
- `src/infolang_langgraph/_matching.py` — client-side `filter` + namespace
  match logic mirroring LangGraph's reference `InMemoryStore`.
- `src/infolang_langgraph/tools.py` — `create_recall_tool` /
  `create_remember_tool` (LangChain `StructuredTool`s).
- `src/infolang_langgraph/nodes.py` — `create_recall_node` /
  `create_retain_node` (prebuilt graph nodes).

## Contract

Depends on two upstream contracts, both external to this repo:

- The **published** `infolang` Python SDK (`>=0.2,<0.3`) — public API only:
  `recall`, `remember`, `remember_batch`, `forget`, `list_recent`, `list_banks`.
  Never reimplement HTTP, never import runtime/engine internals.
- LangGraph's `BaseStore` (`langgraph.store.base`) and LangChain's tool /
  message types. `BaseStore`'s abstract surface has changed across releases —
  read the installed package (`batch`/`abatch` are the only abstract methods in
  the pinned `langgraph>=1.0,<2`), don't guess from memory.

## Rules

- `InfoLangStore` uses a **sync** `infolang.InfoLang` client. `abatch`/`aclose`
  run the sync path in a worker thread (`asyncio.to_thread`); don't add a second
  async client.
- Overwrite semantics: `put` must `forget` any prior memory with the same key
  before `remember`ing, and preserve the original `created_at`.
- Foreign / unparsable records in a bank must be skipped (`_envelope.decode`
  returns `None`), never crash a read.
- `search` is single-bank (prefix == namespace); document, don't fake, cross
  sub-namespace fan-out (InfoLang `recall` is single-bank).

## Commands

```bash
pip install -e ".[dev]"
ruff check .
mypy
pytest
```
