# Changelog

All notable changes to `infolang-langgraph` are documented here. This project
adheres to [Semantic Versioning](https://semver.org).

## [0.1.0] - 2026-07-15

### Added
- Initial release: InfoLang semantic memory for LangGraph / LangChain.
- `InfoLangStore` — a LangGraph `BaseStore` (`put`/`get`/`search`/`delete`/
  `list_namespaces`) backed by the public `infolang` SDK
  (`remember`/`recall`/`forget`/`list_recent`/`list_banks`). Implements the
  `batch`/`abatch` abstract methods; `abatch` runs the sync path in a worker
  thread.
- Namespace tuples map to InfoLang banks via a stable, reversible
  `prefix + separator` join (`infolang_langgraph.to_infolang_namespace`).
- Item envelope with a human-readable content prefix (for good semantic recall)
  plus a base64 JSON trailer for a lossless key/namespace/value/timestamp
  round-trip.
- `create_recall_tool` / `create_remember_tool` — LangChain tools for
  tool-calling agents (sync + async).
- `create_recall_node` / `create_retain_node` — prebuilt graph nodes for
  automatic memory without tool calls.
- Live smoke test (`tests/test_live_smoke.py`), gated on `INFOLANG_API_KEY` and
  the `live` marker; excluded from the default `pytest` run.
