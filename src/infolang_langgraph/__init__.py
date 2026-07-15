"""InfoLang memory integration for LangGraph and LangChain.

Three ways to give a LangGraph/LangChain agent InfoLang long-term memory:

1. :class:`~infolang_langgraph.store.InfoLangStore` -- a LangGraph ``BaseStore``
   (``put``/``get``/``search``/``delete``) backed by InfoLang.
2. :func:`~infolang_langgraph.tools.create_recall_tool` /
   :func:`~infolang_langgraph.tools.create_remember_tool` -- LangChain tools a
   tool-calling agent invokes explicitly.
3. :func:`~infolang_langgraph.nodes.create_recall_node` /
   :func:`~infolang_langgraph.nodes.create_retain_node` -- prebuilt graph nodes
   for automatic memory without tool calls.

Quickstart::

    from infolang import InfoLang
    from infolang_langgraph import InfoLangStore

    store = InfoLangStore(api_key="il_live_...")
    store.put(("memories", "user-1"), "fact-1", {"text": "Ada prefers dark mode"})
    hits = store.search(("memories", "user-1"), query="ui preference")
    print(hits[0].value["text"])
"""

from __future__ import annotations

from ._scoping import (
    DEFAULT_NAMESPACE_PREFIX,
    DEFAULT_SEPARATOR,
    from_infolang_namespace,
    to_infolang_namespace,
)
from ._version import __version__
from .nodes import create_recall_node, create_retain_node
from .store import InfoLangStore
from .tools import create_recall_tool, create_remember_tool

__all__ = [
    "__version__",
    "InfoLangStore",
    "create_recall_tool",
    "create_remember_tool",
    "create_recall_node",
    "create_retain_node",
    "to_infolang_namespace",
    "from_infolang_namespace",
    "DEFAULT_NAMESPACE_PREFIX",
    "DEFAULT_SEPARATOR",
]
