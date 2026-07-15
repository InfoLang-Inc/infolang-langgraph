"""Client-side filter + namespace matching, mirroring LangGraph's semantics.

InfoLang's ``recall`` ranks by semantic similarity but does not apply LangGraph's
structured ``filter`` dict or ``list_namespaces`` match conditions. These helpers
reproduce the exact behavior of LangGraph's reference ``InMemoryStore`` (see
``langgraph.store.memory``) so ``InfoLangStore.search``/``list_namespaces`` behave
the same way callers expect from the built-in store.
"""

from __future__ import annotations

from typing import Any

from langgraph.store.base import MatchCondition


def apply_operator(value: Any, operator: str, op_value: Any) -> bool:
    """Applies one comparison operator, matching PostgreSQL JSONB behavior."""

    if operator == "$eq":
        return bool(value == op_value)
    if operator == "$ne":
        return bool(value != op_value)
    if operator == "$gt":
        return float(value) > float(op_value)
    if operator == "$gte":
        return float(value) >= float(op_value)
    if operator == "$lt":
        return float(value) < float(op_value)
    if operator == "$lte":
        return float(value) <= float(op_value)
    raise ValueError(f"Unsupported operator: {operator}")


def compare_values(item_value: Any, filter_value: Any) -> bool:
    """Compares a stored value against a filter value (nested + operators)."""

    if isinstance(filter_value, dict):
        if any(k.startswith("$") for k in filter_value):
            return all(
                apply_operator(item_value, op_key, op_value)
                for op_key, op_value in filter_value.items()
            )
        if not isinstance(item_value, dict):
            return False
        return all(compare_values(item_value.get(k), v) for k, v in filter_value.items())
    if isinstance(filter_value, (list, tuple)):
        return (
            isinstance(item_value, (list, tuple))
            and len(item_value) == len(filter_value)
            and all(
                compare_values(iv, fv)
                for iv, fv in zip(item_value, filter_value, strict=False)
            )
        )
    return bool(item_value == filter_value)


def matches_filter(value: dict[str, Any], filter: dict[str, Any] | None) -> bool:
    """Whether a value dict satisfies every entry of a LangGraph ``filter``."""

    if not filter:
        return True
    return all(compare_values(value.get(key), fv) for key, fv in filter.items())


def matches_condition(condition: MatchCondition, namespace: tuple[str, ...]) -> bool:
    """Whether a namespace tuple satisfies one ``list_namespaces`` condition."""

    path = condition.path
    if len(namespace) < len(path):
        return False
    if condition.match_type == "prefix":
        pairs = zip(namespace, path, strict=False)
    elif condition.match_type == "suffix":
        pairs = zip(reversed(namespace), reversed(path), strict=False)
    else:
        raise ValueError(f"Unsupported match type: {condition.match_type}")
    return all(p_elem == "*" or k_elem == p_elem for k_elem, p_elem in pairs)
