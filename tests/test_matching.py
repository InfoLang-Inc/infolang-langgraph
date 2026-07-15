from __future__ import annotations

import pytest
from langgraph.store.base import MatchCondition

from infolang_langgraph._matching import (
    apply_operator,
    compare_values,
    matches_condition,
    matches_filter,
)


def test_apply_operators() -> None:
    assert apply_operator(5, "$eq", 5)
    assert apply_operator(5, "$ne", 6)
    assert apply_operator(5, "$gt", 4)
    assert apply_operator(5, "$gte", 5)
    assert apply_operator(4, "$lt", 5)
    assert apply_operator(5, "$lte", 5)


def test_apply_operator_invalid_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported operator"):
        apply_operator(1, "$bogus", 1)


def test_compare_values_scalar_and_nested() -> None:
    assert compare_values("x", "x")
    assert not compare_values("x", "y")
    assert compare_values({"a": 1}, {"a": 1})
    assert not compare_values("notdict", {"a": 1})


def test_compare_values_operator_dict() -> None:
    assert compare_values(10, {"$gt": 5, "$lt": 20})
    assert not compare_values(10, {"$gt": 50})


def test_compare_values_lists() -> None:
    assert compare_values([1, 2], [1, 2])
    assert not compare_values([1, 2], [1, 3])
    assert not compare_values([1], [1, 2])
    assert not compare_values("x", [1])


def test_matches_filter_none_and_multi() -> None:
    assert matches_filter({"a": 1}, None)
    assert matches_filter({"a": 1, "b": 2}, {"a": 1, "b": 2})
    assert not matches_filter({"a": 1}, {"a": 2})


def test_matches_condition_prefix() -> None:
    cond = MatchCondition(match_type="prefix", path=("memories",))
    assert matches_condition(cond, ("memories", "u1"))
    assert not matches_condition(cond, ("other", "u1"))


def test_matches_condition_suffix() -> None:
    cond = MatchCondition(match_type="suffix", path=("u1",))
    assert matches_condition(cond, ("memories", "u1"))
    assert not matches_condition(cond, ("memories", "u2"))


def test_matches_condition_wildcard() -> None:
    cond = MatchCondition(match_type="prefix", path=("*", "u1"))
    assert matches_condition(cond, ("anything", "u1"))


def test_matches_condition_too_short_is_false() -> None:
    cond = MatchCondition(match_type="prefix", path=("a", "b", "c"))
    assert not matches_condition(cond, ("a",))


def test_matches_condition_invalid_type_raises() -> None:
    cond = MatchCondition(match_type="bogus", path=("a",))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unsupported match type"):
        matches_condition(cond, ("a",))
