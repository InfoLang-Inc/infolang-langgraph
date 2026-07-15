from __future__ import annotations

from infolang_langgraph import from_infolang_namespace, to_infolang_namespace


def test_basic_join() -> None:
    assert to_infolang_namespace(("memories", "user-123")) == "lg.memories.user-123"


def test_empty_tuple_maps_to_prefix() -> None:
    assert to_infolang_namespace(()) == "lg"


def test_empty_tuple_and_empty_prefix_maps_to_root() -> None:
    assert to_infolang_namespace((), prefix="") == "root"


def test_custom_prefix_and_separator() -> None:
    assert to_infolang_namespace(("a", "b"), prefix="p", separator="_") == "p_a_b"


def test_sanitizes_separator_and_unsafe_chars() -> None:
    # The separator and other unsafe chars collapse to "-".
    assert to_infolang_namespace(("a.b", "c/d")) == "lg.a-b.c-d"


def test_blank_segment_becomes_unknown() -> None:
    assert to_infolang_namespace(("", "x")) == "lg.unknown.x"


def test_roundtrip_default() -> None:
    ns = ("memories", "user-123")
    bank = to_infolang_namespace(ns)
    assert from_infolang_namespace(bank) == ns


def test_reverse_rejects_foreign_bank() -> None:
    assert from_infolang_namespace("oa-session-abc") is None


def test_reverse_prefix_only_is_empty_tuple() -> None:
    assert from_infolang_namespace("lg") == ()


def test_reverse_no_prefix_config() -> None:
    assert from_infolang_namespace("a.b.c", prefix="") == ("a", "b", "c")


def test_reverse_no_prefix_empty_string() -> None:
    assert from_infolang_namespace("", prefix="") == ()


def test_lossy_collision_is_documented_behavior() -> None:
    # Two distinct tuples whose sanitized forms match collapse to one bank.
    assert to_infolang_namespace(("a.b",)) == to_infolang_namespace(("a-b",))
