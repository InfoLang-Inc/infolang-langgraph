from __future__ import annotations

from datetime import UTC, datetime

from infolang_langgraph import _envelope


def test_encode_decode_roundtrip() -> None:
    created = datetime(2026, 1, 1, tzinfo=UTC)
    updated = datetime(2026, 1, 2, tzinfo=UTC)
    text = _envelope.encode(
        namespace=("memories", "u1"),
        key="k1",
        value={"text": "Ada likes tea", "n": 3},
        created_at=created,
        updated_at=updated,
    )
    decoded = _envelope.decode(text)
    assert decoded is not None
    assert decoded.namespace == ("memories", "u1")
    assert decoded.key == "k1"
    assert decoded.value == {"text": "Ada likes tea", "n": 3}
    assert decoded.created_at == created
    assert decoded.updated_at == updated


def test_content_prefix_is_human_readable() -> None:
    text = _envelope.encode(
        namespace=("x",),
        key="k",
        value={"text": "hello world"},
        created_at=_envelope.utcnow(),
        updated_at=_envelope.utcnow(),
    )
    assert text.startswith("hello world\n")


def test_content_for_embedding_variants() -> None:
    assert _envelope.content_for_embedding({"a": "one", "b": "two"}) == "one\ntwo"
    assert _envelope.content_for_embedding({"n": 5}) == "5"
    # bool is skipped, nested + lists are walked
    assert _envelope.content_for_embedding({"flag": True, "items": ["x", "y"]}) == "x\ny"
    assert "z" in _envelope.content_for_embedding({"d": {"deep": "z"}})


def test_content_for_embedding_falls_back_to_json() -> None:
    # No usable text/scalar leaves -> compact JSON.
    out = _envelope.content_for_embedding({"flag": True})
    assert out == '{"flag":true}'


def test_decode_non_envelope_returns_none() -> None:
    assert _envelope.decode("just some memory text") is None


def test_decode_none_and_non_string() -> None:
    assert _envelope.decode(None) is None
    assert _envelope.decode(123) is None  # type: ignore[arg-type]


def test_decode_bad_base64_returns_none() -> None:
    assert _envelope.decode("content\n<<infolang-langgraph:v1>>!!!not-base64!!!") is None


def test_decode_non_dict_payload_returns_none() -> None:
    import base64
    import json

    blob = base64.b64encode(json.dumps([1, 2, 3]).encode()).decode()
    assert _envelope.decode(f"x\n<<infolang-langgraph:v1>>{blob}") is None


def test_decode_missing_fields_returns_none() -> None:
    import base64
    import json

    blob = base64.b64encode(json.dumps({"key": "k"}).encode()).decode()
    assert _envelope.decode(f"x\n<<infolang-langgraph:v1>>{blob}") is None


def test_decode_bad_namespace_segment_returns_none() -> None:
    import base64
    import json

    payload = {"key": "k", "value": {}, "ns": ["ok", 5]}
    blob = base64.b64encode(json.dumps(payload).encode()).decode()
    assert _envelope.decode(f"x\n<<infolang-langgraph:v1>>{blob}") is None


def test_decode_invalid_timestamp_falls_back() -> None:
    import base64
    import json

    payload = {
        "key": "k",
        "value": {"t": "hi"},
        "ns": ["a"],
        "created_at": "not-a-real-date",
        "updated_at": "also-bad",
    }
    blob = base64.b64encode(json.dumps(payload).encode()).decode()
    decoded = _envelope.decode(f"hi\n<<infolang-langgraph:v1>>{blob}")
    assert decoded is not None
    assert decoded.created_at.tzinfo is not None


def test_decode_missing_timestamps_defaults() -> None:
    import base64
    import json

    payload = {"key": "k", "value": {"t": "hi"}, "ns": ["a"]}
    blob = base64.b64encode(json.dumps(payload).encode()).decode()
    decoded = _envelope.decode(f"hi\n<<infolang-langgraph:v1>>{blob}")
    assert decoded is not None
    assert decoded.created_at.tzinfo is not None
    assert decoded.updated_at == decoded.created_at


def test_display_text_strips_envelope() -> None:
    text = _envelope.encode(
        namespace=("x",),
        key="k",
        value={"text": "visible content"},
        created_at=_envelope.utcnow(),
        updated_at=_envelope.utcnow(),
    )
    assert _envelope.display_text(text) == "visible content"


def test_display_text_passthrough_and_non_string() -> None:
    assert _envelope.display_text("plain memory") == "plain memory"
    assert _envelope.display_text(None) == ""  # type: ignore[arg-type]


def test_record_id_fallbacks() -> None:
    assert _envelope.record_id({"id": "a"}) == "a"
    assert _envelope.record_id({"memory_id": "b"}) == "b"
    assert _envelope.record_id({"i": "c"}) == "c"
    assert _envelope.record_id({"nope": 1}) is None
    assert _envelope.record_id("not a dict") is None


def test_record_text_fallbacks() -> None:
    assert _envelope.record_text({"text": "a"}) == "a"
    assert _envelope.record_text({"t": "b"}) == "b"
    assert _envelope.record_text({"content": "c"}) == "c"
    assert _envelope.record_text({"nope": 1}) is None
    assert _envelope.record_text(42) is None


def test_utcnow_is_timezone_aware() -> None:
    assert _envelope.utcnow().tzinfo is not None
