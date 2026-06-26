"""G14-2: tests for ConversationStore."""

import time
from pathlib import Path


from qwable.conversation_store import (
    Conversation,
    ConversationMessage,
    ConversationStore,
    build_history_block,
)


def test_create_returns_conversation_with_unique_id(tmp_path: Path):
    store = ConversationStore(store_dir=tmp_path, ttl_seconds=3600)
    conv = store.create(metadata={"user": "test"})
    assert conv.id.startswith("conv-")
    assert conv.created_at > 0
    assert conv.updated_at > 0
    assert conv.messages == []
    assert conv.metadata == {"user": "test"}
    # File exists
    assert (tmp_path / f"{conv.id}.json").exists()


def test_get_returns_none_for_missing_id(tmp_path: Path):
    store = ConversationStore(store_dir=tmp_path, ttl_seconds=3600)
    assert store.get("conv-nonexistent") is None


def test_append_message_updates_timestamp(tmp_path: Path):
    store = ConversationStore(store_dir=tmp_path, ttl_seconds=3600)
    conv = store.create()
    original_updated = conv.updated_at
    time.sleep(0.01)  # ensure timestamp advances
    msg = ConversationMessage(role="user", content="hi", preset="budget")
    updated = store.append(conv.id, msg)
    assert updated is not None
    assert len(updated.messages) == 1
    assert updated.messages[0].content == "hi"
    assert updated.messages[0].preset == "budget"
    assert updated.updated_at > original_updated


def test_delete_removes_conversation(tmp_path: Path):
    store = ConversationStore(store_dir=tmp_path, ttl_seconds=3600)
    conv = store.create()
    assert store.delete(conv.id) is True
    assert store.get(conv.id) is None
    # Second delete is no-op
    assert store.delete(conv.id) is False


def test_get_returns_none_for_expired_conversation(tmp_path: Path):
    """TTL=1s, wait 2s, conversation should be expired."""
    store = ConversationStore(store_dir=tmp_path, ttl_seconds=1)
    conv = store.create()
    # First read should succeed
    assert store.get(conv.id) is not None
    time.sleep(1.5)
    # Second read should return None (expired)
    assert store.get(conv.id) is None
    # File should be deleted
    assert not (tmp_path / f"{conv.id}.json").exists()


def test_list_all_returns_recent_first(tmp_path: Path):
    store = ConversationStore(store_dir=tmp_path, ttl_seconds=3600)
    c1 = store.create()
    time.sleep(0.01)
    store.append(
        c1.id,
        ConversationMessage(role="user", content="x"),
    )
    time.sleep(0.01)
    c3 = store.create()
    convs = store.list_all()
    assert len(convs) == 2  # c1 and c3 (c2 is same as c1)
    assert convs[0].id == c3.id  # most recent first


def test_cleanup_expired_removes_only_expired(tmp_path: Path):
    # Use a long TTL so we can selectively expire one without losing both.
    store = ConversationStore(store_dir=tmp_path, ttl_seconds=3600)
    fresh = store.create()
    expired = store.create()
    # Backdate `expired`'s updated_at to past TTL
    expired_path = store._path(expired.id)
    import json as _json

    data = _json.loads(expired_path.read_text())
    data["updated_at"] -= 7200  # 2 hours ago
    expired_path.write_text(_json.dumps(data))

    removed = store.cleanup_expired()
    assert removed == 1
    # Fresh still exists
    assert store.get(fresh.id) is not None
    # Expired is gone
    assert store.get(expired.id) is None


def test_persistence_across_store_instances(tmp_path: Path):
    """Create with one instance, read with another."""
    s1 = ConversationStore(store_dir=tmp_path, ttl_seconds=3600)
    conv = s1.create()
    s1.append(conv.id, ConversationMessage(role="user", content="hello"))
    s2 = ConversationStore(store_dir=tmp_path, ttl_seconds=3600)
    loaded = s2.get(conv.id)
    assert loaded is not None
    assert len(loaded.messages) == 1
    assert loaded.messages[0].content == "hello"


def test_build_history_block_empty_conv():
    """Empty conversation returns empty string."""
    conv = Conversation(id="x", created_at=0, updated_at=0)
    assert build_history_block(conv) == ""


def test_build_history_block_formats_messages():
    """History block lists messages with role and truncated content."""
    conv = Conversation(
        id="x",
        created_at=0,
        updated_at=0,
        messages=[
            ConversationMessage(role="user", content="hi", preset="quality"),
            ConversationMessage(
                role="assistant", content="hello back", preset="quality"
            ),
        ],
    )
    block = build_history_block(conv)
    assert "Prior conversation context" in block
    assert "[user]: hi" in block
    assert "[assistant]: hello back" in block


def test_build_history_block_truncates_long_content():
    """Content > 500 chars is truncated with marker."""
    long_content = "x" * 1000
    conv = Conversation(
        id="x",
        created_at=0,
        updated_at=0,
        messages=[ConversationMessage(role="user", content=long_content)],
    )
    block = build_history_block(conv)
    assert "truncated" in block
    assert long_content not in block  # full content not present
    # Truncated portion should be 500 chars + marker
    assert "x" * 500 in block


def test_build_history_block_caps_at_max_messages():
    """Only the most recent N messages are included."""
    conv = Conversation(
        id="x",
        created_at=0,
        updated_at=0,
        messages=[
            ConversationMessage(role="user", content=f"msg{i}") for i in range(20)
        ],
    )
    block = build_history_block(conv, max_messages=3)
    # Should include msg17, 18, 19 only
    assert "msg17" in block
    assert "msg18" in block
    assert "msg19" in block
    assert "msg0" not in block
    assert "msg16" not in block
    # Counter in header
    assert "(3 exchanges)" in block
