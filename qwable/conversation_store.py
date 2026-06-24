"""G14-2: Persistent cross-request conversation store.

File-based JSON store for fusion deliberation session history.
Each conversation is a JSON file under `~/.qwable/conversations/<id>.json`.

Schema:
    {
        "id": "conv-abc123",
        "created_at": 1781804262.123,
        "updated_at": 1781804362.456,
        "messages": [
            {"role": "user", "content": "...", "preset": "quality"},
            {"role": "assistant", "content": "...", "preset": "quality"},
        ],
        "metadata": {"user": "alice", "topic": "..."}
    }

TTL: 24 hours (configurable). Expired conversations are filtered on read.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger("qwable.conversation_store")


def _default_store_dir() -> Path:
    """Default storage directory: ~/.qwable/conversations/."""
    return Path.home() / ".qwable" / "conversations"


@dataclass
class ConversationMessage:
    role: str  # "user" | "assistant" | "system"
    content: str
    preset: str = "quality"
    timestamp: float = field(default_factory=time.time)


@dataclass
class Conversation:
    id: str
    created_at: float
    updated_at: float
    messages: list[ConversationMessage] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "messages": [asdict(m) for m in self.messages],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Conversation":
        return cls(
            id=data["id"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            messages=[ConversationMessage(**m) for m in data.get("messages", [])],
            metadata=data.get("metadata", {}),
        )


class ConversationStore:
    """File-based conversation store with TTL-based expiration.

    Thread-safety: uses file locks via atomic rename.
    Concurrency: multiple processes can read; writes are atomic.
    """

    def __init__(
        self,
        store_dir: Optional[Path] = None,
        ttl_seconds: int = 24 * 3600,  # 24 hours
    ):
        self.store_dir = store_dir or _default_store_dir()
        self.ttl_seconds = ttl_seconds
        self.store_dir.mkdir(parents=True, exist_ok=True)
        # Serialize read-modify-write so concurrent appends don't lose data.
        self._lock = threading.Lock()

    def _path(self, conv_id: str) -> Path:
        return self.store_dir / f"{conv_id}.json"

    def create(self, metadata: Optional[dict] = None) -> Conversation:
        """Create a new empty conversation."""
        conv = Conversation(
            id=f"conv-{uuid.uuid4().hex[:12]}",
            created_at=time.time(),
            updated_at=time.time(),
            metadata=metadata or {},
        )
        self._save(conv)
        return conv

    def get(self, conv_id: str) -> Optional[Conversation]:
        """Retrieve a conversation by id, or None if not found / expired."""
        path = self._path(conv_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            conv = Conversation.from_dict(data)
            # Check TTL
            if (time.time() - conv.updated_at) > self.ttl_seconds:
                logger.info("conversation %s expired (TTL %ds), deleting", conv_id, self.ttl_seconds)
                self.delete(conv_id)
                return None
            return conv
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("failed to read conversation %s: %s", conv_id, exc)
            return None

    def append(self, conv_id: str, message: ConversationMessage) -> Optional[Conversation]:
        """Append a message to an existing conversation. Returns updated conv or None."""
        # Guard the read-modify-write so concurrent appends don't clobber each other.
        with self._lock:
            conv = self.get(conv_id)
            if conv is None:
                return None
            conv.messages.append(message)
            conv.updated_at = time.time()
            self._save(conv)
            return conv

    def delete(self, conv_id: str) -> bool:
        """Delete a conversation. Returns True if existed, False if not."""
        path = self._path(conv_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def list_all(self) -> list[Conversation]:
        """List all non-expired conversations (most recent first)."""
        results: list[Conversation] = []
        for path in self.store_dir.glob("conv-*.json"):
            conv_id = path.stem
            conv = self.get(conv_id)
            if conv is not None:
                results.append(conv)
        results.sort(key=lambda c: c.updated_at, reverse=True)
        return results

    def _save(self, conv: Conversation) -> None:
        """Atomic save: write to a UNIQUE temp file, fsync, then os.replace.

        A per-write unique temp name avoids two concurrent writers sharing one
        fixed `.json.tmp` (which lost data and could crash with FileNotFoundError
        when one writer's rename raced the other's).
        """
        path = self._path(conv.id)
        tmp_path = path.with_name(f"{conv.id}.{uuid.uuid4().hex}.tmp")
        try:
            with tmp_path.open("w", encoding="utf-8") as f:
                json.dump(conv.to_dict(), f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)  # atomic on POSIX
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

    def cleanup_expired(self) -> int:
        """Delete all expired conversations. Returns count deleted."""
        count = 0
        for path in list(self.store_dir.glob("conv-*.json")):
            conv_id = path.stem
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                updated_at = data.get("updated_at", 0)
                if (time.time() - updated_at) > self.ttl_seconds:
                    path.unlink()
                    count += 1
            except Exception:
                continue
        if count:
            logger.info("cleaned up %d expired conversations", count)
        return count


def build_history_block(conv: Conversation, max_messages: int = 10) -> str:
    """Build a text block summarizing conversation history for the synthesis prompt.

    Returns the most recent `max_messages` exchanges formatted as:
        Prior conversation context (N exchanges):
        [user]: ...
        [assistant]: ...
        ...

    If the conversation has no messages, returns an empty string.
    """
    if not conv.messages:
        return ""
    recent = conv.messages[-max_messages:]
    lines = [f"Prior conversation context ({len(recent)} exchanges):"]
    for msg in recent:
        role = msg.role
        content = msg.content[:500]  # truncate long messages
        if len(msg.content) > 500:
            content += "...[truncated]"
        lines.append(f"[{role}]: {content}")
    return "\n".join(lines)
