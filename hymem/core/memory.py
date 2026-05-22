"""
Memory data structures for HyMem.

Two key abstractions:
  * ConversationSession
      - A FIFO queue of recent dialogue turns used as SHORT-TERM CONTEXT.
      - Token-budgeted: when the queue grows past `max_tokens`, the oldest turns
        are evicted, but only AFTER they have been archived as a long-term
        (second-tier) memory chunk. We track an "archive cursor" so we never
        evict turns that haven't been archived yet.
      - The split point for archival is sentence-level (turn-level), the
        eviction is also sentence-level. The two together implement the
        "FIFO at sentence granularity, archive at token granularity" idea.

  * MemoryNote
      - A second-tier memory: full original text of a chunk of conversation
        (or an externally injected fact). One MemoryNote ↔ many MemorySummary.

  * MemorySummary
      - A first-tier memory: an atomic fact extracted from a MemoryNote.
      - `links` is now a LIST of MemoryNote ids (the result of dedup-merge:
        a merged summary may inherit links from several MemoryNotes).
"""

from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
import uuid

from hymem.utils.helpers import cal_token


class ConversationSession:
    """
    Token-budgeted FIFO queue of recent dialogue turns.

    Roles:
      - "short-term memory" injected into every prompt as recent context
      - source of "second-tier" archives once the buffer exceeds the budget
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        max_tokens: int = 4096,
    ):
        self.session_id = session_id or str(uuid.uuid4())
        # Each turn: {"role": "user"|"assistant", "content": str, "tokens": int}
        self.turns: List[Dict[str, Any]] = []
        # Cursor: turns[archive_cursor:] are the un-archived, in-flight section.
        # turns[:archive_cursor] are already archived but kept as context until evicted.
        self.archive_cursor: int = 0
        self.created_at = datetime.now().isoformat()
        self.max_tokens = max_tokens

    # ------------------------------------------------------------------ #
    # Bookkeeping helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _count(text: str) -> int:
        try:
            return cal_token(text or "")
        except Exception:
            # Fallback heuristic if tiktoken is unavailable for some reason.
            return max(1, len(text or "") // 3)

    @property
    def total_tokens(self) -> int:
        return sum(t["tokens"] for t in self.turns)

    @property
    def unarchived_tokens(self) -> int:
        return sum(t["tokens"] for t in self.turns[self.archive_cursor :])

    def is_empty(self) -> bool:
        return not self.turns

    # ------------------------------------------------------------------ #
    # Append
    # ------------------------------------------------------------------ #

    def add_turn(self, role: str, content: str) -> Dict[str, Any]:
        """Always succeeds. Returns the appended turn dict."""
        turn = {"role": role, "content": content, "tokens": self._count(content)}
        self.turns.append(turn)
        return turn

    # Backwards-compat wrapper (older code calls add_message and expects bool)
    def add_message(self, role: str, content: str) -> bool:
        self.add_turn(role, content)
        return True

    # ------------------------------------------------------------------ #
    # Archive / evict
    # ------------------------------------------------------------------ #

    def needs_archive(self) -> bool:
        """True iff the un-archived suffix has reached the token budget."""
        return self.unarchived_tokens >= self.max_tokens

    def take_archive_chunk(self) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Cut a chunk to archive: starting at archive_cursor, take whole turns
        until adding the next would exceed max_tokens. Advance archive_cursor.
        Returns (chunk_text, chunk_turns). May return ("", []) if nothing to cut.
        """
        if self.archive_cursor >= len(self.turns):
            return "", []

        budget = self.max_tokens
        running = 0
        end = self.archive_cursor
        for i in range(self.archive_cursor, len(self.turns)):
            t = self.turns[i]
            if running + t["tokens"] > budget and end > self.archive_cursor:
                break
            running += t["tokens"]
            end = i + 1

        chunk_turns = self.turns[self.archive_cursor : end]
        self.archive_cursor = end
        return self._render_turns(chunk_turns), chunk_turns

    def take_remaining_chunk(self) -> Tuple[str, List[Dict[str, Any]]]:
        """Take everything still un-archived (used at end-of-session)."""
        if self.archive_cursor >= len(self.turns):
            return "", []
        chunk_turns = self.turns[self.archive_cursor :]
        self.archive_cursor = len(self.turns)
        return self._render_turns(chunk_turns), chunk_turns

    def evict_archived_until_fits(self) -> None:
        """
        After archiving, the un-archived suffix shrunk; but the *queue* may still
        be over budget because we keep already-archived turns as recent context.
        Drop archived turns from the front until total_tokens fits the budget.
        Never evict un-archived turns.
        """
        while self.archive_cursor > 0 and self.total_tokens > self.max_tokens:
            self.turns.pop(0)
            self.archive_cursor -= 1

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #

    @staticmethod
    def _render_turns(turns: List[Dict[str, Any]]) -> str:
        lines = []
        for t in turns:
            role = "User" if t["role"] == "user" else "Assistant"
            lines.append(f"{role}: {t['content']}")
        return "\n".join(lines)

    def render_recent(self, max_turns: int = 8) -> str:
        """Render the last `max_turns` turns as short-term context for the prompt."""
        recent = self.turns[-max_turns:] if max_turns > 0 else []
        return self._render_turns(recent)

    def get_session_content(self) -> str:
        """Render the entire session as a single string."""
        return self._render_turns(self.turns)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turns": self.turns,
            "archive_cursor": self.archive_cursor,
            "created_at": self.created_at,
            "max_tokens": self.max_tokens,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationSession":
        s = cls(session_id=data.get("session_id"), max_tokens=data.get("max_tokens", 4096))
        # Old-format compat: it used "messages" without per-turn token cache.
        if "turns" in data:
            s.turns = data["turns"]
        elif "messages" in data:
            s.turns = [
                {"role": m.get("role", "user"), "content": m.get("content", ""),
                 "tokens": cls._count(m.get("content", ""))}
                for m in data["messages"]
            ]
        s.archive_cursor = data.get("archive_cursor", len(s.turns))
        s.created_at = data.get("created_at", datetime.now().isoformat())
        return s


class MemoryNote:
    """Second-tier memory unit: full original text of a chunk of content."""

    def __init__(
        self,
        content: str,
        id: Optional[str] = None,
        links: Optional[List] = None,
        importance_score: Optional[float] = None,
        retrieval_count: Optional[int] = None,
        timestamp: Optional[str] = None,
        last_accessed: Optional[str] = None,
        context: Optional[str] = None,
        evolution_history: Optional[List] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        memory_type: str = "detailed",
        **kwargs,
    ):
        self.content = content
        self.id = id or str(uuid.uuid4())
        self.links = links or []
        self.importance_score = importance_score if importance_score is not None else 1.0
        self.retrieval_count = retrieval_count or 0

        current_time = datetime.now().strftime("%Y%m%d%H%M")
        self.timestamp = timestamp or current_time
        self.last_accessed = last_accessed or current_time
        self.context = context
        self.evolution_history = evolution_history or []
        self.category = category or "Uncategorized"
        self.tags = tags or []
        self.session_id = session_id
        self.memory_type = memory_type

        self._extra_attrs = kwargs

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        return self._extra_attrs.get(name)

    def update_access_time(self) -> None:
        self.last_accessed = datetime.now().strftime("%Y%m%d%H%M")
        self.retrieval_count += 1

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "links": self.links,
            "importance_score": self.importance_score,
            "retrieval_count": self.retrieval_count,
            "timestamp": self.timestamp,
            "last_accessed": self.last_accessed,
            "context": self.context,
            "evolution_history": self.evolution_history,
            "category": self.category,
            "tags": self.tags,
            "session_id": self.session_id,
            "memory_type": self.memory_type,
            **self._extra_attrs,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryNote":
        return cls(**data)

    def __repr__(self) -> str:
        return f"MemoryNote(id={self.id[:8]}, type={self.memory_type}, cat={self.category})"


class MemorySummary:
    """First-tier memory unit: atomic fact, vector-indexed.

    `links` is a list of MemoryNote ids. Normally one entry, but after a
    dedup-merge a summary can inherit links from several notes.
    """

    def __init__(
        self,
        content: str,
        links: Optional[List[str]] = None,
        link: Optional[str] = None,  # backwards-compat: old single-id form
        timestamp: Optional[str] = None,
        session_id: Optional[str] = None,
        id: Optional[str] = None,
    ):
        self.id = id or str(uuid.uuid4())
        self.content = content
        if links is None:
            links = [link] if link else []
        # Dedup links while preserving order.
        seen = set()
        self.links: List[str] = []
        for l in links:
            if l and l not in seen:
                seen.add(l)
                self.links.append(l)
        self.timestamp = timestamp
        self.session_id = session_id

    # Backwards-compat property: some old code reads `.link`.
    @property
    def link(self) -> Optional[str]:
        return self.links[0] if self.links else None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "links": self.links,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MemorySummary":
        return cls(
            id=data.get("id"),
            content=data["content"],
            links=data.get("links"),
            link=data.get("link"),
            timestamp=data.get("timestamp"),
            session_id=data.get("session_id"),
        )

    def __repr__(self) -> str:
        return f"MemorySummary(links={len(self.links)}, session={self.session_id})"
