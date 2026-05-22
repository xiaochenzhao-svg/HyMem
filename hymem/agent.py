"""
HyMem agent: a thin, conversation-friendly wrapper around HybridMemorySystem.

Per-turn lifecycle (this is what makes user-facing latency low):

  1. Push the user message into the short-term FIFO queue. The queue is the
     SOURCE OF SHORT-TERM CONTEXT injected into every prompt; it is also the
     SOURCE of long-term archives once it crosses the token budget.
  2. Run dynamic_retrieval (light path; deep path only when necessary). The
     LLM gets both the recent dialogue (short-term) and the retrieved long-term
     notes; it answers as if it just naturally remembers the user.
  3. Push the assistant reply into the same short-term queue.
  4. If pushing those two turns made the un-archived suffix exceed the token
     budget, the queue automatically cuts a chunk and submits it to the
     background ARCHIVE EXECUTOR. EX_SUMMARY runs there, not on the hot path.

This means an ordinary turn costs ~1 LLM call (light) or ~2 (deep). EX_SUMMARY
is only paid once every ~max_session_tokens worth of dialogue.
"""

from typing import Optional, Tuple, Dict, Any, List

from hymem.core.memory_system import HybridMemorySystem
from hymem.config.settings import Settings


class HyMemAgent:
    """User-facing conversational agent backed by a hybrid memory system."""

    def __init__(
        self,
        embed_model: str,
        model_name: str,
        embed_api_key: str,
        api_key: str,
        embed_base_url: str,
        base_url: str,
        backend: str = "openai",
        retrieve_k: int = 10,
        retrieve_k_rough: int = 30,
        temperature: float = 0.7,
        max_session_tokens: int = 4096,
        async_indexing: bool = True,
        enable_self_check: bool = False,
        dedup_enabled: bool = True,
        dedup_top_k: int = 5,
        dedup_sim_threshold: float = 0.82,
        short_term_turns: int = 8,
    ):
        self.memory_system = HybridMemorySystem(
            llm_backend=backend,
            embed_llm_model=embed_model,
            embed_api_key=embed_api_key,
            embed_base_url=embed_base_url,
            llm_model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_session_tokens=max_session_tokens,
            enable_self_check=enable_self_check,
            dedup_enabled=dedup_enabled,
            dedup_top_k=dedup_top_k,
            dedup_sim_threshold=dedup_sim_threshold,
            short_term_turns=short_term_turns,
        )
        self.retrieve_k = retrieve_k
        self.retrieve_k_rough = retrieve_k_rough
        self.async_indexing = async_indexing
        self.current_session_id: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Constructors
    # ------------------------------------------------------------------ #

    @classmethod
    def from_settings(cls, settings: Settings) -> "HyMemAgent":
        return cls(
            embed_model=settings.embedding.model_name,
            model_name=settings.llm.model_name,
            embed_api_key=settings.embedding.api_key,
            api_key=settings.llm.api_key,
            embed_base_url=settings.embedding.base_url,
            base_url=settings.llm.base_url,
            backend=settings.backend,
            retrieve_k=settings.retrieval.retrieve_k,
            retrieve_k_rough=settings.retrieval.retrieve_k_rough,
            temperature=settings.llm.temperature,
            max_session_tokens=settings.session.max_session_tokens,
            async_indexing=settings.session.async_indexing,
            enable_self_check=settings.retrieval.enable_self_check,
            dedup_enabled=settings.dedup.enabled,
            dedup_top_k=settings.dedup.top_k,
            dedup_sim_threshold=settings.dedup.sim_threshold,
            short_term_turns=settings.retrieval.short_term_turns,
        )

    # ------------------------------------------------------------------ #
    # Conversation API
    # ------------------------------------------------------------------ #

    def start_conversation(self, session_id: Optional[str] = None) -> str:
        self.current_session_id = self.memory_system.create_session(session_id)
        return self.current_session_id

    def end_conversation(self) -> Optional[str]:
        """Archive the un-archived tail of the current session, if any."""
        if self.current_session_id is None:
            return None
        sid = self.current_session_id
        memory_id = self.memory_system.archive_session(sid)
        self.current_session_id = None
        return memory_id

    def chat(self, user_message: str) -> Tuple[str, str]:
        """Send a user message, get the assistant reply.

        Returns (reply, retrieved_long_term_context).
        """
        if not user_message or not user_message.strip():
            return "", ""

        if self.current_session_id is None:
            self.start_conversation()

        # 1. Push user turn (may trigger background archive if the queue grew
        #    past max_session_tokens).
        self.memory_system.push_turn("user", user_message, self.current_session_id)

        # 2. Run dynamic retrieval. Short-term context is read from the queue
        #    inside memory_system; long-term comes from the vector index.
        answer, context = self.memory_system.dynamic_retrieval(
            user_message,
            k=self.retrieve_k,
            k_rough=self.retrieve_k_rough,
            session_filter=None,
        )

        # 3. Push assistant turn into the short-term queue (same archive logic).
        self.memory_system.push_turn("assistant", answer, self.current_session_id)

        return answer, context

    # Backward-compatible alias
    def get_response(self, question: str, **_) -> Tuple[str, str]:
        return self.chat(question)

    def add_memory(
        self,
        content: str,
        time: Optional[str] = None,
        precomputed_summary: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        memory_type: str = "detailed",
        **kwargs,
    ) -> Optional[str]:
        """Manually inject a memory (e.g. user profile, imported data)."""
        return self.memory_system.add_note(
            content=content,
            time=time,
            precomputed_summary=precomputed_summary,
            session_id=session_id,
            memory_type=memory_type,
            **kwargs,
        )

    # ------------------------------------------------------------------ #
    # Maintenance
    # ------------------------------------------------------------------ #

    def wait_for_indexing(self, timeout: Optional[float] = None) -> None:
        """Block until queued archive tasks finish (best-effort)."""
        # Single-thread executor: submit a no-op and wait.
        try:
            fut = self.memory_system._executor.submit(lambda: None)
            fut.result(timeout=timeout)
        except Exception:
            pass

    def clear_memories(self) -> None:
        self.memory_system.clear_memories()
        self.current_session_id = None

    def get_memory_stats(self) -> Dict[str, Any]:
        return {
            "total_memories": len(self.memory_system.memories),
            "total_summaries": len(self.memory_system.summary_list),
            "active_sessions": len(self.memory_system.sessions),
            "current_session": self.current_session_id,
            "indexed_documents": len(self.memory_system.retriever),
        }

    def save_memories(self, directory: str) -> None:
        self.wait_for_indexing()
        self.memory_system.save_memories(directory)

    def load_memories(self, directory: str) -> None:
        self.memory_system.load_memories(directory)
        if self.memory_system.sessions:
            self.current_session_id = next(reversed(self.memory_system.sessions))

    def __repr__(self) -> str:
        s = self.get_memory_stats()
        return (
            f"HyMemAgent(memories={s['total_memories']}, "
            f"summaries={s['total_summaries']}, "
            f"sessions={s['active_sessions']}, "
            f"k={self.retrieve_k}, k_rough={self.retrieve_k_rough})"
        )


# Backward-compat aliases so older imports keep working.
EnhancedHybridMemAgent = HyMemAgent
HybridMemAgent = HyMemAgent
