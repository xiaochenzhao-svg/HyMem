"""
Hybrid memory system implementing the HyMem architecture.

Two-layer storage:
  * detailed memory (MemoryNote): the original text of a chunk of dialogue
                                   (or an externally injected fact). Kept
                                   verbatim for high-fidelity recall.
  * summary memory (MemorySummary): one or more atomic fact sentences
                                   extracted from a MemoryNote, used as the
                                   cheap-to-search vector index.

Two-layer retrieval (dynamic scheduling):
  * Light path: vector search on the summary index. The LLM is asked to either
                answer right away (finished=0/1) or escalate (finished=2).
  * Deep path:  expand the candidate pool, ask the LLM to pick which summary
                ids matter, gather their underlying detailed memories
                (deduplicated by note id since one note may have many summaries),
                and write the final answer from full text.

Short-term context:
  * The current ConversationSession is a token-budgeted FIFO queue of recent
    turns. The most recent N turns are always injected into the prompt as
    "short-term context" so multi-turn references work even when no archive
    has happened yet.

Archival:
  * When the un-archived suffix of the queue exceeds max_session_tokens, the
    suffix (cut on a turn boundary) becomes a new MemoryNote and is summarised
    into atomic facts. After indexing, those new facts are passed through a
    cross-LLM dedup-merge step so the index doesn't bloat with restatements.
"""

import os
import re
import logging
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any

from hymem.core.memory import MemoryNote, MemorySummary, ConversationSession
from hymem.core.retriever import EnhancedEmbeddingRetriever
from hymem.core.llm_controller import LLMController
from hymem.prompts.templates import PromptTemplates
from hymem.utils.helpers import parse_json_response

logger = logging.getLogger(__name__)


class HybridMemorySystem:
    """Hybrid (summary + detailed) memory with dynamic retrieval scheduling."""

    def __init__(
        self,
        llm_backend: str = "openai",
        embed_llm_model: str = "text-embedding-3-small",
        embed_api_key: Optional[str] = None,
        embed_base_url: Optional[str] = None,
        llm_model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_session_tokens: int = 4096,
        enable_self_check: bool = False,
        dedup_enabled: bool = True,
        dedup_top_k: int = 5,
        dedup_sim_threshold: float = 0.82,
        short_term_turns: int = 8,
    ):
        self.memories: Dict[str, MemoryNote] = {}
        # summary_list[i] holds the live MemorySummary at retriever index i.
        # (Both grow / shrink together; deletions go through both.)
        self.summary_list: Dict[int, MemorySummary] = {}

        self.retriever = EnhancedEmbeddingRetriever(
            embed_llm_model, embed_api_key, embed_base_url
        )
        self.llm_controller = LLMController(llm_backend, llm_model, api_key, base_url)

        self.temperature = temperature
        self.max_session_tokens = max_session_tokens
        self.enable_self_check = enable_self_check
        self.dedup_enabled = dedup_enabled
        self.dedup_top_k = dedup_top_k
        self.dedup_sim_threshold = dedup_sim_threshold
        self.short_term_turns = short_term_turns

        self.sessions: Dict[str, ConversationSession] = {}
        self.current_session_id: Optional[str] = None

        self._write_lock = threading.RLock()
        # Single-thread executor: archival writes are serialised, never bursty.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hymem-arch")

    # ================================================================== #
    # Session / short-term queue
    # ================================================================== #

    def create_session(self, session_id: Optional[str] = None) -> str:
        session = ConversationSession(session_id, self.max_session_tokens)
        self.sessions[session.session_id] = session
        self.current_session_id = session.session_id
        return session.session_id

    def push_turn(
        self,
        role: str,
        content: str,
        session_id: Optional[str] = None,
    ) -> List[str]:
        """
        Append a turn to the active session. May trigger one or more archival
        chunks if the un-archived tail crosses the token budget.

        Returns the list of newly created MemoryNote ids (usually 0 or 1).
        """
        if session_id is None:
            session_id = self.current_session_id
        if session_id is None or session_id not in self.sessions:
            session_id = self.create_session(session_id)
        session = self.sessions[session_id]
        session.add_turn(role, content)

        archived_ids: List[str] = []
        # The un-archived suffix may cross the budget more than once if a single
        # turn is huge. Keep cutting until it fits.
        while session.needs_archive():
            chunk_text, chunk_turns = session.take_archive_chunk()
            if not chunk_turns:
                break
            note_id = self._archive_chunk(chunk_text, session_id)
            if note_id:
                archived_ids.append(note_id)
        # Trim already-archived turns from the front so the queue stays bounded.
        session.evict_archived_until_fits()
        return archived_ids

    def archive_session(self, session_id: str) -> Optional[str]:
        """Archive any remaining un-archived tail (used at /new and /quit)."""
        if session_id not in self.sessions:
            return None
        session = self.sessions[session_id]
        chunk_text, chunk_turns = session.take_remaining_chunk()
        if not chunk_turns:
            return None
        return self._archive_chunk(chunk_text, session_id)

    def get_short_term_context(self, session_id: Optional[str] = None) -> str:
        sid = session_id or self.current_session_id
        if not sid or sid not in self.sessions:
            return ""
        return self.sessions[sid].render_recent(self.short_term_turns)

    # ================================================================== #
    # Memory ingestion
    # ================================================================== #

    def add_note(
        self,
        content: str,
        time: Optional[str] = None,
        precomputed_summary: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        memory_type: str = "detailed",
        category: Optional[str] = None,
        skip_dedup: bool = False,
        **kwargs,
    ) -> Optional[str]:
        """Ingest one chunk: 1 detailed note + N atomic-fact summaries."""
        if not content or not content.strip():
            return None
        if time is None:
            time = datetime.now().strftime("on %Y-%m-%d %H:%M:%S")

        if precomputed_summary is not None:
            atomic_facts = [s for s in precomputed_summary if s and s.strip()]
        else:
            atomic_facts = self._extract_atomic_facts(content)
            if not atomic_facts:
                # Nothing worth indexing (pure pleasantries) — drop the note;
                # the conversation buffer still holds it for short-term recall.
                logger.debug("add_note: nothing summarisable, skipping note creation")
                return None

        note = MemoryNote(
            content=content,
            timestamp=time,
            session_id=session_id,
            memory_type=memory_type,
            category=category or "Uncategorized",
            **kwargs,
        )

        new_summaries = [
            MemorySummary(
                content=fact,
                links=[note.id],
                timestamp=time,
                session_id=session_id,
            )
            for fact in atomic_facts
        ]

        with self._write_lock:
            self.memories[note.id] = note
            assigned_indices = self.retriever.add_documents(
                [s.content for s in new_summaries],
                [
                    {
                        "session_id": session_id,
                        "memory_type": memory_type,
                        "timestamp": time,
                        "summary_id": s.id,
                    }
                    for s in new_summaries
                ],
            )
            for idx, s in zip(assigned_indices, new_summaries):
                self.summary_list[idx] = s

        if self.dedup_enabled and not skip_dedup and assigned_indices:
            try:
                self._dedup_merge(assigned_indices)
            except Exception as e:  # never let dedup take down indexing
                logger.warning("dedup-merge failed: %s", e)

        return note.id

    def add_note_async(self, content: str, **kwargs):
        """Submit indexing to the single-thread archive executor."""
        return self._executor.submit(self._safe_add_note, content, kwargs)

    def _safe_add_note(self, content: str, kwargs: dict):
        try:
            return self.add_note(content, **kwargs)
        except Exception as e:
            logger.exception("Background add_note failed: %s", e)
            return None

    def _archive_chunk(self, chunk_text: str, session_id: str) -> Optional[str]:
        """Index a chunk produced by take_archive_chunk / take_remaining_chunk."""
        return self.add_note(
            content=chunk_text,
            session_id=session_id,
            memory_type="detailed",
            category="conversation_chunk",
        )

    # ================================================================== #
    # Dynamic retrieval (HyMem core)
    # ================================================================== #

    def dynamic_retrieval(
        self,
        question: str,
        k: int = 10,
        k_rough: int = 30,
        session_filter: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Answer `question` using the cheapest path that suffices.

        Returns: (answer, retrieved_context_text)
        """
        short_term = self.get_short_term_context()

        # Cold start: no long-term memory at all -> light path can still answer
        # casually using general knowledge + short-term context.
        if not self.memories:
            answer = self._light_answer(question, "", short_term)[1]
            return answer, ""

        # ---- Light path ----
        light_hits = self.retriever.search(question, k, session_filter)
        light_indices = [idx for idx, _ in light_hits]
        light_text = self._format_summaries(light_indices)
        tag, answer = self._light_answer(question, light_text, short_term)
        retrieved_context = light_text

        # ---- Deep path ----
        if tag == 2:
            wider_hits = self.retriever.search(question, k_rough, session_filter)
            wider_indices = [idx for idx, _ in wider_hits]

            # Single LLM call to pick which summaries are actually relevant
            # (this is the paper's "rough recall + LLM rerank" stage).
            picked_local = self._pick_relevant_summaries(question, wider_indices)
            picked = [wider_indices[i] for i in picked_local if 0 <= i < len(wider_indices)]

            # Fallback: if the selector picked nothing, use the top-5 vector hits
            # so the deep stage always has *some* grounded context.
            if not picked and wider_indices:
                picked = wider_indices[:5]

            note_ids = self._summary_indices_to_note_ids(picked)
            retrieved_context = self._notes_to_text(note_ids)
            answer = self._deep_answer(question, retrieved_context, short_term)

            # Last-resort empty-answer guard.
            if not answer or not answer.strip():
                _, answer = self._light_answer(question, retrieved_context or light_text, short_term)
                if not answer or not answer.strip():
                    answer = (
                        "I'm not sure I can answer that precisely yet — could you "
                        "give me a bit more detail?"
                    )

        # Optional self-check (off by default in real-time chat).
        if self.enable_self_check and answer:
            ok, new_q = self._analyze_answer(question, answer)
            if ok == 0 and new_q:
                wider_hits = self.retriever.search(new_q, k_rough, session_filter)
                wider_indices = [idx for idx, _ in wider_hits]
                picked_local = self._pick_relevant_summaries(new_q, wider_indices)
                picked = [wider_indices[i] for i in picked_local if 0 <= i < len(wider_indices)]
                if not picked and wider_indices:
                    picked = wider_indices[:5]
                note_ids = self._summary_indices_to_note_ids(picked)
                retrieved_context = self._notes_to_text(note_ids)
                answer = self._deep_answer(question, retrieved_context, short_term)

        return answer, retrieved_context

    # ================================================================== #
    # Internal helpers
    # ================================================================== #

    def _format_summaries(self, indices) -> str:
        parts = []
        for idx in indices:
            s = self.summary_list.get(idx)
            if s is not None:
                parts.append(f"- ({self._friendly_time(s.timestamp)}) {s.content}")
        return "\n".join(parts)

    def _format_indexed_summaries(self, indices) -> str:
        """For RETRIEVER prompt: prefix each candidate with a local id."""
        parts = []
        for local_id, idx in enumerate(indices):
            s = self.summary_list.get(idx)
            if s is not None:
                parts.append(
                    f"id:{local_id}, time:{self._friendly_time(s.timestamp)}, {s.content}"
                )
        return "\n".join(parts)

    def _summary_indices_to_note_ids(self, indices: List[int]) -> List[str]:
        """Resolve summary indices to deduplicated MemoryNote ids."""
        seen = set()
        out: List[str] = []
        for idx in indices:
            s = self.summary_list.get(idx)
            if s is None:
                continue
            for nid in s.links:
                if nid in self.memories and nid not in seen:
                    seen.add(nid)
                    out.append(nid)
        return out

    def _notes_to_text(self, note_ids: List[str]) -> str:
        parts = []
        for nid in note_ids:
            note = self.memories.get(nid)
            if note is not None:
                parts.append(note.content)
        return "\n---\n".join(parts)

    @staticmethod
    def _friendly_time(timestamp: Optional[str]) -> str:
        if not timestamp or not isinstance(timestamp, str):
            return "unknown"
        m = re.search(r"on\s+(.+)$", timestamp)
        return m.group(1) if m else timestamp

    # ---- LLM-backed primitives -------------------------------------- #

    def _extract_atomic_facts(self, content: str) -> List[str]:
        prompt = PromptTemplates.EX_SUMMARY + content
        response = self.llm_controller.get_completion(
            prompt,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "keywords": {"type": "array", "items": {"type": "string"}}
                        },
                        "required": ["keywords"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            },
            temperature=self.temperature,
        )
        data = parse_json_response(response)
        if not data:
            return []
        return [k for k in (data.get("keywords") or []) if isinstance(k, str) and k.strip()]

    def _light_answer(self, question: str, summary_text: str, short_term: str) -> Tuple[int, str]:
        prompt_parts = [PromptTemplates.ANSWER_LIGHT, "\nUser message: " + question]
        if short_term:
            prompt_parts.append("\nRecent dialogue (short-term context):\n" + short_term)
        prompt_parts.append(
            "\nBackground notes about the user (long-term context):\n"
            + (summary_text if summary_text else "(none)")
        )
        response = self.llm_controller.get_completion(
            "".join(prompt_parts),
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "finished": {"type": "integer"},
                            "answer": {"type": "string"},
                        },
                        "required": ["finished", "answer"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            },
            temperature=self.temperature,
        )
        data = parse_json_response(response)
        if data is None:
            return 1, ""
        try:
            finished = int(data.get("finished", 1))
        except (TypeError, ValueError):
            finished = 1
        return finished, (data.get("answer") or "")

    def _pick_relevant_summaries(self, question: str, indices: List[int]) -> List[int]:
        """Single-call rerank: tell the LLM which candidate ids matter."""
        if not indices:
            return []
        candidate_block = self._format_indexed_summaries(indices)
        if not candidate_block:
            return []
        prompt = (
            PromptTemplates.RETRIEVER
            + "\nUser question: " + question
            + "\nIndices:\n" + candidate_block
        )
        response = self.llm_controller.get_completion(
            prompt,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "keywords_list": {
                                "type": "array",
                                "items": {"type": "integer"},
                            }
                        },
                        "required": ["keywords_list"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            },
            temperature=self.temperature,
        )
        data = parse_json_response(response)
        if not data:
            return []
        return [int(i) for i in (data.get("keywords_list") or []) if isinstance(i, int)]

    def _deep_answer(self, question: str, full_memory: str, short_term: str) -> str:
        prompt_parts = [PromptTemplates.ANSWER_DEEP, "\nUser message: " + question]
        if short_term:
            prompt_parts.append("\nRecent dialogue (short-term context):\n" + short_term)
        prompt_parts.append(
            "\nBackground context about the user (long-term context):\n"
            + (full_memory if full_memory else "(none)")
        )
        response = self.llm_controller.get_completion(
            "".join(prompt_parts),
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            },
            temperature=self.temperature,
        )
        data = parse_json_response(response)
        return "" if data is None else (data.get("answer") or "")

    def _analyze_answer(self, question: str, answer: str) -> Tuple[int, str]:
        prompt = (
            PromptTemplates.ANALYZE_ANSWER
            + "\nQuestion: " + question
            + "\nDraft answer: " + answer
        )
        response = self.llm_controller.get_completion(
            prompt,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "finished": {"type": "integer"},
                            "new_question": {"type": "string"},
                        },
                        "required": ["finished", "new_question"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            },
            temperature=self.temperature,
        )
        data = parse_json_response(response)
        if data is None:
            return 1, ""
        try:
            finished = int(data.get("finished", 1))
        except (TypeError, ValueError):
            finished = 1
        return finished, data.get("new_question") or ""

    # ================================================================== #
    # Dedup-merge: collapse near-duplicate atomic facts after ingestion.
    # ================================================================== #

    def _dedup_merge(self, new_indices: List[int]) -> None:
        """For each freshly ingested summary, look up similar existing ones and
        ask the LLM whether to merge. All decisions are sent in a single LLM
        call for efficiency."""

        # Find candidates with similar pre-existing entries.
        candidates: List[Dict[str, Any]] = []
        new_set = set(new_indices)
        for new_idx in new_indices:
            s_new = self.summary_list.get(new_idx)
            if s_new is None:
                continue
            # Vector search for top-(k+m) and filter out the new batch.
            hits = self.retriever.search(s_new.content, self.dedup_top_k + len(new_set))
            neighbours = []
            for nb_idx, sim in hits:
                if nb_idx in new_set or nb_idx == new_idx:
                    continue
                if sim < self.dedup_sim_threshold:
                    continue
                s_nb = self.summary_list.get(nb_idx)
                if s_nb is None:
                    continue
                neighbours.append({"id": nb_idx, "text": s_nb.content})
                if len(neighbours) >= self.dedup_top_k:
                    break
            if neighbours:
                candidates.append({"new_id": new_idx, "new_text": s_new.content, "neighbours": neighbours})

        if not candidates:
            return

        decisions = self._llm_dedup_decide(candidates)
        if not decisions:
            return

        for d in decisions:
            try:
                action = d.get("action")
                new_id = int(d["new_id"])
            except (KeyError, TypeError, ValueError):
                continue
            if action != "merge":
                continue

            merged_text = (d.get("merged_text") or "").strip()
            merged_ids_raw = d.get("merged_ids") or []
            try:
                merged_ids = [int(x) for x in merged_ids_raw if isinstance(x, (int, float))]
            except (TypeError, ValueError):
                merged_ids = []
            if not merged_text or not merged_ids:
                continue

            self._apply_merge(new_id, merged_ids, merged_text)

    def _apply_merge(self, new_idx: int, merged_old_ids: List[int], merged_text: str) -> None:
        """Replace old summaries + new summary by ONE merged summary."""
        with self._write_lock:
            new_sum = self.summary_list.get(new_idx)
            if new_sum is None:
                return

            # Collect inherited links from new + merged_old + remove duplicates.
            inherited_links: List[str] = list(new_sum.links)
            inherited_session = new_sum.session_id
            inherited_time = new_sum.timestamp

            old_summaries: List[MemorySummary] = []
            for old_idx in merged_old_ids:
                if old_idx == new_idx:
                    continue
                old_sum = self.summary_list.get(old_idx)
                if old_sum is None:
                    continue
                old_summaries.append(old_sum)
                for l in old_sum.links:
                    if l not in inherited_links:
                        inherited_links.append(l)

            if not old_summaries:
                return  # nothing to merge

            # Drop the merged-out entries (new + olds) from the index.
            drop_indices = [new_idx] + [self._idx_for_summary(s) for s in old_summaries]
            drop_indices = [i for i in drop_indices if i is not None]
            self.retriever.delete(drop_indices)
            for i in drop_indices:
                self.summary_list.pop(i, None)

            # Insert the merged summary as a fresh entry.
            merged_summary = MemorySummary(
                content=merged_text,
                links=inherited_links,
                timestamp=inherited_time,
                session_id=inherited_session,
            )
            assigned = self.retriever.add_documents(
                [merged_text],
                [{
                    "session_id": inherited_session,
                    "memory_type": "summary",
                    "timestamp": inherited_time,
                    "summary_id": merged_summary.id,
                    "merged": True,
                }],
            )
            if assigned:
                self.summary_list[assigned[0]] = merged_summary

    def _idx_for_summary(self, s: MemorySummary) -> Optional[int]:
        """Reverse-lookup: find the retriever index of a given MemorySummary.
        We keep summary_list keyed by index, so this is just a dict scan."""
        for idx, ss in self.summary_list.items():
            if ss is s or ss.id == s.id:
                return idx
        return None

    def _llm_dedup_decide(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        import json
        prompt = PromptTemplates.DEDUP_MERGE + json.dumps(candidates, ensure_ascii=False)
        response = self.llm_controller.get_completion(
            prompt,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "decisions": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "new_id": {"type": "integer"},
                                        "action": {"type": "string"},
                                        "merged_text": {"type": "string"},
                                        "merged_ids": {
                                            "type": "array",
                                            "items": {"type": "integer"},
                                        },
                                    },
                                    "required": ["new_id", "action"],
                                    "additionalProperties": True,
                                },
                            }
                        },
                        "required": ["decisions"],
                        "additionalProperties": False,
                    },
                    "strict": False,
                },
            },
            temperature=0.0,
        )
        data = parse_json_response(response)
        if not data:
            return []
        return data.get("decisions") or []

    # ================================================================== #
    # Persistence & maintenance
    # ================================================================== #

    def clear_memories(self) -> None:
        with self._write_lock:
            self.memories.clear()
            self.summary_list.clear()
            self.sessions.clear()
            self.current_session_id = None
            self.retriever = EnhancedEmbeddingRetriever(
                self.retriever.model.model_name,
                self.retriever.model.api_key,
                self.retriever.model.api_base,
            )

    def save_memories(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        with self._write_lock:
            with open(os.path.join(directory, "memories.pkl"), "wb") as f:
                pickle.dump(self.memories, f)
            with open(os.path.join(directory, "summaries.pkl"), "wb") as f:
                # Save as a list of (idx, summary_dict) so we can rebuild the dict.
                pickle.dump(
                    [(i, s.to_dict()) for i, s in self.summary_list.items()],
                    f,
                )
            with open(os.path.join(directory, "sessions.pkl"), "wb") as f:
                pickle.dump({sid: s.to_dict() for sid, s in self.sessions.items()}, f)
            self.retriever.save(
                os.path.join(directory, "retriever.pkl"),
                os.path.join(directory, "embeddings.npy"),
            )

    def load_memories(self, directory: str) -> None:
        with self._write_lock:
            mfile = os.path.join(directory, "memories.pkl")
            if os.path.exists(mfile):
                with open(mfile, "rb") as f:
                    self.memories = pickle.load(f)
            sfile = os.path.join(directory, "summaries.pkl")
            if os.path.exists(sfile):
                with open(sfile, "rb") as f:
                    raw = pickle.load(f)
                self.summary_list = {}
                if isinstance(raw, list):  # new format
                    for item in raw:
                        try:
                            i, d = item
                            self.summary_list[int(i)] = MemorySummary.from_dict(d)
                        except Exception:
                            continue
                elif isinstance(raw, dict):  # alt new format
                    for i, d in raw.items():
                        try:
                            self.summary_list[int(i)] = MemorySummary.from_dict(d) if isinstance(d, dict) else d
                        except Exception:
                            continue
                else:  # old list-of-MemorySummary format -> map by position
                    for i, s in enumerate(raw or []):
                        if isinstance(s, MemorySummary):
                            self.summary_list[i] = s
            sessfile = os.path.join(directory, "sessions.pkl")
            if os.path.exists(sessfile):
                with open(sessfile, "rb") as f:
                    data = pickle.load(f)
                    self.sessions = {
                        sid: ConversationSession.from_dict(d) for sid, d in data.items()
                    }
            rfile = os.path.join(directory, "retriever.pkl")
            efile = os.path.join(directory, "embeddings.npy")
            if os.path.exists(rfile) and os.path.exists(efile):
                self.retriever.load(rfile, efile)


# Backward compatibility aliases
EnhancedAgenticMemorySystem = HybridMemorySystem
AgenticMemorySystem = HybridMemorySystem
