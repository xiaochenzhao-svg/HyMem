"""
Embedding-based retriever for HyMem.

Backed by a normalised numpy matrix + cosine search. Adds:
  * Tombstone-based logical deletion (so dedup-merge can drop replaced
    summaries without rebuilding the matrix every time).
  * Index compaction (`compact()`) when the live ratio drops below a
    threshold, so the working set stays cache-friendly.
  * Thread-safety on writes/searches.
"""

import os
import pickle
import threading
import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Iterable


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


class EnhancedEmbeddingRetriever:
    """Vector store + cosine search."""

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: str = "",
        base_url: str = "",
        embedding_dim: int = 1536,
    ):
        from llama_index.embeddings.openai import OpenAIEmbedding

        self.model = OpenAIEmbedding(
            model_name=model_name,
            api_base=base_url,
            api_key=api_key,
        )
        self.corpus: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None        # raw vectors
        self._normalized: Optional[np.ndarray] = None       # cache
        self.document_metadata: Dict[int, Dict[str, Any]] = {}
        self.session_embeddings: Dict[str, List[int]] = {}
        # Tombstones: indices marked deleted; excluded from search results.
        self._deleted: set = set()
        self.embedding_dim = embedding_dim
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ #
    # Indexing
    # ------------------------------------------------------------------ #

    def add_documents(
        self,
        documents: List[str],
        metadata_list: Optional[List[Dict[str, Any]]] = None,
    ) -> List[int]:
        """Encode and add documents. Returns the assigned indices."""
        if not documents:
            return []
        if metadata_list is None:
            metadata_list = [{} for _ in documents]
        if len(documents) != len(metadata_list):
            raise ValueError("documents and metadata_list must have the same length")

        new_embeddings = np.asarray(
            self.model.get_text_embedding_batch(documents), dtype=np.float32
        )

        with self._lock:
            if self.embeddings is None:
                self.embeddings = new_embeddings
            else:
                self.embeddings = np.vstack([self.embeddings, new_embeddings])
            self._normalized = None

            start = len(self.corpus)
            assigned: List[int] = []
            for i, (doc, metadata) in enumerate(zip(documents, metadata_list)):
                idx = start + i
                self.corpus.append({"text": doc, "metadata": metadata, "index": idx})
                self.document_metadata[idx] = metadata
                sid = metadata.get("session_id")
                if sid:
                    self.session_embeddings.setdefault(sid, []).append(idx)
                assigned.append(idx)
            return assigned

    def delete(self, indices: Iterable[int]) -> int:
        """Mark indices as deleted (logical removal). Returns count actually deleted."""
        n = 0
        with self._lock:
            for i in indices:
                if isinstance(i, (int, np.integer)) and 0 <= int(i) < len(self.corpus) and int(i) not in self._deleted:
                    self._deleted.add(int(i))
                    n += 1
            if n and self._live_ratio() < 0.5:
                self._compact_locked()
        return n

    def _live_ratio(self) -> float:
        if not self.corpus:
            return 1.0
        return 1.0 - (len(self._deleted) / len(self.corpus))

    def _compact_locked(self) -> None:
        """Physically drop deleted rows to keep the working set small."""
        if not self._deleted:
            return
        keep = [i for i in range(len(self.corpus)) if i not in self._deleted]
        old_to_new = {old: new for new, old in enumerate(keep)}

        new_corpus = []
        new_meta: Dict[int, Dict[str, Any]] = {}
        new_session: Dict[str, List[int]] = {}
        for old in keep:
            new_idx = old_to_new[old]
            entry = self.corpus[old]
            entry = {"text": entry["text"], "metadata": entry["metadata"], "index": new_idx}
            new_corpus.append(entry)
            new_meta[new_idx] = entry["metadata"]
            sid = entry["metadata"].get("session_id")
            if sid:
                new_session.setdefault(sid, []).append(new_idx)

        if self.embeddings is not None:
            self.embeddings = self.embeddings[keep]
        self._normalized = None
        self.corpus = new_corpus
        self.document_metadata = new_meta
        self.session_embeddings = new_session
        self._deleted = set()

    # ------------------------------------------------------------------ #
    # Search
    # ------------------------------------------------------------------ #

    def _ensure_normalized(self) -> Optional[np.ndarray]:
        if self._normalized is None and self.embeddings is not None:
            self._normalized = _l2_normalize(self.embeddings)
        return self._normalized

    def _embed_query(self, query: str) -> Optional[np.ndarray]:
        vec = np.asarray(self.model.get_text_embedding(query), dtype=np.float32)
        n = np.linalg.norm(vec)
        if n == 0:
            return None
        return vec / n

    def search(
        self,
        query: str,
        k: int = 5,
        session_filter: Optional[str] = None,
    ) -> List[Tuple[int, float]]:
        with self._lock:
            if not self.corpus or self.embeddings is None:
                return []
            qv = self._embed_query(query)
            if qv is None:
                return []
            normalized = self._ensure_normalized()
            return self._search_with_vector(qv, k, session_filter, normalized)

    def search_with_vector(
        self,
        query_vec: np.ndarray,
        k: int,
        session_filter: Optional[str] = None,
    ) -> List[Tuple[int, float]]:
        with self._lock:
            if not self.corpus or self.embeddings is None:
                return []
            n = np.linalg.norm(query_vec)
            if n == 0:
                return []
            qv = query_vec / n
            return self._search_with_vector(qv, k, session_filter, self._ensure_normalized())

    def _search_with_vector(
        self,
        qv: np.ndarray,
        k: int,
        session_filter: Optional[str],
        normalized: np.ndarray,
    ) -> List[Tuple[int, float]]:
        if session_filter and session_filter in self.session_embeddings:
            candidate = [i for i in self.session_embeddings[session_filter] if i not in self._deleted]
            if not candidate:
                return []
            sub = normalized[candidate]
            sims = sub @ qv
            order = np.argsort(-sims)[:k]
            return [(candidate[int(i)], float(sims[int(i)])) for i in order]

        sims = normalized @ qv
        if self._deleted:
            for d in self._deleted:
                sims[d] = -np.inf
        top = np.argsort(-sims)[:k]
        return [(int(i), float(sims[int(i)])) for i in top if sims[int(i)] != -np.inf]

    # ------------------------------------------------------------------ #
    # Accessors
    # ------------------------------------------------------------------ #

    def get_document_text(self, index: int) -> Optional[str]:
        if 0 <= index < len(self.corpus) and index not in self._deleted:
            return self.corpus[index]["text"]
        return None

    def get_document_metadata(self, index: int) -> Optional[Dict[str, Any]]:
        if index in self._deleted:
            return None
        return self.document_metadata.get(index)

    def get_session_documents(self, session_id: str) -> List[int]:
        return [i for i in self.session_embeddings.get(session_id, []) if i not in self._deleted]

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, retriever_cache_file: str, retriever_cache_embeddings_file: str) -> None:
        with self._lock:
            self._compact_locked()  # always save a clean snapshot
            cache_dir = os.path.dirname(retriever_cache_file)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
            if self.embeddings is not None:
                np.save(retriever_cache_embeddings_file, self.embeddings)
            with open(retriever_cache_file, "wb") as f:
                pickle.dump(
                    {
                        "corpus": self.corpus,
                        "document_metadata": self.document_metadata,
                        "session_embeddings": self.session_embeddings,
                        "embedding_dim": self.embedding_dim,
                    },
                    f,
                )

    def load(self, retriever_cache_file: str, retriever_cache_embeddings_file: str):
        with self._lock:
            if os.path.exists(retriever_cache_embeddings_file):
                self.embeddings = np.load(retriever_cache_embeddings_file)
                self._normalized = None
            if os.path.exists(retriever_cache_file):
                with open(retriever_cache_file, "rb") as f:
                    state = pickle.load(f)
                    self.corpus = state["corpus"]
                    self.document_metadata = state["document_metadata"]
                    self.session_embeddings = state["session_embeddings"]
                    self.embedding_dim = state["embedding_dim"]
            self._deleted = set()
        return self

    def __len__(self) -> int:
        return len(self.corpus) - len(self._deleted)

    def __repr__(self) -> str:
        return (
            f"EnhancedEmbeddingRetriever(documents={len(self)}, "
            f"sessions={len(self.session_embeddings)})"
        )


# Backward compatibility
SimpleEmbeddingRetriever = EnhancedEmbeddingRetriever
