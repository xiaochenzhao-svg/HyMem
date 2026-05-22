"""
Configuration management for HyMem.

Supports loading from:
- A JSON config file (recommended)
- Environment variables
- A plain dict (programmatic use)
"""

import os
import json
from typing import Optional
from dataclasses import dataclass, field, asdict


@dataclass
class LLMConfig:
    model_name: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: int = 60

    def __post_init__(self):
        if self.api_key is None:
            self.api_key = os.getenv("OPENAI_API_KEY")
        if self.base_url is None:
            self.base_url = os.getenv("OPENAI_BASE_URL")


@dataclass
class EmbeddingConfig:
    model_name: str = "text-embedding-3-small"
    api_key: Optional[str] = None
    base_url: Optional[str] = None

    def __post_init__(self):
        if self.api_key is None:
            self.api_key = os.getenv("OPENAI_API_KEY")
        if self.base_url is None:
            self.base_url = os.getenv("OPENAI_BASE_URL")


@dataclass
class RetrievalConfig:
    """Dynamic-retrieval hyper-parameters."""

    retrieve_k: int = 10           # top-k for the light path's summary index
    retrieve_k_rough: int = 30     # top-k for the deep path's rough recall
    enable_self_check: bool = False  # extra self-check pass (slower, used in eval)
    short_term_turns: int = 8      # how many recent turns to inject as short-term context


@dataclass
class CacheConfig:
    enable_cache: bool = True
    cache_dir: str = "cached_memories"
    use_pickle: bool = True


@dataclass
class SessionConfig:
    """Short-term queue / second-tier memory configuration."""

    max_session_tokens: int = 4096   # token budget for the short-term FIFO queue
    auto_save_sessions: bool = True  # archive remaining session content on /quit
    async_indexing: bool = True      # archive in a background thread


@dataclass
class DedupConfig:
    """Atomic-fact deduplication via top-k similarity + LLM merge."""

    enabled: bool = True
    top_k: int = 5
    sim_threshold: float = 0.82


@dataclass
class Settings:
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    dedup: DedupConfig = field(default_factory=DedupConfig)
    backend: str = "openai"
    log_level: str = "INFO"

    @classmethod
    def from_dict(cls, config_dict: dict) -> "Settings":
        def _filter(cls_, data):
            allowed = {f.name for f in cls_.__dataclass_fields__.values()}
            return {k: v for k, v in (data or {}).items() if k in allowed}

        return cls(
            llm=LLMConfig(**_filter(LLMConfig, config_dict.get("llm"))),
            embedding=EmbeddingConfig(**_filter(EmbeddingConfig, config_dict.get("embedding"))),
            retrieval=RetrievalConfig(**_filter(RetrievalConfig, config_dict.get("retrieval"))),
            cache=CacheConfig(**_filter(CacheConfig, config_dict.get("cache"))),
            session=SessionConfig(**_filter(SessionConfig, config_dict.get("session"))),
            dedup=DedupConfig(**_filter(DedupConfig, config_dict.get("dedup"))),
            backend=config_dict.get("backend", "openai"),
            log_level=config_dict.get("log_level", "INFO"),
        )

    def to_dict(self) -> dict:
        return {
            "llm": asdict(self.llm),
            "embedding": asdict(self.embedding),
            "retrieval": asdict(self.retrieval),
            "cache": asdict(self.cache),
            "session": asdict(self.session),
            "dedup": asdict(self.dedup),
            "backend": self.backend,
            "log_level": self.log_level,
        }


class ConfigManager:
    def __init__(self):
        self._settings: Optional[Settings] = None

    def load_from_dict(self, config_dict: dict) -> Settings:
        self._settings = Settings.from_dict(config_dict)
        return self._settings

    def load_from_file(self, file_path: str) -> Settings:
        with open(file_path, "r", encoding="utf-8") as f:
            return self.load_from_dict(json.load(f))

    def load_from_env(self) -> Settings:
        self._settings = Settings(
            llm=LLMConfig(
                model_name=os.getenv("LLM_MODEL_NAME", "gpt-4o-mini"),
                temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
                max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2048")),
            ),
            embedding=EmbeddingConfig(
                model_name=os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small"),
            ),
            retrieval=RetrievalConfig(
                retrieve_k=int(os.getenv("RETRIEVE_K", "10")),
                retrieve_k_rough=int(os.getenv("RETRIEVE_K_ROUGH", "30")),
            ),
        )
        return self._settings

    @property
    def settings(self) -> Settings:
        if self._settings is None:
            self._settings = Settings()
        return self._settings
