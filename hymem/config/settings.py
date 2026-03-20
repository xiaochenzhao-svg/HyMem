"""
Configuration management for HyMem.

This module provides centralized configuration management with support for:
- Environment variables
- Configuration files
- Default values
"""

import os
from typing import Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LLMConfig:
    """LLM (Large Language Model) configuration."""
    
    model_name: str = "gpt-4.1-mini"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: int = 60
    
    def __post_init__(self):
        """Load from environment if not provided."""
        if self.api_key is None:
            self.api_key = os.getenv("OPENAI_API_KEY")
        if self.base_url is None:
            self.base_url = os.getenv("OPENAI_BASE_URL")

@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""
    
    model_name: str = "text-embedding-3-small"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    
    def __post_init__(self):
        """Load from environment if not provided."""
        if self.api_key is None:
            self.api_key = os.getenv("OPENAI_API_KEY")
        if self.base_url is None:
            self.base_url = os.getenv("OPENAI_BASE_URL")

@dataclass
class RetrievalConfig:
    """Retrieval configuration."""
    
    retrieve_k: int = 10
    retrieve_k_rough: int = 30
    max_iterations: int = 1
    

@dataclass
class CacheConfig:
    """Cache configuration."""
    
    enable_cache: bool = True
    cache_dir: str = "cached_memories"
    use_pickle: bool = True
    

@dataclass
class Settings:
    """
    Main settings class containing all configuration sections.
    
    Attributes:
        llm: LLM configuration
        embedding: Embedding model configuration
        retrieval: Retrieval configuration
        cache: Cache configuration
        backend: Backend type ('openai')
        log_level: Logging level
    """
    
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    backend: str = "openai"
    log_level: str = "INFO"
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "Settings":
        """Create Settings from dictionary."""
        llm_config = LLMConfig(**config_dict.get("llm", {}))
        embedding_config = EmbeddingConfig(**config_dict.get("embedding", {}))
        retrieval_config = RetrievalConfig(**config_dict.get("retrieval", {}))
        cache_config = CacheConfig(**config_dict.get("cache", {}))
        
        return cls(
            llm=llm_config,
            embedding=embedding_config,
            retrieval=retrieval_config,
            cache=cache_config,
            backend=config_dict.get("backend", "openai"),
            log_level=config_dict.get("log_level", "INFO"),
        )
    
    def to_dict(self) -> dict:
        """Convert Settings to dictionary."""
        from dataclasses import asdict
        return {
            "llm": asdict(self.llm),
            "embedding": asdict(self.embedding),
            "retrieval": asdict(self.retrieval),
            "cache": asdict(self.cache),
            "backend": self.backend,
            "log_level": self.log_level,
        }


class ConfigManager:
    """
    Configuration manager for loading and managing settings.
    
    Supports loading configuration from:
    - Dictionary
    - JSON file
    - Environment variables (with defaults)
    
    Example:
        >>> config_manager = ConfigManager()
        >>> settings = config_manager.load_from_env()
        >>> settings = config_manager.load_from_file("config.json")
    """
    
    def __init__(self):
        self._settings: Optional[Settings] = None
    
    def load_from_dict(self, config_dict: dict) -> Settings:
        """Load configuration from dictionary."""
        self._settings = Settings.from_dict(config_dict)
        return self._settings
    
    def load_from_env(self) -> Settings:
        """Load configuration from environment variables."""
        llm_config = LLMConfig(
            model_name=os.getenv("LLM_MODEL_NAME", "gpt-4.1-mini"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2048")),
        )
        
        embedding_config = EmbeddingConfig(
            model_name=os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small"),
        )
        
        retrieval_config = RetrievalConfig(
            retrieve_k=int(os.getenv("RETRIEVE_K", "10")),
            retrieve_k_rough=int(os.getenv("RETRIEVE_K_ROUGH", "30")),
        )
        
        self._settings = Settings(
            llm=llm_config,
            embedding=embedding_config,
            retrieval=retrieval_config,
        )
        return self._settings
    
    def load_from_file(self, file_path: str) -> Settings:
        """Load configuration from JSON file."""
        import json
        with open(file_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return self.load_from_dict(config_dict)
    
    @property
    def settings(self) -> Settings:
        """Get current settings."""
        if self._settings is None:
            self._settings = Settings()
        return self._settings
