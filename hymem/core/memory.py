"""
Memory data structures for HyMem.

This module defines the basic memory units used in the hybrid memory system.
"""

from typing import Optional, List, Dict
from datetime import datetime
import uuid


class MemoryNote:
    """
    Basic memory unit with metadata.
    
    Represents a single memory entry with content, timestamps, and metadata
    for tracking importance and retrieval patterns.
    
    Attributes:
        content: The main content of the memory
        id: Unique identifier for the memory
        links: Links to related memories or resources
        importance_score: Score indicating memory importance (default: 1.0)
        retrieval_count: Number of times this memory has been retrieved
        timestamp: Creation time of the memory
        last_accessed: Last access time of the memory
        category: Category classification of the memory
    """
    
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
        **kwargs
    ):
        """
        Initialize a MemoryNote instance.
        
        Args:
            content: The memory content
            id: Optional unique identifier (auto-generated if not provided)
            links: Optional list of related memory links
            importance_score: Optional importance score
            retrieval_count: Optional retrieval count
            timestamp: Optional creation timestamp
            last_accessed: Optional last accessed timestamp
            context: Optional context information
            evolution_history: Optional evolution history
            category: Optional category
            tags: Optional list of tags
            **kwargs: Additional keyword arguments for extensibility
        """
        self.content = content
        self.id = id or str(uuid.uuid4())
        self.links = links or []
        self.importance_score = importance_score or 1.0
        self.retrieval_count = retrieval_count or 0
        
        current_time = datetime.now().strftime("%Y%m%d%H%M")
        self.timestamp = timestamp or current_time
        self.last_accessed = last_accessed or current_time
        self.context = context
        self.evolution_history = evolution_history or []
        self.category = category or "Uncategorized"
        self.tags = tags or []
        
        # Store any additional kwargs for extensibility
        self._extra_attrs = kwargs
    
    def __getattr__(self, name):
        """Allow access to extra attributes."""
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        return self._extra_attrs.get(name)
    
    def update_access_time(self) -> None:
        """Update the last accessed timestamp to current time."""
        self.last_accessed = datetime.now().strftime("%Y%m%d%H%M")
        self.retrieval_count += 1
    
    def to_dict(self) -> dict:
        """Convert MemoryNote to dictionary representation."""
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
            **self._extra_attrs
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "MemoryNote":
        """Create MemoryNote from dictionary."""
        return cls(**data)
    
    def __repr__(self) -> str:
        return f"MemoryNote(id={self.id}, category={self.category}, retrieval_count={self.retrieval_count})"


class MemorySummary:
    """
    Summary of memory content for efficient retrieval.
    
    Represents a condensed version of memory content with links to the
    original memory entries, used for efficient semantic retrieval.
    
    Attributes:
        content: Summary content
        memory_links: Indices of linked memory entries
        timestamp: Timestamp of the summary
    """
    
    def __init__(
        self,
        content: str,
        link: Optional[str] = None,
        timestamp: Optional[str] = None
    ):
        """
        Initialize a MemorySummary instance.
        
        Args:
            content: The summary content
            links: Optional list of memory indices this summary links to
            timestamp: Optional timestamp for the summary
        """
        self.content = content
        self.link = link
        self.timestamp = timestamp
    
    def to_dict(self) -> dict:
        """Convert MemorySummary to dictionary representation."""
        return {
            "content": self.content,
            "memory_links": self.memory_links,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "MemorySummary":
        """Create MemorySummary from dictionary."""
        return cls(**data)
    
    def __repr__(self) -> str:
        return f"MemorySummary(links={len(self.memory_links)}, timestamp={self.timestamp})"
