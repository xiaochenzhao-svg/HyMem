"""
Hybrid Memory Agent for HyMem.

This module provides the main agent class that wraps the memory system
and provides a simple interface for adding memories and answering questions.
"""

from typing import Optional, Tuple
from hymem.core.memory_system import AgenticMemorySystem
from hymem.config.settings import Settings


class HybridMemAgent:
    """
    Hybrid Memory Agent that combines memory management with question answering.
    
    This agent wraps the AgenticMemorySystem and provides a simplified interface
    for memory operations and retrieval-based question answering.
    
    Attributes:
        memory_system: The underlying memory management system
        retrieve_k: Number of results for initial retrieval
        retrieve_k_rough: Number of results for rough retrieval
        temperature: Temperature parameter for generation
    
    Example:
        >>> agent = HybridMemAgent(
        ...     embed_model="text-embedding-ada-002",
        ...     model_name="gpt-4",
        ...     api_key="your-api-key",
        ...     embed_api_key="your-embed-api-key"
        ... )
        >>> agent.add_memory("Alice lives in New York")
        >>> answer, context = agent.answer_question("Where does Alice live?", category=1)
    """
    
    def __init__(
        self,
        embed_model: str,
        model_name: str,
        embed_api_key: str,
        api_key: str,
        embed_base_url: str,
        base_url: str,
        backend: str,
        retrieve_k: int = 15,
        temperature: float = 0.7,
        k_rough: int = 30
    ):
        """
        Initialize the HybridMemAgent.
        
        Args:
            embed_model: Name of the embedding model
            model_name: Name of the LLM model
            embed_api_key: API key for embedding model
            api_key: API key for LLM
            embed_base_url: Base URL for embedding model API
            base_url: Base URL for LLM API
            backend: Backend type ("openai")
            retrieve_k: Number of results for initial retrieval
            temperature: Temperature for generation
            k_rough: Number of results for rough retrieval
        """
        self.memory_system = AgenticMemorySystem(
            llm_backend=backend,
            embed_llm_model=embed_model,
            embed_api_key=embed_api_key,
            embed_base_url=embed_base_url,
            llm_model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature
        )
        
        self.retrieve_k = retrieve_k
        self.temperature = temperature
        self.retrieve_k_rough = k_rough
    
    @classmethod
    def from_settings(cls, settings: Settings) -> "HybridMemAgent":
        """
        Create HybridMemAgent from Settings object.
        
        Args:
            settings: Settings object containing configuration
            
        Returns:
            Configured HybridMemAgent instance
        """
        return cls(
            embed_model=settings.embedding.model_name,
            model_name=settings.llm.model_name,
            embed_api_key=settings.embedding.api_key,
            api_key=settings.llm.api_key,
            embed_base_url=settings.embedding.base_url,
            base_url=settings.llm.base_url,
            backend=settings.backend,
            retrieve_k=settings.retrieval.retrieve_k,
            temperature=settings.llm.temperature,
            k_rough=settings.retrieval.retrieve_k_rough
        )

    def add_memory(
            self,
            content: str,
            time: Optional[str] = None,
            precomputed_summary: Optional[list] = None
    ) -> None:

        self.memory_system.add_note(
            content,
            time=time,
            precomputed_summary=precomputed_summary
        )
    
    def answer_question(
        self,
        question: str,
        category: int,
        answer: str
    ) -> Tuple[str, str]:
        """
        Answer a question using dynamic retrieval.
        
        Args:
            question: Question to answer
            category: Question category (1-5)
            answer: Reference answer (not used in current implementation)
            
        Returns:
            Tuple of (prediction, user_prompt)
            - prediction: Generated answer
            - user_prompt: Retrieved memory context used for generation
        """
        return self.memory_system.dynamic_retrieval(
            question,
            k=self.retrieve_k,
            k_rough=self.retrieve_k_rough
        )
    
    def clear_memories(self) -> None:
        """Clear all memories from the agent."""
        self.memory_system.memories.clear()
        self.memory_system.summary_list.clear()
        self.memory_system.retriever.corpus.clear()
        self.memory_system.retriever.embeddings = None
        self.memory_system.retriever.document_ids.clear()
    
    def __repr__(self) -> str:
        return (
            f"HybridMemAgent("
            f"memories={len(self.memory_system.memories)}, "
            f"summaries={len(self.memory_system.summary_list)}, "
            f"k={self.retrieve_k}, "
            f"k_rough={self.retrieve_k_rough}"
            f")"
        )
