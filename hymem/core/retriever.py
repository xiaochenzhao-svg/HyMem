"""
Embedding-based retrieval system for HyMem.

This module provides semantic search capabilities using text embeddings
and cosine similarity for memory retrieval.
"""

import os
from typing import List, Dict, Optional
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class SimpleEmbeddingRetriever:
    """
    Simple retrieval system using text embeddings for semantic search.
    
    This retriever uses OpenAI embeddings to encode documents and queries,
    then performs similarity search using cosine similarity.
    
    Attributes:
        model: OpenAI embedding model instance
        corpus: List of document texts
        embeddings: Numpy array of document embeddings
        document_ids: Mapping from document text to index
    
    Example:
        >>> retriever = SimpleEmbeddingRetriever(
        ...     model_name="text-embedding-ada-002",
        ...     api_key="your-api-key",
        ...     base_url="https://api.openai.com/v1"
        ... )
        >>> retriever.add_documents(["Hello world", "Python is great"])
        >>> results = retriever.search("greeting", k=1)
    """
    
    def __init__(
        self,
        model_name: str = '',
        api_key: str = '',
        base_url: str = ''
    ):
        """
        Initialize the SimpleEmbeddingRetriever.
        
        Args:
            model_name: Name of the embedding model to use
            api_key: OpenAI API key
            base_url: Base URL for the OpenAI API
        """
        # Lazy import to avoid dependency issues
        from llama_index.embeddings.openai import OpenAIEmbedding
        
        self.model = OpenAIEmbedding(
            model_name=model_name,
            api_base=base_url,
            api_key=api_key
        )
        self.corpus: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.document_ids: Dict[str, int] = {}
    
    def add_documents(self, documents: List[str]) -> None:
        """
        Add documents to the retriever's corpus.
        
        If the corpus is empty, initializes it with the provided documents.
        Otherwise, extends the existing corpus.
        
        Args:
            documents: List of document texts to add
        """
        if not self.corpus:
            # First batch of documents
            self.corpus = documents
            self.embeddings = np.array(self.model.get_text_embedding_batch(documents))
            self.document_ids = {doc: idx for idx, doc in enumerate(documents)}
        else:
            # Extend existing corpus
            start_idx = len(self.corpus)
            self.corpus.extend(documents)
            
            # Get embeddings for new documents
            new_embeddings = np.array(self.model.get_text_embedding_batch(documents))
            if self.embeddings is None:
                self.embeddings = new_embeddings
            else:
                self.embeddings = np.vstack([self.embeddings, new_embeddings])
            
            # Update document index mapping
            for idx, doc in enumerate(documents):
                self.document_ids[doc] = start_idx + idx
    
    def search(self, query: str, k: int = 5) -> np.ndarray:
        """
        Search for similar documents using cosine similarity.
        
        Args:
            query: Query text to search for
            k: Number of top results to return
            
        Returns:
            Array of indices of the top-k most similar documents
        """
        if not self.corpus or self.embeddings is None:
            return np.array([])
        
        # Get query embedding
        query_embedding = self.model.get_text_embedding(query)
        
        # Calculate cosine similarities
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # Get top k results (indices)
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        return top_k_indices
    
    def save(
        self,
        retriever_cache_file: str,
        retriever_cache_embeddings_file: str
    ) -> None:
        """
        Save retriever state to disk.
        
        Saves embeddings as numpy array and other state as pickle.
        
        Args:
            retriever_cache_file: Path to save the corpus and document_ids
            retriever_cache_embeddings_file: Path to save the embeddings array
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(retriever_cache_file), exist_ok=True)
        
        # Save embeddings using numpy
        if self.embeddings is not None:
            np.save(retriever_cache_embeddings_file, self.embeddings)
        
        # Save other attributes
        state = {
            'corpus': self.corpus,
            'document_ids': self.document_ids
        }
        with open(retriever_cache_file, 'wb') as f:
            pickle.dump(state, f)
    
    def load(
        self,
        retriever_cache_file: str,
        retriever_cache_embeddings_file: str
    ) -> "SimpleEmbeddingRetriever":
        """
        Load retriever state from disk.
        
        Args:
            retriever_cache_file: Path to load the corpus and document_ids from
            retriever_cache_embeddings_file: Path to load the embeddings array from
            
        Returns:
            Self with loaded state
        """
        # Load embeddings
        if os.path.exists(retriever_cache_embeddings_file):
            print(f"Loading embeddings from {retriever_cache_embeddings_file}")
            self.embeddings = np.load(retriever_cache_embeddings_file)
            print(f"Embeddings shape: {self.embeddings.shape}")
        else:
            print(f"Embeddings file not found: {retriever_cache_embeddings_file}")
        
        # Load corpus and document_ids
        if os.path.exists(retriever_cache_file):
            print(f"Loading corpus from {retriever_cache_file}")
            with open(retriever_cache_file, 'rb') as f:
                state = pickle.load(f)
                self.corpus = state['corpus']
                self.document_ids = state['document_ids']
                print(f"Loaded corpus with {len(self.corpus)} documents")
        else:
            print(f"Corpus file not found: {retriever_cache_file}")
        
        return self
    
    @classmethod
    def load_from_local_memory(
        cls,
        memories: Dict,
        model_name: str,
        api_key: str = '',
        base_url: str = ''
    ) -> "SimpleEmbeddingRetriever":
        """
        Create a retriever from existing memory entries.
        
        This is a factory method that creates a new retriever instance
        and initializes it with memory content and metadata.
        
        Args:
            memories: Dictionary of memory entries (id -> MemoryNote)
            model_name: Name of the embedding model
            api_key: OpenAI API key
            base_url: Base URL for the OpenAI API
            
        Returns:
            Initialized SimpleEmbeddingRetriever instance
        """
        # Create documents combining content and metadata for each memory
        all_docs = []
        for m in memories.values():
            # Combine content with metadata for better retrieval
            metadata_text = f"{getattr(m, 'context', '')} {' '.join(getattr(m, 'keywords', []))} {' '.join(getattr(m, 'tags', []))}"
            doc = f"{m.content} , {metadata_text}"
            all_docs.append(doc)
        
        # Create and initialize retriever
        retriever = cls(model_name, api_key, base_url)
        retriever.add_documents(all_docs)
        
        return retriever
    
    def __len__(self) -> int:
        """Return the number of documents in the corpus."""
        return len(self.corpus)
    
    def __repr__(self) -> str:
        return f"SimpleEmbeddingRetriever(corpus_size={len(self.corpus)})"
