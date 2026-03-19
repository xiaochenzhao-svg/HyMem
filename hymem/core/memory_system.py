"""
Agentic Memory System for HyMem.

This module implements the core memory management system with embedding-based
retrieval and LLM-powered content analysis and question answering.
"""

import os
import re
from typing import List, Dict, Optional, Tuple
import pickle
import numpy as np
from hymem.core.memory import MemoryNote, MemorySummary
from hymem.core.retriever import SimpleEmbeddingRetriever
from hymem.core.llm_controller import LLMController
from hymem.prompts.templates import PromptTemplates
from hymem.utils.helpers import parse_json_response, cal_token

class AgenticMemorySystem:
    """
    Memory management system with embedding-based retrieval.
    
    This system manages memory notes, generates summaries, and provides
    intelligent retrieval and question-answering capabilities using
    LLM and embedding models.
    
    Attributes:
        memories: Dictionary mapping memory IDs to MemoryNote objects
        retriever: Embedding-based retriever for semantic search
        llm_controller: LLM controller for text generation
        summary_list: List of MemorySummary objects
        temperature: Temperature parameter for LLM generation
    """
    
    def __init__(
        self,
        llm_backend: str = "openai",
        embed_llm_model: str = "",
        embed_api_key: Optional[str] = None,
        embed_base_url: Optional[str] = None,
        llm_model: str = "",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7
    ):
        """
        Initialize the AgenticMemorySystem.
        
        Args:
            llm_backend: Backend type ("openai")
            embed_llm_model: Embedding model name
            embed_api_key: API key for embedding model
            embed_base_url: Base URL for embedding model API
            llm_model: LLM model name
            api_key: API key for LLM
            base_url: Base URL for LLM API
            temperature: Temperature for LLM generation
        """
        self.memories: Dict[str, MemoryNote] = {}
        self.retriever = SimpleEmbeddingRetriever(
            embed_llm_model, embed_api_key, embed_base_url
        )
        self.llm_controller = LLMController(llm_backend, llm_model, api_key, base_url)
        self.summary_list: List[MemorySummary] = []
        self.temperature = temperature
    
    def analyze_content(
        self,
        content: str,
        llm_controller: Optional[LLMController] = None
    ) -> List[str]:
        """
        Analyze content and extract key information using LLM.
        
        Uses the EX_SUMMARY prompt to extract key information from content.
        
        Args:
            content: Content to analyze
            llm_controller: Optional LLM controller (uses self.llm_controller if not provided)
            
        Returns:
            List of extracted key information strings
        """
        if llm_controller is None:
            llm_controller = self.llm_controller

        # Build prompt
        prompt = PromptTemplates.EX_SUMMARY + content
        # Get LLM response
        response = llm_controller.llm.get_completion(
            prompt,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "keywords": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 1,
                            }
                        },
                        "required": ["keywords"],
                        "additionalProperties": False
                    },
                    "strict": True,
                },
            },
            temperature=self.temperature
        )
        
        # Parse response
        analysis = parse_json_response(response)
        if analysis is None:
            return []
        
        return analysis.get("keywords", [])

    def add_note(
            self,
            content: str,
            time: Optional[str] = None,
            precomputed_summary: Optional[list] = None,
            **kwargs
    ) -> None:


        if precomputed_summary is not None:
            summary = precomputed_summary
        else:
            max_retries = 2
            summary = []
            for _ in range(max_retries + 1):
                summary = self.analyze_content(content, self.llm_controller)
                if summary:
                    break
            if not summary:
                return

        note = MemoryNote(
            content=content,
            timestamp=time,
            **kwargs
        )

        memsums = [
            MemorySummary(
                content=su,
                link=note.id,
                timestamp=time
            )
            for su in summary
        ]

        docs = [m.content for m in memsums]
        embeddings = self.retriever.model.get_text_embedding_batch(docs)

        start_idx = len(self.retriever.corpus)
        self.summary_list.extend(memsums)
        self.retriever.corpus.extend(docs)
        embeddings = np.array(embeddings)
        if self.retriever.embeddings is None:
            self.retriever.embeddings = embeddings
        else:
            self.retriever.embeddings = np.vstack([
                self.retriever.embeddings,
                embeddings
            ])

        for i, doc in enumerate(docs):
            self.retriever.document_ids[doc] = start_idx + i

        self.memories[note.id] = note
    
    def dynamic_retrieval(
        self,
        question: str,
        k: int = 10,
        k_rough: int = 30
    ):
        """
        Perform dynamic retrieval to answer a question.
        
        Uses a two-stage retrieval process:
        1. Light retrieval with summary-level search
        2. Deep retrieval with detailed memory search if needed
        
        Args:
            question: Question to answer
            k: Number of results for initial retrieval
            k_rough: Number of results for rough retrieval
            
        Returns:
            Tuple of (answer, retrieved_memory_text)
        """

        if not self.memories:
            return "", ""
        answer = ""
        memory_buffer = ""
        max_iterations = 1
        
        for _ in range(max_iterations):
            # Initial retrieval
            initial_indices = self.retriever.search(question, k)
            summary_text = self._format_summaries(initial_indices) + memory_buffer
            
            # Light memory retrieval
            tag, answer = self.retrieval_light_memory(question, summary_text)
            retrieved_memory = summary_text
            # If light retrieval insufficient, do deep retrieval
            if tag == 2:
                expanded_indices = self.retriever.search(question, k_rough)
                memory_groups = self._group_summaries(expanded_indices, group_size=50)
                deep_selected_indices = []
                
                for group in memory_groups:
                    group_indices = self.retrieval_deep_memory(question, group)
                    deep_selected_indices.extend(group_indices)
                
                final_indices = [expanded_indices[i] for i in deep_selected_indices if i < len(expanded_indices)]
                retrieved_memory = self._build_memory_text(final_indices) + memory_buffer
                answer = self.answer_deep_memory(question, retrieved_memory)

            memory_buffer = answer
            
            # Analyze answer quality
            should_accept, revised_question = self.analyze_answer(question, answer)
            if should_accept == 1:
                return answer, retrieved_memory
            else:
                question = revised_question
        
        return answer, retrieved_memory
    
    def _format_summaries(self, indices) -> str:
        """Format summaries for display."""
        summary_parts = []
        for idx in indices:
            timestamp_match = re.search(r'on\s+(.+)$', self.summary_list[idx].timestamp)
            time_str = timestamp_match.group(1) if timestamp_match else self.summary_list[idx].timestamp
            summary_parts.append(f"time:{time_str}, {self.summary_list[idx].content}")
        return '\n'.join(summary_parts) + '\n'
    
    def _group_summaries(self, indices, group_size: int = 50) -> List[str]:
        """Group summaries into batches for processing."""
        groups = []
        current_group = []
        
        for idx, summary_idx in enumerate(indices):
            timestamp_match = re.search(r'on\s+(.+)$', self.summary_list[summary_idx].timestamp)
            time_str = timestamp_match.group(1) if timestamp_match else self.summary_list[summary_idx].timestamp
            entry = f"id:{idx}, time:{time_str}, {self.summary_list[summary_idx].content}"
            current_group.append(entry)
            
            if len(current_group) >= group_size or idx == len(indices) - 1:
                groups.append('\n'.join(current_group))
                current_group = []
        
        return groups
    
    def _build_memory_text(
        self,
        indices,
    ) -> str:
        memory_text = ""
        used_indices = set()
        for idx in indices:
            if not isinstance(idx, int):
                try:
                    idx = int(idx)
                except ValueError:
                    continue
            if 0 <= idx < len(self.summary_list) and self.summary_list[idx].link in self.memories:
                link = self.summary_list[idx].link
                if link not in used_indices:
                    used_indices.add(link)
                    memory_text += f"{self.memories[link].content}\n"
        return memory_text
    
    def answer_deep_memory(
        self,
        question: str,
        current_memory: str
    ) -> str:
        """
        Generate answer using deep memory retrieval.
        
        Args:
            question: Question to answer
            current_memory: Retrieved memory context
            
        Returns:
            Generated answer
        """
        # Build prompt
        prompt_memory = PromptTemplates.ANSWER_DEEP
        prompt_memory += "Here is the question: " + question + "\n"
        prompt_memory += " Here is the memory: " + current_memory + "\n"
        
        # Get LLM response
        response = self.llm_controller.llm.get_completion(
            prompt_memory,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "answer": {"type": "string"}
                        },
                        "required": ["answer"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            },
            temperature=self.temperature
        )
        
        # Parse response
        result = parse_json_response(response)
        if result is None:
            return ""
        
        return result.get("answer", "")
    
    def analyze_answer(
        self,
        question: str,
        answer: str
    ) -> Tuple[int, str]:
        """
        Analyze answer quality and optionally generate revised question.
        
        Args:
            question: Original question
            answer: Generated answer
            
        Returns:
            Tuple of (finished_flag, new_question)
            - finished_flag: 1 if answer is acceptable, 0 if needs revision
            - new_question: Revised question if finished_flag is 0
        """
        # Build prompt
        prompt = PromptTemplates.ANALYZE_ANSWER
        prompt += "Here is the question: " + question + "\n"
        prompt += " Here is the answer: " + answer + "\n"
        
        # Get LLM response
        response = self.llm_controller.llm.get_completion(
            prompt,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "finished": {"type": "integer"},
                            "new_question": {"type": "string"}
                        },
                        "required": ["finished", "new_question"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            }
        )
        
        # Parse response
        result = parse_json_response(response)
        if result is None:
            return 1, ""
        
        finished = int(result.get("finished", 1))
        new_question = result.get("new_question", "")
        
        return finished, new_question
    
    def retrieval_light_memory(
        self,
        question: str,
        memory_list_str: str
    ) -> Tuple[int, str]:
        """
        Perform light retrieval using summary context.
        
        Args:
            question: Question to answer
            memory_list_str: Memory summary context
            
        Returns:
            Tuple of (tag, answer)
            - tag: 2 if needs deep retrieval, 0 if answer is ready
            - answer: Generated answer or empty string
        """
        # Build prompt
        prompt_memory = PromptTemplates.ANSWER_LIGHT
        prompt_memory += "Here is the question: " + question + "\n"
        prompt_memory += " Here is the memory: " + memory_list_str + "\n"
        
        # Get LLM response
        response = self.llm_controller.llm.get_completion(
            prompt_memory,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "finished": {"type": "integer"},
                            "answer": {"type": "string"}
                        },
                        "required": ["finished", "answer"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            },
            temperature=self.temperature
        )
        
        # Parse response
        result = parse_json_response(response)
        if result is None:
            return 1, ""
        
        finished = int(result.get("finished", 1))
        answer = result.get("answer", "")
        
        return finished, answer
    
    def retrieval_deep_memory(
        self,
        question: str,
        current_memory_str: str
    ) -> List[int]:
        """
        Perform deep retrieval to select relevant memory indices.
        
        Args:
            question: Question to answer
            current_memory_str: Memory context string
            
        Returns:
            List of selected memory indices
        """
        # Build prompt
        prompt_memory = PromptTemplates.RETRIEVER
        prompt_memory += "Below is the current question to answer: " + question + "\n"
        prompt_memory += "Below is the memory content indices:\n " + current_memory_str + "\n"
        
        # Get LLM response
        response = self.llm_controller.llm.get_completion(
            prompt_memory,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "keywords_list": {
                                "type": "array",
                                "items": {"type": "integer"}
                            }
                        },
                        "required": ["keywords_list"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            },
            temperature=self.temperature
        )
        
        # Parse response
        result = parse_json_response(response)
        if result is None:
            return []
        
        return result.get("keywords_list", [])
    
    def save_memories(self, directory: str) -> None:
        """
        Save memories and summaries to disk.
        
        Args:
            directory: Directory to save files
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save memories
        memories_file = os.path.join(directory, "memories.pkl")
        with open(memories_file, 'wb') as f:
            pickle.dump(self.memories, f)
        
        # Save summaries
        summaries_file = os.path.join(directory, "summaries.pkl")
        with open(summaries_file, 'wb') as f:
            pickle.dump(self.summary_list, f)
        
        # Save retriever
        retriever_file = os.path.join(directory, "retriever.pkl")
        embeddings_file = os.path.join(directory, "embeddings.npy")
        self.retriever.save(retriever_file, embeddings_file)
    
    def load_memories(self, directory: str) -> None:
        """
        Load memories and summaries from disk.
        
        Args:
            directory: Directory containing saved files
        """
        # Load memories
        memories_file = os.path.join(directory, "memories.pkl")
        if os.path.exists(memories_file):
            with open(memories_file, 'rb') as f:
                self.memories = pickle.load(f)
        
        # Load summaries
        summaries_file = os.path.join(directory, "summaries.pkl")
        if os.path.exists(summaries_file):
            with open(summaries_file, 'rb') as f:
                self.summary_list = pickle.load(f)
        
        # Load retriever
        retriever_file = os.path.join(directory, "retriever.pkl")
        embeddings_file = os.path.join(directory, "embeddings.npy")
        if os.path.exists(retriever_file) and os.path.exists(embeddings_file):
            self.retriever.load(retriever_file, embeddings_file)
