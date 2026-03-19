"""
LLM Controller for HyMem.

This module provides abstractions for interacting with Large Language Models,
with support for multiple backends (currently OpenAI).
"""

from abc import ABC, abstractmethod
from typing import Optional, Literal
import socket
import httpx


class BaseLLMController(ABC):
    """
    Abstract base class for LLM controllers.
    
    Defines the interface for LLM interaction that all backend implementations
    must follow.
    """
    
    @abstractmethod
    def get_completion(
        self,
        prompt: str,
        response_format: dict,
        temperature: float = 0.7
    ) -> str:
        """
        Get completion from the LLM.
        
        Args:
            prompt: Input prompt for the LLM
            response_format: Expected response format specification
            temperature: Sampling temperature for generation
            
        Returns:
            LLM response as string
        """
        pass


class OpenAIController(BaseLLMController):
    """
    OpenAI-specific LLM controller implementation.
    
    Handles communication with OpenAI's API for text generation,
    with proper connection management and error handling.
    
    Attributes:
        model: Name of the OpenAI model to use
        client: OpenAI client instance
    
    Example:
        >>> controller = OpenAIController(
        ...     model="gpt-4",
        ...     api_key="your-api-key",
        ...     base_url="https://api.openai.com/v1"
        ... )
        >>> response = controller.get_completion(
        ...     prompt="Hello, how are you?",
        ...     response_format={"type": "json_object"},
        ...     temperature=0.7
        ... )
    """
    
    def __init__(
        self,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        """
        Initialize the OpenAI controller.
        
        Args:
            model: Name of the OpenAI model to use
            api_key: OpenAI API key (can also be set via environment)
            base_url: Base URL for the API (for custom endpoints)
            
        Raises:
            ImportError: If openai package is not installed
            ValueError: If API key is not provided
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI package not found. Install it with: pip install openai"
            )
        
        if api_key is None:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                "or provide api_key parameter."
            )
        
        # Configure connection with keep-alive settings
        socket_options = [
            (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),
            (socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3),
            (socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 30),
            (socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 30)
        ]
        transport = httpx.HTTPTransport(socket_options=socket_options)
        http_client = httpx.Client(transport=transport)
        
        self.model = model
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=http_client
        )
    
    def get_completion(
        self,
        prompt: str,
        response_format: dict,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> str:
        """
        Get completion from OpenAI's API.
        
        Args:
            prompt: Input prompt for the model
            response_format: Response format specification (e.g., JSON schema)
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens in the response
            
        Returns:
            Model's response content as string
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You must respond with a JSON object."},
                {"role": "user", "content": prompt}
            ],
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content


class LLMController:
    """
    Factory class for creating LLM controller instances.
    
    Provides a unified interface for creating LLM controllers based on
    the specified backend type.
    
    Attributes:
        llm: The underlying LLM controller instance
    
    Example:
        >>> controller = LLMController(
        ...     backend="openai",
        ...     model="gpt-4",
        ...     api_key="your-api-key"
        ... )
    """
    
    def __init__(
        self,
        backend: Literal["openai"] = "openai",
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        """
        Initialize LLM controller with specified backend.
        
        Args:
            backend: Backend type (currently only "openai" supported)
            model: Model name for the chosen backend
            api_key: API key for authentication
            base_url: Base URL for the API (for custom endpoints)
            
        Raises:
            ValueError: If an unsupported backend is specified
        """
        if backend == "openai":
            self.llm = OpenAIController(model, api_key, base_url)
        else:
            raise ValueError(f"Unsupported backend: {backend}. Currently only 'openai' is supported.")
    
    def get_completion(
        self,
        prompt: str,
        response_format: dict,
        temperature: float = 0.7
    ) -> str:
        """
        Delegate completion request to the underlying controller.
        
        Args:
            prompt: Input prompt
            response_format: Response format specification
            temperature: Sampling temperature
            
        Returns:
            LLM response as string
        """
        return self.llm.get_completion(prompt, response_format, temperature)
    
    def __repr__(self) -> str:
        return f"LLMController(backend={type(self.llm).__name__})"
