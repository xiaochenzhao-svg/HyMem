"""
LLM Controller for HyMem.

Wraps an OpenAI-compatible chat-completions endpoint with:
  * a single shared httpx connection pool (TCP keepalive)
  * automatic retries with exponential backoff on transient errors
  * defensive handling of empty/null responses
"""

from abc import ABC, abstractmethod
from typing import Optional, Literal
import socket
import logging
import time

import httpx

logger = logging.getLogger(__name__)


# Optional dependency: tenacity. Fall back to a tiny in-house retry if missing.
try:
    from tenacity import (
        retry,
        stop_after_attempt,
        wait_exponential,
        retry_if_exception_type,
    )

    def _retry_decorator():
        return retry(
            reraise=True,
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=0.6, min=0.6, max=4.0),
            retry=retry_if_exception_type((httpx.HTTPError, TimeoutError, ConnectionError)),
        )
except ImportError:  # pragma: no cover - tenacity is optional
    def _retry_decorator():
        def deco(fn):
            def wrapper(*a, **kw):
                last = None
                for attempt in range(3):
                    try:
                        return fn(*a, **kw)
                    except (httpx.HTTPError, TimeoutError, ConnectionError) as e:
                        last = e
                        time.sleep(0.6 * (2 ** attempt))
                raise last  # type: ignore[misc]
            return wrapper
        return deco


def _build_http_client(timeout: float = 60.0) -> httpx.Client:
    """Build a tuned httpx.Client (cross-platform TCP keepalive)."""
    socket_options = [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)]
    if hasattr(socket, "TCP_KEEPCNT"):
        socket_options.append((socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3))
    if hasattr(socket, "TCP_KEEPINTVL"):
        socket_options.append((socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 30))
    if hasattr(socket, "TCP_KEEPIDLE"):
        socket_options.append((socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 30))
    transport = httpx.HTTPTransport(socket_options=socket_options)
    return httpx.Client(transport=transport, timeout=timeout)


class BaseLLMController(ABC):
    @abstractmethod
    def get_completion(self, prompt: str, response_format: dict, temperature: float = 0.7) -> str:
        pass


class OpenAIController(BaseLLMController):
    """OpenAI-compatible chat completions controller with retries."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
    ):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("openai package not found. Install with: pip install openai") from e

        if not api_key:
            raise ValueError("OpenAI API key not found.")

        self.model = model
        self._http_client = _build_http_client(timeout)
        self.client = OpenAI(api_key=api_key, base_url=base_url, http_client=self._http_client)

    @_retry_decorator()
    def get_completion(
        self,
        prompt: str,
        response_format: dict,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You must respond with a JSON object."},
                    {"role": "user", "content": prompt},
                ],
                response_format=response_format,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            logger.warning("LLM call failed: %s", e)
            raise

        if not response.choices or not response.choices[0].message:
            raise ValueError("Empty response from LLM API")
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("LLM response content is None")
        return content


class LLMController:
    """Factory wrapper. Currently OpenAI-compatible only."""

    def __init__(
        self,
        backend: Literal["openai"] = "openai",
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        if backend == "openai":
            self.llm = OpenAIController(model, api_key, base_url)
        else:
            raise ValueError(f"Unsupported backend: {backend}.")

    def get_completion(self, prompt: str, response_format: dict, temperature: float = 0.7) -> str:
        return self.llm.get_completion(prompt, response_format, temperature)

    def __repr__(self) -> str:
        return f"LLMController(backend={type(self.llm).__name__})"
