"""Ollama API client with connection pooling and retry logic."""
import httpx
import logging
import asyncio
import json
from typing import List, Dict, Any, Optional, AsyncIterator
from app.config import settings

logger = logging.getLogger(__name__)


class OllamaClient:
    """
    Client for interacting with Ollama API.

    Provides methods for testing connection, listing models,
    and streaming chat completions.
    """

    def __init__(self, base_url: Optional[str] = None):
        """
        Initialize Ollama client.

        Args:
            base_url: Base URL for Ollama API (defaults to settings)
        """
        self.base_url = (base_url or settings.OLLAMA_URL).rstrip("/")
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(120.0, connect=5.0),  # Longer timeout for streaming
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        )
        logger.info(f"Ollama client initialized with base_url: {self.base_url}")

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    async def test_connection(self) -> bool:
        """
        Test connection to Ollama API.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            response = await self.client.get("/api/tags", timeout=5.0)
            response.raise_for_status()
            logger.info("Ollama connection test successful")
            return True
        except httpx.HTTPError as e:
            logger.error(f"Ollama connection test failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during connection test: {e}")
            return False

    async def list_models(self, retry_count: int = 3) -> List[Dict[str, Any]]:
        """
        List all available Ollama models with retry logic.

        Args:
            retry_count: Number of retries on failure

        Returns:
            List of model dictionaries

        Raises:
            Exception: If all retries fail
        """
        last_exception = None

        for attempt in range(retry_count):
            try:
                response = await self.client.get("/api/tags")
                response.raise_for_status()
                data = response.json()
                models = data.get("models", [])
                logger.info(f"Successfully retrieved {len(models)} models from Ollama")
                return models
            except httpx.HTTPError as e:
                last_exception = e
                logger.warning(
                    f"Failed to list models (attempt {attempt + 1}/{retry_count}): {e}"
                )
                if attempt < retry_count - 1:
                    # Exponential backoff
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
            except Exception as e:
                last_exception = e
                logger.error(f"Unexpected error listing models: {e}")
                break

        # All retries failed
        error_msg = f"Failed to list models after {retry_count} attempts: {last_exception}"
        logger.error(error_msg)
        raise Exception(error_msg)

    async def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific model.

        Args:
            model_name: Name of the model

        Returns:
            Model information dictionary or None if not found
        """
        try:
            # Use the /api/show endpoint for detailed model info
            response = await self.client.post(
                "/api/show",
                json={"name": model_name}
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.warning(f"Model info request failed for {model_name}: {e}")
            # Fallback to listing models
            try:
                models = await self.list_models()
                for model in models:
                    if model.get("name") == model_name or model.get("model") == model_name:
                        return model
            except Exception:
                pass
            logger.warning(f"Model not found: {model_name}")
            return None
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return None

    async def is_ollama_available(self) -> bool:
        """
        Check if Ollama service is available and responsive.

        Returns:
            True if available, False otherwise
        """
        return await self.test_connection()

    async def stream_generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        context: Optional[List[int]] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream text generation from Ollama.

        Args:
            model: Model name
            prompt: User prompt
            system: Optional system prompt
            temperature: Temperature for generation (0.0-2.0)
            context: Optional context from previous generation

        Yields:
            Dictionary with streaming response chunks

        Raises:
            Exception: If streaming fails
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
            },
        }

        if system:
            payload["system"] = system

        if context:
            payload["context"] = context

        logger.info(f"Starting stream generation with model: {model}")

        try:
            async with self.client.stream(
                "POST", "/api/generate", json=payload
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if line.strip():
                        try:
                            chunk = json.loads(line)
                            yield chunk
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to decode JSON chunk: {e}")
                            continue

        except httpx.HTTPError as e:
            logger.error(f"HTTP error during streaming: {e}")
            raise Exception(f"Ollama streaming failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during streaming: {e}")
            raise Exception(f"Streaming error: {str(e)}")

    async def stream_chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream chat completion from Ollama.

        Args:
            model: Model name
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Temperature for generation (0.0-2.0)

        Yields:
            Dictionary with streaming response chunks

        Raises:
            Exception: If streaming fails
        """
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
            },
        }

        logger.info(f"Starting stream chat with model: {model}, messages: {len(messages)}")

        try:
            async with self.client.stream(
                "POST", "/api/chat", json=payload
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if line.strip():
                        try:
                            chunk = json.loads(line)
                            yield chunk
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to decode JSON chunk: {e}")
                            continue

        except httpx.HTTPError as e:
            error_msg = str(e)
            logger.error(f"HTTP error during chat streaming: {e}")

            # Special handling for 404 - usually means model not found
            if "404" in error_msg:
                try:
                    models = await self.list_models()
                    model_names = [m.get("name", m.get("model", "unknown")) for m in models]
                    raise Exception(
                        f"Model '{model}' not found. Available models: {', '.join(model_names)}. "
                        f"Please select a valid model or pull the model using: ollama pull {model}"
                    )
                except Exception as list_error:
                    # If we can't list models, fall back to generic error
                    if "Model" in str(list_error):
                        raise list_error
                    raise Exception(
                        f"Model '{model}' not found. Please check if the model exists using: ollama list"
                    )

            raise Exception(f"Ollama chat streaming failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during chat streaming: {e}")
            raise Exception(f"Chat streaming error: {str(e)}")


# Singleton instance
_ollama_client: Optional[OllamaClient] = None


def get_ollama_client(base_url: Optional[str] = None) -> OllamaClient:
    """
    Get or create the Ollama client singleton.

    Args:
        base_url: Optional base URL override

    Returns:
        OllamaClient instance
    """
    global _ollama_client
    if _ollama_client is None or base_url is not None:
        _ollama_client = OllamaClient(base_url)
    return _ollama_client
