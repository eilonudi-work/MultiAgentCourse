"""Ollama API client with connection pooling and retry logic."""
import httpx
import logging
import asyncio
from typing import List, Dict, Any, Optional
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
            timeout=httpx.Timeout(30.0, connect=5.0),
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
            models = await self.list_models()
            for model in models:
                if model.get("name") == model_name or model.get("model") == model_name:
                    return model
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
