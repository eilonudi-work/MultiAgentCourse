"""Ollama model management routes."""
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from app.models.user import User
from app.schemas.models import ModelsListResponse, ModelInfo
from app.services.ollama_client import get_ollama_client
from app.middleware.auth import require_auth

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/models", tags=["models"])

# In-memory cache for model lists (5 minute TTL)
_model_cache: Dict[str, Dict[str, Any]] = {}
CACHE_TTL_SECONDS = 300  # 5 minutes


def _get_cached_models(cache_key: str) -> Optional[ModelsListResponse]:
    """
    Get models from cache if available and not expired.

    Args:
        cache_key: Cache key (typically user's Ollama URL)

    Returns:
        Cached ModelsListResponse or None if not cached or expired
    """
    if cache_key in _model_cache:
        cached_data = _model_cache[cache_key]
        cached_at = cached_data.get("cached_at")
        models_response = cached_data.get("data")

        if cached_at and models_response:
            # Check if cache is still valid
            age = (datetime.utcnow() - cached_at).total_seconds()
            if age < CACHE_TTL_SECONDS:
                logger.info(f"Returning cached models (age: {age:.1f}s)")
                return models_response

    return None


def _set_cached_models(cache_key: str, models_response: ModelsListResponse):
    """
    Store models in cache.

    Args:
        cache_key: Cache key
        models_response: Models response to cache
    """
    _model_cache[cache_key] = {
        "cached_at": datetime.utcnow(),
        "data": models_response,
    }
    logger.info(f"Cached models list (TTL: {CACHE_TTL_SECONDS}s)")


@router.get("/list", response_model=ModelsListResponse)
async def list_models(current_user: User = Depends(require_auth)):
    """
    List all available Ollama models with caching.

    Models are cached for 5 minutes to reduce load on Ollama API.

    Args:
        current_user: Authenticated user

    Returns:
        ModelsListResponse with list of available models

    Raises:
        HTTPException: If Ollama is unreachable or request fails
    """
    try:
        cache_key = current_user.ollama_url

        # Check cache first
        cached_response = _get_cached_models(cache_key)
        if cached_response:
            return cached_response

        # Get Ollama client with user's configured URL
        ollama_client = get_ollama_client(current_user.ollama_url)

        # Test connection first
        is_available = await ollama_client.is_ollama_available()
        if not is_available:
            logger.error("Ollama service is not available")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Ollama service is not running or unreachable. Please ensure Ollama is started.",
            )

        # List models
        models_data = await ollama_client.list_models()

        # Convert to response schema
        models = [
            ModelInfo(
                name=model.get("name", ""),
                model=model.get("model", ""),
                size=model.get("size", 0),
                modified_at=model.get("modified_at", ""),
                digest=model.get("digest", ""),
                details=model.get("details"),
            )
            for model in models_data
        ]

        logger.info(f"Successfully listed {len(models)} models for user {current_user.id}")

        response = ModelsListResponse(models=models, count=len(models))

        # Cache the response
        _set_cached_models(cache_key, response)

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list models: {str(e)}",
        )


@router.get("/{model_name}/info")
async def get_model_info(
    model_name: str,
    current_user: User = Depends(require_auth),
):
    """
    Get detailed information about a specific model.

    Args:
        model_name: Name of the model
        current_user: Authenticated user

    Returns:
        Detailed model information

    Raises:
        HTTPException: If model not found or request fails
    """
    try:
        # Get Ollama client with user's configured URL
        ollama_client = get_ollama_client(current_user.ollama_url)

        # Get model info
        model_info = await ollama_client.get_model_info(model_name)

        if not model_info:
            logger.warning(f"Model {model_name} not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{model_name}' not found",
            )

        logger.info(f"Retrieved info for model {model_name} for user {current_user.id}")

        return model_info

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model info: {str(e)}",
        )


@router.post("/cache/clear")
async def clear_model_cache(current_user: User = Depends(require_auth)):
    """
    Clear the model list cache.

    Useful when models have been added or removed and cache needs refresh.

    Args:
        current_user: Authenticated user

    Returns:
        Success message
    """
    cache_key = current_user.ollama_url
    if cache_key in _model_cache:
        del _model_cache[cache_key]
        logger.info(f"Cleared model cache for user {current_user.id}")
        return {"success": True, "message": "Model cache cleared"}
    else:
        return {"success": True, "message": "No cache to clear"}
