"""Ollama model management routes."""
import logging
from fastapi import APIRouter, Depends, HTTPException, status
from app.models.user import User
from app.schemas.models import ModelsListResponse, ModelInfo
from app.services.ollama_client import get_ollama_client
from app.middleware.auth import require_auth

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/models", tags=["models"])


@router.get("/list", response_model=ModelsListResponse)
async def list_models(current_user: User = Depends(require_auth)):
    """
    List all available Ollama models.

    Args:
        current_user: Authenticated user

    Returns:
        ModelsListResponse with list of available models

    Raises:
        HTTPException: If Ollama is unreachable or request fails
    """
    try:
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

        return ModelsListResponse(models=models, count=len(models))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list models: {str(e)}",
        )
