"""Configuration management routes."""
import logging
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.user import User
from app.models.setting import Setting
from app.schemas.config import ConfigSaveRequest, ConfigSaveResponse, ConfigGetResponse
from app.middleware.auth import require_auth

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/config", tags=["configuration"])


@router.post("/save", response_model=ConfigSaveResponse)
async def save_config(
    request: ConfigSaveRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_auth),
):
    """
    Save user configuration (Ollama URL and settings).

    Args:
        request: Configuration save request
        db: Database session
        current_user: Authenticated user

    Returns:
        ConfigSaveResponse with success status

    Raises:
        HTTPException: If save fails
    """
    try:
        # Update Ollama URL if provided
        if request.ollama_url:
            current_user.ollama_url = request.ollama_url
            logger.info(f"Updated Ollama URL for user {current_user.id}: {request.ollama_url}")

        # Save additional settings
        if request.settings:
            for key, value in request.settings.items():
                # Check if setting exists
                setting = (
                    db.query(Setting)
                    .filter(Setting.user_id == current_user.id, Setting.key == key)
                    .first()
                )

                if setting:
                    setting.value = str(value)
                    logger.info(f"Updated setting {key} for user {current_user.id}")
                else:
                    new_setting = Setting(
                        user_id=current_user.id, key=key, value=str(value)
                    )
                    db.add(new_setting)
                    logger.info(f"Created setting {key} for user {current_user.id}")

        db.commit()

        return ConfigSaveResponse(
            success=True,
            message="Configuration saved successfully.",
        )

    except Exception as e:
        logger.error(f"Error saving configuration: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save configuration: {str(e)}",
        )


@router.get("/get", response_model=ConfigGetResponse)
async def get_config(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_auth),
):
    """
    Retrieve user configuration.

    Args:
        db: Database session
        current_user: Authenticated user

    Returns:
        ConfigGetResponse with user configuration

    Raises:
        HTTPException: If retrieval fails
    """
    try:
        # Get all settings for user
        settings = (
            db.query(Setting)
            .filter(Setting.user_id == current_user.id)
            .all()
        )

        settings_dict = {s.key: s.value for s in settings}

        logger.info(f"Retrieved configuration for user {current_user.id}")

        return ConfigGetResponse(
            ollama_url=current_user.ollama_url,
            settings=settings_dict,
            user_id=current_user.id,
        )

    except Exception as e:
        logger.error(f"Error retrieving configuration: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve configuration: {str(e)}",
        )
