"""Authentication routes for API key management."""
import logging
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.user import User
from app.schemas.auth import (
    SetupRequest,
    SetupResponse,
    VerifyRequest,
    VerifyResponse,
)
from app.utils.auth import hash_api_key, verify_api_key
from app.services.ollama_client import get_ollama_client

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["authentication"])


@router.post("/setup", response_model=SetupResponse)
async def setup_api_key(request: SetupRequest, db: Session = Depends(get_db)):
    """
    Setup API key and Ollama URL for first-time users.

    This endpoint creates a new user with the provided API key and tests
    the connection to Ollama.

    Args:
        request: Setup request with API key and Ollama URL
        db: Database session

    Returns:
        SetupResponse with success status and user ID

    Raises:
        HTTPException: If setup fails or Ollama is unreachable
    """
    try:
        # Check if user already exists (only one user supported in MVP)
        existing_user = db.query(User).first()
        if existing_user:
            logger.warning("User already exists, updating existing user")
            # Update existing user
            existing_user.api_key_hash = hash_api_key(request.api_key)
            existing_user.ollama_url = request.ollama_url
            db.commit()
            db.refresh(existing_user)

            # Test Ollama connection
            ollama_client = get_ollama_client(request.ollama_url)
            is_connected = await ollama_client.test_connection()

            if not is_connected:
                logger.warning("Ollama connection test failed during setup")
                return SetupResponse(
                    success=False,
                    message="API key saved but Ollama connection failed. Please check if Ollama is running.",
                    user_id=existing_user.id,
                )

            return SetupResponse(
                success=True,
                message="API key updated successfully and Ollama connection verified.",
                user_id=existing_user.id,
            )

        # Create new user
        hashed_key = hash_api_key(request.api_key)
        new_user = User(api_key_hash=hashed_key, ollama_url=request.ollama_url)

        db.add(new_user)
        db.commit()
        db.refresh(new_user)

        logger.info(f"New user created with ID: {new_user.id}")

        # Test Ollama connection
        ollama_client = get_ollama_client(request.ollama_url)
        is_connected = await ollama_client.test_connection()

        if not is_connected:
            logger.warning("Ollama connection test failed during setup")
            return SetupResponse(
                success=False,
                message="API key saved but Ollama connection failed. Please check if Ollama is running.",
                user_id=new_user.id,
            )

        return SetupResponse(
            success=True,
            message="API key setup successful and Ollama connection verified.",
            user_id=new_user.id,
        )

    except Exception as e:
        logger.error(f"Error during API key setup: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to setup API key: {str(e)}",
        )


@router.post("/verify", response_model=VerifyResponse)
async def verify_api_key_endpoint(request: VerifyRequest, db: Session = Depends(get_db)):
    """
    Verify if an API key is valid.

    Args:
        request: Verification request with API key
        db: Database session

    Returns:
        VerifyResponse with validation status
    """
    try:
        # Check against all users (though we only have one in MVP)
        users = db.query(User).all()

        for user in users:
            if verify_api_key(request.api_key, user.api_key_hash):
                logger.info(f"API key verified for user {user.id}")
                return VerifyResponse(
                    valid=True,
                    message="API key is valid.",
                    user_id=user.id,
                )

        logger.warning("Invalid API key verification attempt")
        return VerifyResponse(
            valid=False,
            message="API key is invalid.",
            user_id=None,
        )

    except Exception as e:
        logger.error(f"Error during API key verification: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to verify API key: {str(e)}",
        )
