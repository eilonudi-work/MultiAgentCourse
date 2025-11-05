"""Authentication middleware for API key validation."""
import logging
from typing import Optional
from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from app.models.user import User
from app.utils.auth import verify_api_key, hash_api_key
from app.database import SessionLocal
from app.config import settings

logger = logging.getLogger(__name__)

security = HTTPBearer(auto_error=False)


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = None,
) -> Optional[User]:
    """
    Get the current authenticated user from the API key.

    Args:
        request: FastAPI request object
        credentials: Optional HTTP bearer credentials

    Returns:
        User object if authenticated, None otherwise

    Raises:
        HTTPException: If authentication fails
    """
    # Extract API key from Authorization header
    api_key = None

    if credentials:
        api_key = credentials.credentials
    elif "authorization" in request.headers:
        auth_header = request.headers["authorization"]
        if auth_header.startswith("Bearer "):
            api_key = auth_header[7:]

    if not api_key:
        logger.warning("No API key provided in request")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key is required. Please provide it in the Authorization header.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Validate API key
    db: Session = SessionLocal()
    try:
        # Find user by API key hash
        users = db.query(User).all()
        for user in users:
            if verify_api_key(api_key, user.api_key_hash):
                logger.info(f"User authenticated: {user.id}")
                return user

        # No matching user found
        logger.warning("Invalid API key provided")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key. Please check your credentials.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during authentication: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication error. Please try again.",
        )
    finally:
        db.close()


async def require_auth(request: Request) -> User:
    """
    Dependency to require authentication for protected routes.

    Args:
        request: FastAPI request object

    Returns:
        Authenticated User object

    Raises:
        HTTPException: If authentication fails
    """
    # If authentication is disabled (development mode), get or create a default user
    if not settings.AUTH_REQUIRED:
        logger.info("Authentication disabled - using default user for development")
        db: Session = SessionLocal()
        try:
            # Get or create default user
            user = db.query(User).first()
            if not user:
                logger.info("Creating default development user")
                user = User(
                    api_key_hash=hash_api_key("dev-key-12345"),
                    ollama_url=settings.OLLAMA_URL
                )
                db.add(user)
                db.commit()
                db.refresh(user)
            return user
        finally:
            db.close()

    return await get_current_user(request)
