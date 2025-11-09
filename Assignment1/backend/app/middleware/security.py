"""Security middleware for CSRF protection and security headers."""
import secrets
import logging
from typing import Optional
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from app.utils.exceptions import AuthorizationError

logger = logging.getLogger(__name__)


# CSRF token storage (in production, use Redis or database)
_csrf_tokens = {}


def generate_csrf_token() -> str:
    """
    Generate a new CSRF token.

    Returns:
        CSRF token string
    """
    return secrets.token_urlsafe(32)


def store_csrf_token(session_id: str, token: str) -> None:
    """
    Store CSRF token for session.

    Args:
        session_id: Session identifier
        token: CSRF token
    """
    _csrf_tokens[session_id] = token


def validate_csrf_token(session_id: str, token: str) -> bool:
    """
    Validate CSRF token for session.

    Args:
        session_id: Session identifier
        token: CSRF token to validate

    Returns:
        True if valid, False otherwise
    """
    stored_token = _csrf_tokens.get(session_id)
    return stored_token is not None and secrets.compare_digest(stored_token, token)


def cleanup_csrf_tokens(max_age_seconds: int = 3600):
    """
    Clean up old CSRF tokens.

    In production, implement with Redis TTL or database cleanup.

    Args:
        max_age_seconds: Maximum age of tokens to keep
    """
    # This is a placeholder - in production use proper storage with TTL
    pass


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add security headers to responses.

    Implements best practices for web security headers.
    """

    async def dispatch(self, request: Request, call_next):
        """
        Add security headers to response.

        Args:
            request: FastAPI request
            call_next: Next middleware in chain

        Returns:
            Response with security headers
        """
        response = await call_next(request)

        # Content Security Policy
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self' data:; "
            "connect-src 'self' http://localhost:* ws://localhost:*; "
            "frame-ancestors 'none';"
        )

        # Prevent clickjacking
        response.headers["X-Frame-Options"] = "DENY"

        # Prevent MIME type sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"

        # Enable XSS protection
        response.headers["X-XSS-Protection"] = "1; mode=block"

        # Referrer policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Permissions policy
        response.headers["Permissions-Policy"] = (
            "geolocation=(), microphone=(), camera=(), payment=()"
        )

        # HSTS for HTTPS (only add if using HTTPS)
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains"
            )

        return response


class CSRFProtectionMiddleware(BaseHTTPMiddleware):
    """
    Middleware to protect against CSRF attacks.

    Validates CSRF tokens for state-changing operations.
    """

    # Methods that require CSRF protection
    PROTECTED_METHODS = ["POST", "PUT", "PATCH", "DELETE"]

    # Paths that don't require CSRF protection
    EXEMPT_PATHS = [
        "/api/auth/register",
        "/api/auth/login",
        "/api/auth/setup",
        "/api/auth/verify",
        "/health",
        "/docs",
        "/openapi.json",
        "/redoc",
    ]

    async def dispatch(self, request: Request, call_next):
        """
        Validate CSRF token for protected requests.

        Args:
            request: FastAPI request
            call_next: Next middleware in chain

        Returns:
            Response or CSRF error
        """
        # Check if request needs CSRF protection
        needs_protection = (
            request.method in self.PROTECTED_METHODS
            and not any(request.url.path.startswith(path) for path in self.EXEMPT_PATHS)
        )

        if needs_protection:
            # Get CSRF token from header
            csrf_token = request.headers.get("X-CSRF-Token")

            if not csrf_token:
                logger.warning(
                    f"Missing CSRF token for {request.method} {request.url.path}",
                    extra={"path": request.url.path, "method": request.method},
                )
                raise AuthorizationError(
                    message="CSRF token is required for this operation",
                    error_code="CSRF_TOKEN_MISSING",
                )

            # Get session ID from authorization header
            auth_header = request.headers.get("authorization", "")
            if auth_header.startswith("Bearer "):
                api_key = auth_header[7:]
                # Use API key hash as session ID
                import hashlib
                session_id = hashlib.sha256(api_key.encode()).hexdigest()[:16]

                # Validate CSRF token
                if not validate_csrf_token(session_id, csrf_token):
                    logger.warning(
                        f"Invalid CSRF token for {request.method} {request.url.path}",
                        extra={"path": request.url.path, "method": request.method},
                    )
                    raise AuthorizationError(
                        message="Invalid CSRF token",
                        error_code="CSRF_TOKEN_INVALID",
                    )

        # Process request
        response = await call_next(request)
        return response


class APIKeyRotationHelper:
    """
    Helper class for API key rotation.

    Manages graceful API key rotation with transition periods.
    """

    def __init__(self):
        """Initialize API key rotation helper."""
        # Storage: {user_id: {"current": key, "previous": key, "rotation_time": timestamp}}
        self.rotation_state = {}

    def initiate_rotation(self, user_id: int, new_key_hash: str) -> None:
        """
        Initiate key rotation for user.

        Args:
            user_id: User ID
            new_key_hash: New API key hash
        """
        import time

        if user_id in self.rotation_state:
            # Move current to previous
            current_key = self.rotation_state[user_id].get("current")
            self.rotation_state[user_id] = {
                "current": new_key_hash,
                "previous": current_key,
                "rotation_time": time.time(),
            }
        else:
            self.rotation_state[user_id] = {
                "current": new_key_hash,
                "previous": None,
                "rotation_time": time.time(),
            }

        logger.info(f"API key rotation initiated for user {user_id}")

    def is_valid_key(self, user_id: int, key_hash: str, grace_period_seconds: int = 3600) -> bool:
        """
        Check if key is valid (current or within grace period).

        Args:
            user_id: User ID
            key_hash: API key hash to validate
            grace_period_seconds: Grace period for old keys (default: 1 hour)

        Returns:
            True if key is valid, False otherwise
        """
        if user_id not in self.rotation_state:
            return False

        state = self.rotation_state[user_id]
        current_key = state.get("current")
        previous_key = state.get("previous")
        rotation_time = state.get("rotation_time", 0)

        # Check current key
        if current_key == key_hash:
            return True

        # Check previous key within grace period
        import time
        if previous_key and previous_key == key_hash:
            time_since_rotation = time.time() - rotation_time
            if time_since_rotation < grace_period_seconds:
                return True

        return False

    def cleanup_old_rotations(self, grace_period_seconds: int = 3600):
        """
        Clean up old rotation records after grace period.

        Args:
            grace_period_seconds: Grace period in seconds
        """
        import time
        now = time.time()

        expired_users = [
            user_id
            for user_id, state in self.rotation_state.items()
            if now - state.get("rotation_time", 0) > grace_period_seconds
        ]

        for user_id in expired_users:
            # Remove previous key
            if "previous" in self.rotation_state[user_id]:
                self.rotation_state[user_id]["previous"] = None

        logger.info(f"Cleaned up {len(expired_users)} expired key rotations")


# Global API key rotation helper
_api_key_rotation = APIKeyRotationHelper()


def get_api_key_rotation_helper() -> APIKeyRotationHelper:
    """
    Get the global API key rotation helper.

    Returns:
        APIKeyRotationHelper instance
    """
    return _api_key_rotation
