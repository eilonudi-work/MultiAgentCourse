"""Authentication utilities for API key management."""
import bcrypt
import secrets
import logging

logger = logging.getLogger(__name__)


def hash_api_key(api_key: str) -> str:
    """
    Hash an API key using bcrypt.

    Args:
        api_key: Plain text API key

    Returns:
        Hashed API key
    """
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(api_key.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def verify_api_key(plain_key: str, hashed_key: str) -> bool:
    """
    Verify an API key against its hash.

    Args:
        plain_key: Plain text API key
        hashed_key: Hashed API key from database

    Returns:
        True if the key matches, False otherwise
    """
    try:
        return bcrypt.checkpw(plain_key.encode("utf-8"), hashed_key.encode("utf-8"))
    except Exception as e:
        logger.error(f"Error verifying API key: {e}")
        return False


def generate_api_key() -> str:
    """
    Generate a secure random API key.

    Returns:
        A randomly generated API key with prefix
    """
    random_part = secrets.token_urlsafe(32)
    return f"ollama_sk_{random_part}"


def mask_api_key(api_key: str) -> str:
    """
    Mask an API key for safe display.

    Args:
        api_key: Plain text API key

    Returns:
        Masked version showing only prefix and last 4 characters
    """
    if len(api_key) < 12:
        return "****"
    return f"{api_key[:10]}****{api_key[-4:]}"
