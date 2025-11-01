"""Custom exception classes for the application."""


class OllamaWebException(Exception):
    """Base exception for Ollama Web GUI."""

    pass


class AuthenticationError(OllamaWebException):
    """Raised when authentication fails."""

    pass


class OllamaConnectionError(OllamaWebException):
    """Raised when Ollama service is unreachable."""

    pass


class ConfigurationError(OllamaWebException):
    """Raised when configuration is invalid."""

    pass


class DatabaseError(OllamaWebException):
    """Raised when database operations fail."""

    pass


class ModelNotFoundError(OllamaWebException):
    """Raised when a requested model is not found."""

    pass
