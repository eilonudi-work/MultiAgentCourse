"""Custom exception classes for the application with comprehensive error handling."""
from typing import Any, Dict, Optional
from fastapi import HTTPException, status


class OllamaWebException(Exception):
    """Base exception for Ollama Web GUI with error code support."""

    def __init__(
        self,
        message: str,
        error_code: str = "UNKNOWN_ERROR",
        details: Optional[Dict[str, Any]] = None,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
    ):
        """
        Initialize exception with error details.

        Args:
            message: Human-readable error message
            error_code: Machine-readable error code
            details: Additional error details for debugging
            status_code: HTTP status code
        """
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.status_code = status_code
        super().__init__(message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for JSON response."""
        response = {
            "error": self.error_code,
            "message": self.message,
        }
        if self.details:
            response["details"] = self.details
        return response


class AuthenticationError(OllamaWebException):
    """Raised when authentication fails."""

    def __init__(
        self,
        message: str = "Authentication failed",
        error_code: str = "AUTH_FAILED",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            details=details,
            status_code=status.HTTP_401_UNAUTHORIZED,
        )


class InvalidAPIKeyError(AuthenticationError):
    """Raised when API key is invalid."""

    def __init__(self, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message="Invalid or missing API key",
            error_code="INVALID_API_KEY",
            details=details,
        )


class ExpiredAPIKeyError(AuthenticationError):
    """Raised when API key has expired."""

    def __init__(self, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message="API key has expired",
            error_code="EXPIRED_API_KEY",
            details=details,
        )


class SessionExpiredError(AuthenticationError):
    """Raised when session has expired."""

    def __init__(self, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message="Session has expired. Please login again.",
            error_code="SESSION_EXPIRED",
            details=details,
        )


class AuthorizationError(OllamaWebException):
    """Raised when user is not authorized to perform action."""

    def __init__(
        self,
        message: str = "You are not authorized to perform this action",
        error_code: str = "FORBIDDEN",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            details=details,
            status_code=status.HTTP_403_FORBIDDEN,
        )


class ResourceNotFoundError(OllamaWebException):
    """Raised when a requested resource is not found."""

    def __init__(
        self,
        resource_type: str,
        resource_id: Any,
        details: Optional[Dict[str, Any]] = None,
    ):
        message = f"{resource_type} with ID {resource_id} not found"
        super().__init__(
            message=message,
            error_code="RESOURCE_NOT_FOUND",
            details=details or {"resource_type": resource_type, "resource_id": str(resource_id)},
            status_code=status.HTTP_404_NOT_FOUND,
        )


class ModelNotFoundError(ResourceNotFoundError):
    """Raised when a requested Ollama model is not found."""

    def __init__(self, model_name: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            resource_type="Model",
            resource_id=model_name,
            details=details or {
                "model_name": model_name,
                "suggestion": "Check available models using /api/models endpoint",
            },
        )
        self.error_code = "MODEL_NOT_FOUND"


class ConversationNotFoundError(ResourceNotFoundError):
    """Raised when a requested conversation is not found."""

    def __init__(self, conversation_id: int, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            resource_type="Conversation",
            resource_id=conversation_id,
            details=details,
        )
        self.error_code = "CONVERSATION_NOT_FOUND"


class ValidationError(OllamaWebException):
    """Raised when input validation fails."""

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        error_details = details or {}
        if field:
            error_details["field"] = field
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            details=error_details,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        )


class OllamaConnectionError(OllamaWebException):
    """Raised when Ollama service is unreachable."""

    def __init__(
        self,
        message: str = "Unable to connect to Ollama service",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            error_code="OLLAMA_CONNECTION_ERROR",
            details=details or {
                "suggestion": "Please ensure Ollama is running and accessible"
            },
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )


class OllamaAPIError(OllamaWebException):
    """Raised when Ollama API returns an error."""

    def __init__(
        self,
        message: str,
        status_code: int = status.HTTP_502_BAD_GATEWAY,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            error_code="OLLAMA_API_ERROR",
            details=details,
            status_code=status_code,
        )


class DatabaseError(OllamaWebException):
    """Raised when database operations fail."""

    def __init__(
        self,
        message: str = "Database operation failed",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            error_code="DATABASE_ERROR",
            details=details,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


class ConfigurationError(OllamaWebException):
    """Raised when configuration is invalid."""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            details=details,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


class RateLimitError(OllamaWebException):
    """Raised when rate limit is exceeded."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        error_details = details or {}
        if retry_after:
            error_details["retry_after"] = retry_after
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_EXCEEDED",
            details=error_details,
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        )


class InputSanitizationError(ValidationError):
    """Raised when input contains potentially malicious content."""

    def __init__(
        self,
        message: str = "Input contains invalid or potentially malicious content",
        field: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            field=field,
            details=details,
        )
        self.error_code = "INPUT_SANITIZATION_ERROR"


class ExportError(OllamaWebException):
    """Raised when export operations fail."""

    def __init__(
        self,
        message: str = "Export operation failed",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            error_code="EXPORT_ERROR",
            details=details,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


class ImportError(OllamaWebException):
    """Raised when import operations fail."""

    def __init__(
        self,
        message: str = "Import operation failed",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            error_code="IMPORT_ERROR",
            details=details,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        )


# Error code documentation for API consumers
ERROR_CODES = {
    # Authentication & Authorization (4xx)
    "AUTH_FAILED": "General authentication failure",
    "INVALID_API_KEY": "API key is invalid or missing",
    "EXPIRED_API_KEY": "API key has expired and needs renewal",
    "SESSION_EXPIRED": "Session has expired, re-authentication required",
    "FORBIDDEN": "User lacks permission for this action",

    # Resource Errors (4xx)
    "RESOURCE_NOT_FOUND": "Requested resource does not exist",
    "MODEL_NOT_FOUND": "Ollama model not found or not available",
    "CONVERSATION_NOT_FOUND": "Conversation does not exist or was deleted",

    # Validation Errors (4xx)
    "VALIDATION_ERROR": "Input validation failed",
    "INPUT_SANITIZATION_ERROR": "Input contains invalid or malicious content",

    # Rate Limiting (4xx)
    "RATE_LIMIT_EXCEEDED": "Too many requests, please slow down",

    # External Service Errors (5xx)
    "OLLAMA_CONNECTION_ERROR": "Cannot connect to Ollama service",
    "OLLAMA_API_ERROR": "Ollama API returned an error",

    # Internal Errors (5xx)
    "DATABASE_ERROR": "Database operation failed",
    "CONFIGURATION_ERROR": "Server configuration is invalid",
    "EXPORT_ERROR": "Failed to export data",
    "IMPORT_ERROR": "Failed to import data",
    "UNKNOWN_ERROR": "An unexpected error occurred",
}


def get_error_description(error_code: str) -> str:
    """
    Get human-readable description for error code.

    Args:
        error_code: Error code to look up

    Returns:
        Error description or generic message
    """
    return ERROR_CODES.get(error_code, "An unexpected error occurred")
