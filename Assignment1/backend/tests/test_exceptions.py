"""Tests for custom exception classes."""
import pytest
from app.utils.exceptions import (
    OllamaWebException,
    AuthenticationError,
    InvalidAPIKeyError,
    ResourceNotFoundError,
    ModelNotFoundError,
    ValidationError,
    RateLimitError,
    get_error_description,
)


def test_base_exception():
    """Test base OllamaWebException."""
    exc = OllamaWebException(
        message="Test error",
        error_code="TEST_ERROR",
        details={"key": "value"},
        status_code=500,
    )

    assert exc.message == "Test error"
    assert exc.error_code == "TEST_ERROR"
    assert exc.details == {"key": "value"}
    assert exc.status_code == 500

    # Test to_dict
    dict_repr = exc.to_dict()
    assert dict_repr["error"] == "TEST_ERROR"
    assert dict_repr["message"] == "Test error"
    assert dict_repr["details"] == {"key": "value"}


def test_authentication_error():
    """Test AuthenticationError."""
    exc = AuthenticationError()

    assert exc.message == "Authentication failed"
    assert exc.error_code == "AUTH_FAILED"
    assert exc.status_code == 401


def test_invalid_api_key_error():
    """Test InvalidAPIKeyError."""
    exc = InvalidAPIKeyError()

    assert exc.message == "Invalid or missing API key"
    assert exc.error_code == "INVALID_API_KEY"
    assert exc.status_code == 401


def test_resource_not_found_error():
    """Test ResourceNotFoundError."""
    exc = ResourceNotFoundError(
        resource_type="User",
        resource_id=123,
    )

    assert "User" in exc.message
    assert "123" in exc.message
    assert exc.error_code == "RESOURCE_NOT_FOUND"
    assert exc.status_code == 404
    assert exc.details["resource_type"] == "User"
    assert exc.details["resource_id"] == "123"


def test_model_not_found_error():
    """Test ModelNotFoundError."""
    exc = ModelNotFoundError(model_name="llama3")

    assert "llama3" in exc.message
    assert exc.error_code == "MODEL_NOT_FOUND"
    assert exc.status_code == 404
    assert exc.details["model_name"] == "llama3"
    assert "suggestion" in exc.details


def test_validation_error():
    """Test ValidationError."""
    exc = ValidationError(
        message="Invalid input",
        field="email",
        details={"expected": "email format"},
    )

    assert exc.message == "Invalid input"
    assert exc.error_code == "VALIDATION_ERROR"
    assert exc.status_code == 422
    assert exc.details["field"] == "email"
    assert exc.details["expected"] == "email format"


def test_rate_limit_error():
    """Test RateLimitError."""
    exc = RateLimitError(retry_after=60)

    assert exc.message == "Rate limit exceeded"
    assert exc.error_code == "RATE_LIMIT_EXCEEDED"
    assert exc.status_code == 429
    assert exc.details["retry_after"] == 60


def test_get_error_description():
    """Test error description lookup."""
    assert "authentication" in get_error_description("AUTH_FAILED").lower()
    assert "api key" in get_error_description("INVALID_API_KEY").lower()
    assert "model" in get_error_description("MODEL_NOT_FOUND").lower()
    assert "unexpected" in get_error_description("UNKNOWN_CODE").lower()
