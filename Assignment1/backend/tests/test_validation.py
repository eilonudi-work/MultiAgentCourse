"""Tests for input validation and sanitization."""
import pytest
from app.utils.validation import (
    sanitize_string,
    sanitize_conversation_title,
    sanitize_message_content,
    sanitize_model_name,
    validate_temperature,
    validate_pagination_params,
    sanitize_export_format,
    sanitize_url,
)
from app.utils.exceptions import ValidationError, InputSanitizationError


class TestSanitizeString:
    """Tests for sanitize_string function."""

    def test_valid_string(self):
        """Test sanitizing valid string."""
        result = sanitize_string("  Hello World  ", "test")
        assert result == "Hello World"

    def test_sql_injection_pattern(self):
        """Test SQL injection detection."""
        with pytest.raises(InputSanitizationError) as exc_info:
            sanitize_string("SELECT * FROM users", "input")
        assert "SQL" in str(exc_info.value.message)

    def test_xss_pattern(self):
        """Test XSS pattern detection."""
        with pytest.raises(InputSanitizationError) as exc_info:
            sanitize_string("<script>alert('xss')</script>", "input")
        assert "JavaScript" in str(exc_info.value.message) or "HTML" in str(exc_info.value.message)

    def test_path_traversal(self):
        """Test path traversal detection."""
        with pytest.raises(InputSanitizationError) as exc_info:
            sanitize_string("../../etc/passwd", "input")
        assert "path traversal" in str(exc_info.value.message).lower()

    def test_max_length(self):
        """Test maximum length validation."""
        with pytest.raises(ValidationError) as exc_info:
            sanitize_string("a" * 1000, "input", max_length=100)
        assert "maximum length" in str(exc_info.value.message).lower()

    def test_non_string_input(self):
        """Test non-string input."""
        with pytest.raises(ValidationError) as exc_info:
            sanitize_string(123, "input")
        assert "must be a string" in str(exc_info.value.message).lower()


class TestSanitizeConversationTitle:
    """Tests for sanitize_conversation_title function."""

    def test_valid_title(self):
        """Test valid conversation title."""
        result = sanitize_conversation_title("My Conversation")
        assert result == "My Conversation"

    def test_empty_title(self):
        """Test empty title."""
        with pytest.raises(ValidationError) as exc_info:
            sanitize_conversation_title("  ")
        assert "cannot be empty" in str(exc_info.value.message).lower()

    def test_title_with_sql(self):
        """Test title with SQL injection."""
        with pytest.raises(InputSanitizationError):
            sanitize_conversation_title("My Title'; DROP TABLE users--")


class TestSanitizeModelName:
    """Tests for sanitize_model_name function."""

    def test_valid_model_name(self):
        """Test valid model names."""
        assert sanitize_model_name("llama2") == "llama2"
        assert sanitize_model_name("llama2:latest") == "llama2:latest"
        assert sanitize_model_name("codellama-7b") == "codellama-7b"

    def test_empty_model_name(self):
        """Test empty model name."""
        with pytest.raises(ValidationError) as exc_info:
            sanitize_model_name("")
        assert "cannot be empty" in str(exc_info.value.message).lower()

    def test_invalid_characters(self):
        """Test model name with invalid characters."""
        with pytest.raises(ValidationError) as exc_info:
            sanitize_model_name("llama2<script>")
        assert "invalid characters" in str(exc_info.value.message).lower()


class TestValidateTemperature:
    """Tests for validate_temperature function."""

    def test_valid_temperature(self):
        """Test valid temperature values."""
        assert validate_temperature(0.0) == 0.0
        assert validate_temperature(0.7) == 0.7
        assert validate_temperature(1.0) == 1.0
        assert validate_temperature(2.0) == 2.0

    def test_out_of_range(self):
        """Test temperature out of range."""
        with pytest.raises(ValidationError) as exc_info:
            validate_temperature(-0.1)
        assert "between 0.0 and 2.0" in str(exc_info.value.message)

        with pytest.raises(ValidationError) as exc_info:
            validate_temperature(2.1)
        assert "between 0.0 and 2.0" in str(exc_info.value.message)

    def test_non_numeric(self):
        """Test non-numeric temperature."""
        with pytest.raises(ValidationError) as exc_info:
            validate_temperature("hot")
        assert "must be a number" in str(exc_info.value.message).lower()


class TestValidatePaginationParams:
    """Tests for validate_pagination_params function."""

    def test_valid_params(self):
        """Test valid pagination parameters."""
        skip, limit = validate_pagination_params(0, 10)
        assert skip == 0
        assert limit == 10

    def test_negative_skip(self):
        """Test negative skip value."""
        with pytest.raises(ValidationError) as exc_info:
            validate_pagination_params(-1, 10)
        assert "non-negative" in str(exc_info.value.message).lower()

    def test_zero_limit(self):
        """Test zero limit value."""
        with pytest.raises(ValidationError) as exc_info:
            validate_pagination_params(0, 0)
        assert "positive" in str(exc_info.value.message).lower()

    def test_limit_exceeds_max(self):
        """Test limit exceeding maximum."""
        with pytest.raises(ValidationError) as exc_info:
            validate_pagination_params(0, 200, max_limit=100)
        assert "cannot exceed" in str(exc_info.value.message).lower()


class TestSanitizeExportFormat:
    """Tests for sanitize_export_format function."""

    def test_valid_formats(self):
        """Test valid export formats."""
        assert sanitize_export_format("json") == "json"
        assert sanitize_export_format("markdown") == "markdown"
        assert sanitize_export_format("txt") == "txt"
        assert sanitize_export_format("JSON") == "json"  # Case insensitive

    def test_invalid_format(self):
        """Test invalid export format."""
        with pytest.raises(ValidationError) as exc_info:
            sanitize_export_format("pdf")
        assert "invalid export format" in str(exc_info.value.message).lower()


class TestSanitizeUrl:
    """Tests for sanitize_url function."""

    def test_valid_urls(self):
        """Test valid URLs."""
        assert sanitize_url("http://localhost:11434") == "http://localhost:11434"
        assert sanitize_url("https://api.example.com") == "https://api.example.com"

    def test_invalid_url(self):
        """Test invalid URL."""
        with pytest.raises(ValidationError) as exc_info:
            sanitize_url("not-a-url")
        assert "invalid" in str(exc_info.value.message).lower()

    def test_url_with_path_traversal(self):
        """Test URL with path traversal."""
        with pytest.raises(InputSanitizationError):
            sanitize_url("http://localhost/../../etc/passwd")
