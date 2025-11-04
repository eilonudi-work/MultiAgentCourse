"""Input validation and sanitization utilities."""
import re
from typing import Any, Dict, List, Optional
from app.utils.exceptions import InputSanitizationError, ValidationError


# Potentially dangerous patterns
SQL_INJECTION_PATTERNS = [
    r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE)\b)",
    r"(--|#|/\*|\*/)",
    r"(\bOR\b.*=.*|1=1|'=')",
]

XSS_PATTERNS = [
    r"<script[^>]*>.*?</script>",
    r"javascript:",
    r"on\w+\s*=",
    r"<iframe",
    r"<embed",
    r"<object",
]

PATH_TRAVERSAL_PATTERNS = [
    r"\.\./",
    r"\.\.",
    r"%2e%2e",
    r"\.\.\\",
]


def sanitize_string(
    value: str,
    field_name: str = "input",
    max_length: Optional[int] = None,
    allow_html: bool = False,
) -> str:
    """
    Sanitize string input to prevent injection attacks.

    Args:
        value: String to sanitize
        field_name: Name of the field (for error messages)
        max_length: Maximum allowed length
        allow_html: Whether to allow HTML tags (default: False)

    Returns:
        Sanitized string

    Raises:
        InputSanitizationError: If input contains malicious content
        ValidationError: If input is invalid
    """
    if not isinstance(value, str):
        raise ValidationError(
            message=f"{field_name} must be a string",
            field=field_name,
        )

    # Check length
    if max_length and len(value) > max_length:
        raise ValidationError(
            message=f"{field_name} exceeds maximum length of {max_length}",
            field=field_name,
            details={"max_length": max_length, "actual_length": len(value)},
        )

    # Check for SQL injection patterns
    for pattern in SQL_INJECTION_PATTERNS:
        if re.search(pattern, value, re.IGNORECASE):
            raise InputSanitizationError(
                message=f"{field_name} contains potentially malicious SQL patterns",
                field=field_name,
                details={"pattern": "SQL_INJECTION"},
            )

    # Check for XSS patterns if HTML is not allowed
    if not allow_html:
        for pattern in XSS_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                raise InputSanitizationError(
                    message=f"{field_name} contains potentially malicious HTML/JavaScript",
                    field=field_name,
                    details={"pattern": "XSS"},
                )

    # Check for path traversal
    for pattern in PATH_TRAVERSAL_PATTERNS:
        if re.search(pattern, value, re.IGNORECASE):
            raise InputSanitizationError(
                message=f"{field_name} contains path traversal patterns",
                field=field_name,
                details={"pattern": "PATH_TRAVERSAL"},
            )

    # Strip leading/trailing whitespace
    return value.strip()


def sanitize_conversation_title(title: str) -> str:
    """
    Sanitize conversation title.

    Args:
        title: Conversation title

    Returns:
        Sanitized title

    Raises:
        ValidationError: If title is invalid
    """
    title = sanitize_string(title, field_name="title", max_length=200)

    if not title:
        raise ValidationError(
            message="Conversation title cannot be empty",
            field="title",
        )

    return title


def sanitize_message_content(content: str) -> str:
    """
    Sanitize message content.

    Args:
        content: Message content

    Returns:
        Sanitized content

    Raises:
        ValidationError: If content is invalid
    """
    content = sanitize_string(content, field_name="content", max_length=100000)

    if not content:
        raise ValidationError(
            message="Message content cannot be empty",
            field="content",
        )

    return content


def sanitize_model_name(model_name: str) -> str:
    """
    Sanitize model name.

    Args:
        model_name: Model name

    Returns:
        Sanitized model name

    Raises:
        ValidationError: If model name is invalid
    """
    model_name = sanitize_string(model_name, field_name="model", max_length=100)

    if not model_name:
        raise ValidationError(
            message="Model name cannot be empty",
            field="model",
        )

    # Model names should only contain alphanumeric, dash, underscore, colon, and dot
    if not re.match(r"^[a-zA-Z0-9\-_:.]+$", model_name):
        raise ValidationError(
            message="Model name contains invalid characters",
            field="model",
            details={
                "allowed_characters": "alphanumeric, dash, underscore, colon, and dot"
            },
        )

    return model_name


def sanitize_system_prompt(prompt: str) -> str:
    """
    Sanitize system prompt.

    Args:
        prompt: System prompt

    Returns:
        Sanitized prompt

    Raises:
        ValidationError: If prompt is invalid
    """
    return sanitize_string(prompt, field_name="system_prompt", max_length=10000)


def validate_temperature(temperature: float) -> float:
    """
    Validate temperature parameter.

    Args:
        temperature: Temperature value

    Returns:
        Validated temperature

    Raises:
        ValidationError: If temperature is invalid
    """
    if not isinstance(temperature, (int, float)):
        raise ValidationError(
            message="Temperature must be a number",
            field="temperature",
        )

    if not 0.0 <= temperature <= 2.0:
        raise ValidationError(
            message="Temperature must be between 0.0 and 2.0",
            field="temperature",
            details={"min": 0.0, "max": 2.0, "provided": temperature},
        )

    return float(temperature)


def validate_pagination_params(
    skip: int = 0, limit: int = 50, max_limit: int = 100
) -> tuple[int, int]:
    """
    Validate pagination parameters.

    Args:
        skip: Number of items to skip
        limit: Maximum number of items to return
        max_limit: Maximum allowed limit

    Returns:
        Tuple of (skip, limit)

    Raises:
        ValidationError: If parameters are invalid
    """
    if not isinstance(skip, int) or skip < 0:
        raise ValidationError(
            message="Skip parameter must be a non-negative integer",
            field="skip",
        )

    if not isinstance(limit, int) or limit <= 0:
        raise ValidationError(
            message="Limit parameter must be a positive integer",
            field="limit",
        )

    if limit > max_limit:
        raise ValidationError(
            message=f"Limit cannot exceed {max_limit}",
            field="limit",
            details={"max_limit": max_limit, "provided": limit},
        )

    return skip, limit


def sanitize_export_format(format_type: str) -> str:
    """
    Validate and sanitize export format.

    Args:
        format_type: Export format (json, markdown, txt)

    Returns:
        Sanitized format

    Raises:
        ValidationError: If format is invalid
    """
    format_type = format_type.lower().strip()

    allowed_formats = ["json", "markdown", "txt"]
    if format_type not in allowed_formats:
        raise ValidationError(
            message=f"Invalid export format. Allowed formats: {', '.join(allowed_formats)}",
            field="format",
            details={"allowed_formats": allowed_formats},
        )

    return format_type


def sanitize_url(url: str, field_name: str = "url") -> str:
    """
    Sanitize and validate URL.

    Args:
        url: URL to sanitize
        field_name: Field name for error messages

    Returns:
        Sanitized URL

    Raises:
        ValidationError: If URL is invalid
    """
    url = sanitize_string(url, field_name=field_name, max_length=2048)

    # Basic URL validation
    url_pattern = r"^https?://[^\s/$.?#].[^\s]*$"
    if not re.match(url_pattern, url, re.IGNORECASE):
        raise ValidationError(
            message=f"Invalid {field_name} format",
            field=field_name,
            details={"expected_format": "http:// or https://"},
        )

    return url


def sanitize_dict_keys(data: Dict[str, Any], allowed_keys: List[str]) -> Dict[str, Any]:
    """
    Filter dictionary to only include allowed keys.

    Args:
        data: Dictionary to filter
        allowed_keys: List of allowed keys

    Returns:
        Filtered dictionary
    """
    return {k: v for k, v in data.items() if k in allowed_keys}
