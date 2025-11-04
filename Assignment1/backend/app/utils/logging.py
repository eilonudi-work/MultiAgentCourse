"""Logging configuration for the application with structured logging support."""
import logging
import logging.handlers
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Dict
from app.config import settings


class StructuredFormatter(logging.Formatter):
    """
    Custom formatter that outputs logs in JSON format for structured logging.

    Makes logs easier to parse and analyze with log aggregation tools.
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON.

        Args:
            record: Log record

        Returns:
            JSON-formatted log string
        """
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)

        # Add any additional attributes that were passed
        for key, value in record.__dict__.items():
            if key not in [
                "name",
                "msg",
                "args",
                "created",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "message",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "thread",
                "threadName",
                "exc_info",
                "exc_text",
                "stack_info",
                "extra_fields",
            ]:
                try:
                    # Only include JSON-serializable values
                    json.dumps(value)
                    log_data[key] = value
                except (TypeError, ValueError):
                    pass

        return json.dumps(log_data)


class RequestFormatter(logging.Formatter):
    """
    Custom formatter for request/response logging.

    Formats HTTP request and response details in a readable way.
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Format request log record.

        Args:
            record: Log record

        Returns:
            Formatted log string
        """
        # Base format
        base_format = super().format(record)

        # Add request details if present
        if hasattr(record, "request_id"):
            base_format = f"[{record.request_id}] {base_format}"

        return base_format


def setup_logging():
    """
    Configure logging for the application.

    Sets up console and file logging with appropriate formatting.
    Supports both traditional and structured (JSON) logging.
    """
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Configure log level
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    # Determine which formatter to use
    if settings.STRUCTURED_LOGGING:
        primary_formatter = StructuredFormatter()
    else:
        primary_formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    simple_formatter = logging.Formatter(
        fmt="%(levelname)s: %(message)s",
    )

    # Console handler (use simple format for readability)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(simple_formatter if not settings.STRUCTURED_LOGGING else primary_formatter)

    # Main application log file with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "app.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=10,
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(primary_formatter)

    # Error log file (only errors and above)
    error_handler = logging.handlers.RotatingFileHandler(
        log_dir / "error.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=10,
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(primary_formatter)

    # Access log file (for HTTP requests)
    access_handler = logging.handlers.RotatingFileHandler(
        log_dir / "access.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=10,
    )
    access_handler.setLevel(logging.INFO)
    access_handler.setFormatter(RequestFormatter(
        fmt="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_handler)

    # Configure access logger
    access_logger = logging.getLogger("access")
    access_logger.setLevel(logging.INFO)
    access_logger.addHandler(access_handler)
    access_logger.propagate = False

    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    logging.info("Logging configured successfully")
    if settings.STRUCTURED_LOGGING:
        logging.info("Structured (JSON) logging enabled")


def log_with_context(
    logger: logging.Logger,
    level: int,
    message: str,
    **context: Any
) -> None:
    """
    Log message with additional context.

    Args:
        logger: Logger instance
        level: Log level
        message: Log message
        **context: Additional context fields
    """
    extra = {"extra_fields": context}
    logger.log(level, message, extra=extra)
