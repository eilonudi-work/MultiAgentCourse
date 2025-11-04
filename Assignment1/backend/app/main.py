"""Main FastAPI application entry point with Phase 3 security and monitoring."""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.database import init_db
from app.utils.logging import setup_logging
from app.utils.migrations import run_migrations
from app.utils.backup import create_backup_job
from app.middleware.error_handler import register_error_handlers
from app.middleware.rate_limiter import RateLimitMiddleware
from app.middleware.security import SecurityHeadersMiddleware, CSRFProtectionMiddleware
from app.utils.metrics import MetricsMiddleware
from app.routes import (
    auth,
    config,
    models,
    conversations,
    chat,
    prompts,
    export,
    health,
)

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.

    Args:
        app: FastAPI application instance
    """
    # Startup
    logger.info("=" * 60)
    logger.info("Starting Ollama Web GUI Backend API")
    logger.info(f"Version: {settings.VERSION}")
    logger.info(f"Ollama URL: {settings.OLLAMA_URL}")
    logger.info(f"Database: {settings.DATABASE_URL}")
    logger.info("=" * 60)

    # Initialize database
    init_db()

    # Run database migrations
    try:
        applied = run_migrations()
        logger.info(f"Database migrations completed ({applied} applied)")
    except Exception as e:
        logger.error(f"Migration failed: {e}", exc_info=True)
        # Don't fail startup, just log the error

    # Create initial backup if enabled
    if settings.BACKUP_ENABLED:
        try:
            backup_path = create_backup_job()
            if backup_path:
                logger.info(f"Initial backup created: {backup_path}")
        except Exception as e:
            logger.warning(f"Initial backup failed: {e}")

    # Log security features
    logger.info("Security Features:")
    logger.info(f"  - Rate Limiting: {settings.RATE_LIMIT_ENABLED}")
    logger.info(f"  - CSRF Protection: {settings.CSRF_PROTECTION_ENABLED}")
    logger.info(f"  - Security Headers: {settings.SECURITY_HEADERS_ENABLED}")
    logger.info(f"  - Metrics Collection: {settings.METRICS_ENABLED}")
    logger.info(f"  - Session Timeout: {settings.SESSION_TIMEOUT_MINUTES} minutes")

    logger.info("Application startup complete")

    yield

    # Shutdown
    logger.info("Shutting down Ollama Web GUI Backend API")

    # Create shutdown backup if enabled
    if settings.BACKUP_ENABLED:
        try:
            backup_path = create_backup_job()
            if backup_path:
                logger.info(f"Shutdown backup created: {backup_path}")
        except Exception as e:
            logger.warning(f"Shutdown backup failed: {e}")

    logger.info("Shutdown complete")


# Create FastAPI application
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Backend API for Ollama Web GUI - A ChatGPT-like interface for local LLMs with production-ready security",
    lifespan=lifespan,
)

# Register error handlers first (before other middleware)
register_error_handlers(app)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add security headers middleware
if settings.SECURITY_HEADERS_ENABLED:
    app.add_middleware(SecurityHeadersMiddleware)
    logger.info("Security headers middleware enabled")

# Add CSRF protection middleware
if settings.CSRF_PROTECTION_ENABLED:
    app.add_middleware(CSRFProtectionMiddleware)
    logger.info("CSRF protection middleware enabled")

# Add rate limiting middleware
if settings.RATE_LIMIT_ENABLED:
    app.add_middleware(RateLimitMiddleware)
    logger.info("Rate limiting middleware enabled")

# Add metrics middleware
if settings.METRICS_ENABLED:
    app.add_middleware(MetricsMiddleware)
    logger.info("Metrics collection middleware enabled")


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Middleware to log all HTTP requests.

    Args:
        request: FastAPI request
        call_next: Next middleware in chain

    Returns:
        Response from the next middleware
    """
    # Get access logger for HTTP requests
    access_logger = logging.getLogger("access")

    # Log request
    access_logger.info(
        f"{request.method} {request.url.path}",
        extra={
            "method": request.method,
            "path": request.url.path,
            "client_ip": request.client.host if request.client else "unknown",
        },
    )

    # Don't log API keys
    if "authorization" in request.headers:
        logger.debug("Authorization header present (not logged for security)")

    response = await call_next(request)

    # Log response
    access_logger.info(
        f"{request.method} {request.url.path} - {response.status_code}",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
        },
    )

    return response


# Include routers
app.include_router(health.router)  # Health/monitoring endpoints (no prefix)
app.include_router(auth.router, prefix=settings.API_V1_PREFIX)
app.include_router(config.router, prefix=settings.API_V1_PREFIX)
app.include_router(models.router, prefix=settings.API_V1_PREFIX)
app.include_router(conversations.router, prefix=settings.API_V1_PREFIX)
app.include_router(chat.router, prefix=settings.API_V1_PREFIX)
app.include_router(prompts.router, prefix=settings.API_V1_PREFIX)
app.include_router(export.router, prefix=settings.API_V1_PREFIX)


# Root endpoint
@app.get("/", tags=["root"])
async def root():
    """
    Root endpoint with API information.

    Returns:
        API information
    """
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "docs": "/docs",
        "health": "/health",
        "api": settings.API_V1_PREFIX,
        "features": {
            "rate_limiting": settings.RATE_LIMIT_ENABLED,
            "csrf_protection": settings.CSRF_PROTECTION_ENABLED,
            "security_headers": settings.SECURITY_HEADERS_ENABLED,
            "metrics": settings.METRICS_ENABLED,
            "backup": settings.BACKUP_ENABLED,
        },
    }


logger.info("FastAPI application initialized with Phase 3 security features")
