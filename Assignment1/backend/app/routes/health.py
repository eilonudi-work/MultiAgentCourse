"""Health check and monitoring endpoints."""
import logging
from datetime import datetime
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database import get_db, engine
from app.config import settings
from app.services.ollama_client import get_ollama_client
from app.utils.metrics import get_metrics_collector
from app.utils.backup import DatabaseBackup

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check(db: Session = Depends(get_db)):
    """
    Comprehensive health check endpoint.

    Checks the health of:
    - API server
    - Database connection
    - Ollama service
    - File system (logs, backups)

    Returns:
        Health status with detailed component checks
    """
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "version": settings.VERSION,
        "service": settings.PROJECT_NAME,
        "checks": {},
    }

    # Check database
    try:
        # Execute a simple query
        db.execute("SELECT 1")
        health_status["checks"]["database"] = {
            "status": "healthy",
            "message": "Database connection successful",
        }
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        health_status["checks"]["database"] = {
            "status": "unhealthy",
            "message": f"Database connection failed: {str(e)}",
        }
        health_status["status"] = "degraded"

    # Check Ollama service
    try:
        ollama_client = get_ollama_client()
        is_available = await ollama_client.test_connection()
        if is_available:
            health_status["checks"]["ollama"] = {
                "status": "healthy",
                "message": "Ollama service is reachable",
                "url": settings.OLLAMA_URL,
            }
        else:
            health_status["checks"]["ollama"] = {
                "status": "unhealthy",
                "message": "Ollama service is not responding",
                "url": settings.OLLAMA_URL,
            }
            health_status["status"] = "degraded"
    except Exception as e:
        logger.error(f"Ollama health check failed: {e}")
        health_status["checks"]["ollama"] = {
            "status": "unhealthy",
            "message": f"Ollama connection error: {str(e)}",
            "url": settings.OLLAMA_URL,
        }
        health_status["status"] = "degraded"

    # Check file system
    try:
        # Check if logs directory is writable
        from pathlib import Path
        log_dir = Path("logs")
        if log_dir.exists() and log_dir.is_dir():
            # Try to create a test file
            test_file = log_dir / ".health_check"
            test_file.touch()
            test_file.unlink()
            health_status["checks"]["filesystem"] = {
                "status": "healthy",
                "message": "File system is writable",
            }
        else:
            health_status["checks"]["filesystem"] = {
                "status": "warning",
                "message": "Logs directory does not exist",
            }
    except Exception as e:
        logger.error(f"Filesystem health check failed: {e}")
        health_status["checks"]["filesystem"] = {
            "status": "unhealthy",
            "message": f"File system error: {str(e)}",
        }
        health_status["status"] = "degraded"

    return health_status


@router.get("/metrics")
async def get_metrics():
    """
    Get application performance metrics.

    Returns:
        Metrics summary including request counts, response times, and error rates
    """
    if not settings.METRICS_ENABLED:
        return {
            "message": "Metrics collection is disabled",
            "enabled": False,
        }

    collector = get_metrics_collector()

    return {
        "enabled": True,
        "summary": collector.get_summary(),
        "endpoints": collector.get_all_endpoints(),
    }


@router.get("/metrics/summary")
async def get_metrics_summary():
    """
    Get summarized metrics (lighter weight than full metrics).

    Returns:
        Summary metrics only
    """
    if not settings.METRICS_ENABLED:
        return {
            "message": "Metrics collection is disabled",
            "enabled": False,
        }

    collector = get_metrics_collector()

    return {
        "enabled": True,
        "summary": collector.get_summary(),
    }


@router.get("/backup/status")
async def get_backup_status():
    """
    Get database backup status.

    Returns:
        Backup statistics and configuration
    """
    if not settings.BACKUP_ENABLED:
        return {
            "message": "Backup is disabled",
            "enabled": False,
        }

    try:
        backup_manager = DatabaseBackup()
        stats = backup_manager.get_backup_stats()
        backups = backup_manager.list_backups()

        return {
            "enabled": True,
            "stats": stats,
            "recent_backups": backups[:5],  # Last 5 backups
            "retention_days": settings.BACKUP_RETENTION_DAYS,
        }
    except Exception as e:
        logger.error(f"Failed to get backup status: {e}")
        return {
            "enabled": True,
            "error": str(e),
        }


@router.get("/info")
async def get_info():
    """
    Get API information and configuration.

    Returns:
        API version, configuration, and feature flags
    """
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "api_prefix": settings.API_V1_PREFIX,
        "features": {
            "rate_limiting": settings.RATE_LIMIT_ENABLED,
            "csrf_protection": settings.CSRF_PROTECTION_ENABLED,
            "security_headers": settings.SECURITY_HEADERS_ENABLED,
            "metrics": settings.METRICS_ENABLED,
            "backup": settings.BACKUP_ENABLED,
            "structured_logging": settings.STRUCTURED_LOGGING,
        },
        "configuration": {
            "session_timeout_minutes": settings.SESSION_TIMEOUT_MINUTES,
            "backup_retention_days": settings.BACKUP_RETENTION_DAYS,
            "ollama_url": settings.OLLAMA_URL,
        },
        "docs": "/docs",
        "health": "/health",
    }
