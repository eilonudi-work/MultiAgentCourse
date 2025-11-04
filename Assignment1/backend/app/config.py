"""Configuration management for the application."""
import os
from typing import List
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Settings(BaseModel):
    """Application settings."""

    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./ollama_web.db")

    # Ollama
    OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434")

    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-change-this-in-production")

    # CORS
    CORS_ORIGINS: List[str] = [
        origin.strip()
        for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
    ]

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # API Settings
    API_V1_PREFIX: str = "/api"
    PROJECT_NAME: str = "Ollama Web GUI API"
    VERSION: str = "1.0.0"

    # SQLite Settings
    SQLITE_WAL_MODE: bool = True

    # Session Settings
    SESSION_TIMEOUT_MINUTES: int = int(os.getenv("SESSION_TIMEOUT_MINUTES", "60"))
    API_KEY_EXPIRY_DAYS: int = int(os.getenv("API_KEY_EXPIRY_DAYS", "0"))  # 0 = never expires

    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "100"))

    # Security
    CSRF_PROTECTION_ENABLED: bool = os.getenv("CSRF_PROTECTION_ENABLED", "true").lower() == "true"
    SECURITY_HEADERS_ENABLED: bool = os.getenv("SECURITY_HEADERS_ENABLED", "true").lower() == "true"

    # Database Backup
    BACKUP_ENABLED: bool = os.getenv("BACKUP_ENABLED", "true").lower() == "true"
    BACKUP_DIRECTORY: str = os.getenv("BACKUP_DIRECTORY", "./backups")
    BACKUP_RETENTION_DAYS: int = int(os.getenv("BACKUP_RETENTION_DAYS", "30"))

    # Monitoring
    METRICS_ENABLED: bool = os.getenv("METRICS_ENABLED", "true").lower() == "true"
    STRUCTURED_LOGGING: bool = os.getenv("STRUCTURED_LOGGING", "false").lower() == "true"

    class Config:
        case_sensitive = True


settings = Settings()
