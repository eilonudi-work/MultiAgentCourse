"""Authentication-related schemas."""
from pydantic import BaseModel, Field, validator
from typing import Optional


class SetupRequest(BaseModel):
    """Request schema for initial API key setup."""

    api_key: str = Field(..., min_length=10, description="API key for Ollama access")
    ollama_url: Optional[str] = Field(
        "http://localhost:11434",
        description="URL for Ollama API endpoint"
    )

    @validator("ollama_url")
    def validate_ollama_url(cls, v):
        """Validate Ollama URL format."""
        if not v.startswith(("http://", "https://")):
            raise ValueError("Ollama URL must start with http:// or https://")
        return v.rstrip("/")


class SetupResponse(BaseModel):
    """Response schema for setup endpoint."""

    success: bool
    message: str
    user_id: Optional[int] = None


class VerifyRequest(BaseModel):
    """Request schema for API key verification."""

    api_key: str = Field(..., min_length=10, description="API key to verify")


class VerifyResponse(BaseModel):
    """Response schema for verification endpoint."""

    valid: bool
    message: str
    user_id: Optional[int] = None
