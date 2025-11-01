"""Configuration-related schemas."""
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any


class ConfigSaveRequest(BaseModel):
    """Request schema for saving configuration."""

    ollama_url: Optional[str] = Field(None, description="Ollama API URL")
    settings: Optional[Dict[str, Any]] = Field(None, description="Additional settings")

    @validator("ollama_url")
    def validate_ollama_url(cls, v):
        """Validate Ollama URL format if provided."""
        if v and not v.startswith(("http://", "https://")):
            raise ValueError("Ollama URL must start with http:// or https://")
        return v.rstrip("/") if v else v


class ConfigSaveResponse(BaseModel):
    """Response schema for config save endpoint."""

    success: bool
    message: str


class ConfigGetResponse(BaseModel):
    """Response schema for config retrieval."""

    ollama_url: str
    settings: Dict[str, Any]
    user_id: int
