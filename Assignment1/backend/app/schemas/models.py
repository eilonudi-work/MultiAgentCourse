"""Ollama model-related schemas."""
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


class ModelInfo(BaseModel):
    """Information about an Ollama model."""

    name: str
    model: str
    size: int
    modified_at: str
    digest: str
    details: Optional[dict] = None


class ModelsListResponse(BaseModel):
    """Response schema for models list endpoint."""

    models: List[ModelInfo]
    count: int
