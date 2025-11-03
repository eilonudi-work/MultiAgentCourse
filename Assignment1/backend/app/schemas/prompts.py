"""System prompt related schemas."""
from pydantic import BaseModel, Field
from typing import List


class PromptTemplate(BaseModel):
    """Schema for a system prompt template."""

    id: str = Field(..., description="Unique template identifier")
    name: str = Field(..., description="Template display name")
    description: str = Field(..., description="Template description")
    prompt: str = Field(..., description="The actual prompt text")
    category: str = Field(..., description="Template category")


class PromptTemplateListResponse(BaseModel):
    """Response schema for prompt template list."""

    templates: List[PromptTemplate]
    total: int
