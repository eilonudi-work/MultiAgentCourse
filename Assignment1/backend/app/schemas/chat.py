"""Chat-related schemas."""
from pydantic import BaseModel, Field, validator
from typing import Optional, List


class ChatMessage(BaseModel):
    """Schema for a single chat message."""

    role: str = Field(..., description="Role: user, assistant, or system")
    content: str = Field(..., min_length=1, description="Message content")

    @validator("role")
    def validate_role(cls, v):
        """Validate message role."""
        if v not in ["user", "assistant", "system"]:
            raise ValueError("Role must be user, assistant, or system")
        return v


class ChatStreamRequest(BaseModel):
    """Request schema for streaming chat."""

    conversation_id: Optional[int] = Field(None, description="Existing conversation ID (optional for new chat)")
    message: str = Field(..., min_length=1, max_length=10000, description="User message")
    model_name: Optional[str] = Field(None, description="Model to use (required if no conversation_id)")
    system_prompt: Optional[str] = Field(None, max_length=4000, description="System prompt (for new conversations)")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Temperature for generation")
    stream: bool = Field(True, description="Whether to stream the response")

    @validator("message")
    def validate_message(cls, v):
        """Validate and sanitize message."""
        if not v.strip():
            raise ValueError("Message cannot be empty")
        return v.strip()

    @validator("system_prompt")
    def validate_system_prompt(cls, v):
        """Validate system prompt."""
        if v and len(v.strip()) == 0:
            return None
        return v


class ChatResponse(BaseModel):
    """Response schema for non-streaming chat."""

    conversation_id: int
    message_id: int
    role: str
    content: str
    tokens_used: Optional[int]
    model_name: str


class MessageSearchRequest(BaseModel):
    """Request schema for message search."""

    query: str = Field(..., min_length=1, description="Search query")
    conversation_id: Optional[int] = Field(None, description="Limit search to specific conversation")


class MessageSearchResponse(BaseModel):
    """Response schema for message search results."""

    conversation_id: int
    message_id: int
    role: str
    content: str
    created_at: str
    snippet: str
