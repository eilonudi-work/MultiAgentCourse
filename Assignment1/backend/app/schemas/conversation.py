"""Conversation-related schemas."""
from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime


class ConversationCreate(BaseModel):
    """Request schema for creating a new conversation."""

    title: Optional[str] = Field(None, max_length=200, description="Conversation title")
    model_name: str = Field(..., min_length=1, description="Name of the LLM model to use")
    system_prompt: Optional[str] = Field(None, max_length=4000, description="System prompt for the conversation")

    @validator("system_prompt")
    def validate_system_prompt(cls, v):
        """Validate system prompt to prevent injection."""
        if v and len(v.strip()) == 0:
            return None
        return v


class ConversationUpdate(BaseModel):
    """Request schema for updating a conversation."""

    title: Optional[str] = Field(None, max_length=200, description="Updated conversation title")
    model_name: Optional[str] = Field(None, min_length=1, description="Updated model name")
    system_prompt: Optional[str] = Field(None, max_length=4000, description="Updated system prompt")

    @validator("system_prompt")
    def validate_system_prompt(cls, v):
        """Validate system prompt to prevent injection."""
        if v and len(v.strip()) == 0:
            return None
        return v


class MessageResponse(BaseModel):
    """Response schema for a message."""

    id: int
    conversation_id: int
    role: str
    content: str
    tokens_used: Optional[int]
    created_at: datetime

    class Config:
        from_attributes = True


class ConversationResponse(BaseModel):
    """Response schema for a conversation without messages."""

    id: int
    user_id: int
    title: Optional[str]
    model_name: str
    system_prompt: Optional[str]
    created_at: datetime
    updated_at: datetime
    message_count: Optional[int] = 0

    class Config:
        from_attributes = True


class ConversationDetailResponse(BaseModel):
    """Response schema for a conversation with messages."""

    id: int
    user_id: int
    title: Optional[str]
    model_name: str
    system_prompt: Optional[str]
    created_at: datetime
    updated_at: datetime
    messages: List[MessageResponse] = []

    class Config:
        from_attributes = True


class ConversationListResponse(BaseModel):
    """Response schema for paginated conversation list."""

    conversations: List[ConversationResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class ConversationDeleteResponse(BaseModel):
    """Response schema for conversation deletion."""

    success: bool
    message: str
    conversation_id: int
