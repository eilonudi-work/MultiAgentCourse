"""Export/Import related schemas."""
from pydantic import BaseModel, Field, validator, ConfigDict
from typing import List, Optional
from datetime import datetime


class ExportMessageSchema(BaseModel):
    """Schema for exported message."""

    role: str
    content: str
    tokens_used: Optional[int]
    created_at: str


class ExportConversationSchema(BaseModel):
    """Schema for exported conversation in JSON format."""

    model_config = ConfigDict(protected_namespaces=())

    id: int
    title: Optional[str]
    model_name: str
    system_prompt: Optional[str]
    created_at: str
    updated_at: str
    messages: List[ExportMessageSchema]
    export_version: str = "1.0"
    exported_at: str


class ImportConversationRequest(BaseModel):
    """Request schema for importing a conversation."""

    model_config = ConfigDict(protected_namespaces=())

    title: Optional[str] = Field(None, max_length=200)
    model_name: str = Field(..., min_length=1)
    system_prompt: Optional[str] = Field(None, max_length=4000)
    messages: List[ExportMessageSchema]

    @validator("messages")
    def validate_messages(cls, v):
        """Validate message list."""
        if not v or len(v) == 0:
            raise ValueError("Messages list cannot be empty")
        if len(v) > 1000:
            raise ValueError("Cannot import more than 1000 messages at once")
        return v

    @validator("messages")
    def validate_message_roles(cls, v):
        """Validate message roles."""
        for msg in v:
            if msg.role not in ["user", "assistant", "system"]:
                raise ValueError(f"Invalid role: {msg.role}")
        return v


class ImportConversationResponse(BaseModel):
    """Response schema for conversation import."""

    success: bool
    message: str
    conversation_id: Optional[int]
    imported_messages: int
