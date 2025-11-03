"""Pydantic schemas for request/response models."""
from app.schemas.auth import SetupRequest, SetupResponse, VerifyRequest, VerifyResponse
from app.schemas.config import ConfigSaveRequest, ConfigSaveResponse, ConfigGetResponse
from app.schemas.models import ModelInfo, ModelsListResponse
from app.schemas.conversation import (
    ConversationCreate,
    ConversationUpdate,
    ConversationResponse,
    ConversationDetailResponse,
    ConversationListResponse,
    ConversationDeleteResponse,
    MessageResponse,
)
from app.schemas.chat import (
    ChatStreamRequest,
    ChatResponse,
    ChatMessage,
    MessageSearchRequest,
    MessageSearchResponse,
)
from app.schemas.prompts import PromptTemplate, PromptTemplateListResponse
from app.schemas.export import (
    ExportConversationSchema,
    ExportMessageSchema,
    ImportConversationRequest,
    ImportConversationResponse,
)

__all__ = [
    # Auth
    "SetupRequest",
    "SetupResponse",
    "VerifyRequest",
    "VerifyResponse",
    # Config
    "ConfigSaveRequest",
    "ConfigSaveResponse",
    "ConfigGetResponse",
    # Models
    "ModelInfo",
    "ModelsListResponse",
    # Conversations
    "ConversationCreate",
    "ConversationUpdate",
    "ConversationResponse",
    "ConversationDetailResponse",
    "ConversationListResponse",
    "ConversationDeleteResponse",
    "MessageResponse",
    # Chat
    "ChatStreamRequest",
    "ChatResponse",
    "ChatMessage",
    "MessageSearchRequest",
    "MessageSearchResponse",
    # Prompts
    "PromptTemplate",
    "PromptTemplateListResponse",
    # Export/Import
    "ExportConversationSchema",
    "ExportMessageSchema",
    "ImportConversationRequest",
    "ImportConversationResponse",
]
