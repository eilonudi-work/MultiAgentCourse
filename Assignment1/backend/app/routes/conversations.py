"""Conversation management routes."""
import logging
import math
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from datetime import datetime

from app.database import get_db
from app.models.user import User
from app.models.conversation import Conversation
from app.models.message import Message
from app.middleware.auth import require_auth
from app.schemas.conversation import (
    ConversationCreate,
    ConversationUpdate,
    ConversationResponse,
    ConversationDetailResponse,
    ConversationListResponse,
    ConversationDeleteResponse,
    MessageResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.post("", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
async def create_conversation(
    conversation: ConversationCreate,
    user: User = Depends(require_auth),
    db: Session = Depends(get_db),
):
    """
    Create a new conversation.

    Args:
        conversation: Conversation creation data
        user: Authenticated user
        db: Database session

    Returns:
        Created conversation

    Raises:
        HTTPException: If creation fails
    """
    try:
        # Create new conversation
        new_conversation = Conversation(
            user_id=user.id,
            title=conversation.title or f"New Chat - {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}",
            model_name=conversation.model_name,
            system_prompt=conversation.system_prompt,
        )

        db.add(new_conversation)
        db.commit()
        db.refresh(new_conversation)

        logger.info(f"Created conversation {new_conversation.id} for user {user.id}")

        # Return conversation with message count
        response = ConversationResponse.model_validate(new_conversation)
        response.message_count = 0
        return response

    except Exception as e:
        logger.error(f"Error creating conversation: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create conversation: {str(e)}",
        )


@router.get("", response_model=ConversationListResponse)
async def list_conversations(
    user: User = Depends(require_auth),
    db: Session = Depends(get_db),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    search: Optional[str] = Query(None, description="Search in conversation titles"),
):
    """
    List all conversations for the authenticated user with pagination.

    Args:
        user: Authenticated user
        db: Database session
        page: Page number (1-indexed)
        page_size: Number of items per page
        search: Optional search query for filtering by title

    Returns:
        Paginated list of conversations

    Raises:
        HTTPException: If listing fails
    """
    try:
        # Build base query
        query = db.query(Conversation).filter(Conversation.user_id == user.id)

        # Add search filter if provided
        if search:
            query = query.filter(Conversation.title.ilike(f"%{search}%"))

        # Get total count
        total = query.count()

        # Calculate pagination
        total_pages = math.ceil(total / page_size)
        offset = (page - 1) * page_size

        # Get conversations ordered by updated_at (most recent first)
        conversations = (
            query.order_by(desc(Conversation.updated_at))
            .offset(offset)
            .limit(page_size)
            .all()
        )

        # Get message counts for each conversation
        conversation_responses = []
        for conv in conversations:
            message_count = db.query(func.count(Message.id)).filter(
                Message.conversation_id == conv.id
            ).scalar()

            conv_response = ConversationResponse.model_validate(conv)
            conv_response.message_count = message_count
            conversation_responses.append(conv_response)

        logger.info(
            f"Listed {len(conversations)} conversations for user {user.id} (page {page}/{total_pages})"
        )

        return ConversationListResponse(
            conversations=conversation_responses,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
        )

    except Exception as e:
        logger.error(f"Error listing conversations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list conversations: {str(e)}",
        )


@router.get("/{conversation_id}", response_model=ConversationDetailResponse)
async def get_conversation(
    conversation_id: int,
    user: User = Depends(require_auth),
    db: Session = Depends(get_db),
):
    """
    Get a single conversation with all its messages.

    Args:
        conversation_id: ID of the conversation
        user: Authenticated user
        db: Database session

    Returns:
        Conversation with messages

    Raises:
        HTTPException: If conversation not found or access denied
    """
    try:
        # Query conversation with messages
        conversation = (
            db.query(Conversation)
            .filter(Conversation.id == conversation_id, Conversation.user_id == user.id)
            .first()
        )

        if not conversation:
            logger.warning(
                f"Conversation {conversation_id} not found for user {user.id}"
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found",
            )

        # Convert messages to response schema
        message_responses = [
            MessageResponse.model_validate(msg) for msg in conversation.messages
        ]

        logger.info(
            f"Retrieved conversation {conversation_id} with {len(message_responses)} messages"
        )

        return ConversationDetailResponse(
            id=conversation.id,
            user_id=conversation.user_id,
            title=conversation.title,
            model_name=conversation.model_name,
            system_prompt=conversation.system_prompt,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
            messages=message_responses,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting conversation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get conversation: {str(e)}",
        )


@router.put("/{conversation_id}", response_model=ConversationResponse)
async def update_conversation(
    conversation_id: int,
    update_data: ConversationUpdate,
    user: User = Depends(require_auth),
    db: Session = Depends(get_db),
):
    """
    Update a conversation's title, model, or system prompt.

    Args:
        conversation_id: ID of the conversation
        update_data: Update data
        user: Authenticated user
        db: Database session

    Returns:
        Updated conversation

    Raises:
        HTTPException: If conversation not found or update fails
    """
    try:
        # Find conversation
        conversation = (
            db.query(Conversation)
            .filter(Conversation.id == conversation_id, Conversation.user_id == user.id)
            .first()
        )

        if not conversation:
            logger.warning(
                f"Conversation {conversation_id} not found for user {user.id}"
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found",
            )

        # Update fields if provided
        update_count = 0
        if update_data.title is not None:
            conversation.title = update_data.title
            update_count += 1

        if update_data.model_name is not None:
            conversation.model_name = update_data.model_name
            update_count += 1

        if update_data.system_prompt is not None:
            conversation.system_prompt = update_data.system_prompt
            update_count += 1

        if update_count == 0:
            logger.info(f"No updates provided for conversation {conversation_id}")
            response = ConversationResponse.model_validate(conversation)
            response.message_count = len(conversation.messages)
            return response

        # Update timestamp
        conversation.updated_at = datetime.utcnow()

        db.commit()
        db.refresh(conversation)

        logger.info(
            f"Updated conversation {conversation_id} ({update_count} fields changed)"
        )

        # Get message count
        message_count = db.query(func.count(Message.id)).filter(
            Message.conversation_id == conversation.id
        ).scalar()

        response = ConversationResponse.model_validate(conversation)
        response.message_count = message_count
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating conversation: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update conversation: {str(e)}",
        )


@router.delete("/{conversation_id}", response_model=ConversationDeleteResponse)
async def delete_conversation(
    conversation_id: int,
    user: User = Depends(require_auth),
    db: Session = Depends(get_db),
):
    """
    Delete a conversation and all its messages.

    Args:
        conversation_id: ID of the conversation
        user: Authenticated user
        db: Database session

    Returns:
        Deletion confirmation

    Raises:
        HTTPException: If conversation not found or deletion fails
    """
    try:
        # Find conversation
        conversation = (
            db.query(Conversation)
            .filter(Conversation.id == conversation_id, Conversation.user_id == user.id)
            .first()
        )

        if not conversation:
            logger.warning(
                f"Conversation {conversation_id} not found for user {user.id}"
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found",
            )

        # Delete conversation (cascade will delete messages)
        db.delete(conversation)
        db.commit()

        logger.info(f"Deleted conversation {conversation_id} for user {user.id}")

        return ConversationDeleteResponse(
            success=True,
            message="Conversation deleted successfully",
            conversation_id=conversation_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting conversation: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete conversation: {str(e)}",
        )
