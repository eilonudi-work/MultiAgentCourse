"""Chat streaming routes with Server-Sent Events (SSE) support."""
import logging
import json
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.user import User
from app.models.conversation import Conversation
from app.models.message import Message
from app.middleware.auth import require_auth
from app.schemas.chat import ChatStreamRequest, MessageSearchRequest, MessageSearchResponse
from app.services.ollama_client import get_ollama_client

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


async def stream_chat_response(
    request: ChatStreamRequest,
    user: User,
    db: Session,
):
    """
    Generate streaming chat response using Server-Sent Events (SSE).

    Args:
        request: Chat stream request
        user: Authenticated user
        db: Database session

    Yields:
        SSE formatted messages with chat tokens
    """
    conversation = None
    user_message = None
    assistant_message_content = ""
    assistant_message = None

    try:
        # Get or create conversation
        if request.conversation_id:
            # Load existing conversation
            conversation = (
                db.query(Conversation)
                .filter(
                    Conversation.id == request.conversation_id,
                    Conversation.user_id == user.id,
                )
                .first()
            )

            if not conversation:
                yield f"event: error\ndata: {json.dumps({'error': 'Conversation not found'})}\n\n"
                return

            model_name = conversation.model_name
            system_prompt = conversation.system_prompt
        else:
            # Create new conversation
            if not request.model_name:
                yield f"event: error\ndata: {json.dumps({'error': 'model_name is required for new conversations'})}\n\n"
                return

            conversation = Conversation(
                user_id=user.id,
                title=request.message[:50] + "..." if len(request.message) > 50 else request.message,
                model_name=request.model_name,
                system_prompt=request.system_prompt,
            )
            db.add(conversation)
            db.commit()
            db.refresh(conversation)

            model_name = request.model_name
            system_prompt = request.system_prompt

            # Send conversation created event
            yield f"event: conversation_created\ndata: {json.dumps({'conversation_id': conversation.id})}\n\n"

        # Save user message
        user_message = Message(
            conversation_id=conversation.id,
            role="user",
            content=request.message,
        )
        db.add(user_message)
        db.commit()
        db.refresh(user_message)

        # Send user message saved event
        yield f"event: message_created\ndata: {json.dumps({'message_id': user_message.id, 'role': 'user'})}\n\n"

        # Build message history for context
        messages = []

        # Add system prompt if exists
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Add conversation history (last 20 messages for context)
        previous_messages = (
            db.query(Message)
            .filter(Message.conversation_id == conversation.id)
            .order_by(Message.created_at.desc())
            .limit(20)
            .all()
        )

        # Reverse to get chronological order
        previous_messages = list(reversed(previous_messages))

        # Add previous messages to context
        for msg in previous_messages[:-1]:  # Exclude the user message we just added
            messages.append({"role": msg.role, "content": msg.content})

        # Add current user message
        messages.append({"role": "user", "content": request.message})

        # Get Ollama client
        ollama_client = get_ollama_client(user.ollama_url)

        # Check if Ollama is available
        if not await ollama_client.is_ollama_available():
            yield f"event: error\ndata: {json.dumps({'error': 'Ollama service is not available'})}\n\n"
            return

        # Stream response from Ollama
        logger.info(f"Starting stream for conversation {conversation.id} with model {model_name}")

        try:
            async for chunk in ollama_client.stream_chat(
                model=model_name,
                messages=messages,
                temperature=request.temperature,
            ):
                # Extract response token
                if "message" in chunk:
                    content = chunk["message"].get("content", "")
                    if content:
                        assistant_message_content += content
                        # Send token to client
                        yield f"event: token\ndata: {json.dumps({'content': content})}\n\n"

                # Check if streaming is complete
                if chunk.get("done", False):
                    # Extract token count if available
                    tokens_used = None
                    if "eval_count" in chunk:
                        tokens_used = chunk.get("eval_count")

                    # Save assistant message
                    assistant_message = Message(
                        conversation_id=conversation.id,
                        role="assistant",
                        content=assistant_message_content,
                        tokens_used=tokens_used,
                    )
                    db.add(assistant_message)

                    # Update conversation timestamp
                    conversation.updated_at = datetime.utcnow()
                    db.commit()
                    db.refresh(assistant_message)

                    # Send completion event
                    yield f"event: done\ndata: {json.dumps({{'message_id': assistant_message.id, 'tokens_used': tokens_used}})}\n\n"

                    logger.info(
                        f"Completed stream for conversation {conversation.id}, "
                        f"generated {len(assistant_message_content)} characters, "
                        f"tokens: {tokens_used}"
                    )

        except Exception as stream_error:
            logger.error(f"Error during streaming: {stream_error}")

            # If we have partial content, save it
            if assistant_message_content:
                assistant_message = Message(
                    conversation_id=conversation.id,
                    role="assistant",
                    content=assistant_message_content + "\n\n[Stream interrupted]",
                )
                db.add(assistant_message)
                conversation.updated_at = datetime.utcnow()
                db.commit()

            yield f"event: error\ndata: {json.dumps({'error': str(stream_error)})}\n\n"

    except Exception as e:
        logger.error(f"Error in chat stream: {e}", exc_info=True)
        yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

        # Rollback any pending changes
        db.rollback()


@router.post("/stream")
async def stream_chat_post(
    request: ChatStreamRequest,
    req: Request,
    user: User = Depends(require_auth),
    db: Session = Depends(get_db),
):
    """
    Stream chat responses using Server-Sent Events (SSE) - POST version.

    This endpoint creates a new conversation or continues an existing one,
    streams the LLM response token-by-token, and saves messages to the database.

    Args:
        request: Chat stream request
        req: FastAPI request object
        user: Authenticated user
        db: Database session

    Returns:
        StreamingResponse with SSE events

    Event types:
        - conversation_created: Emitted when a new conversation is created
        - message_created: Emitted when user message is saved
        - token: Emitted for each response token
        - done: Emitted when streaming is complete
        - error: Emitted on errors
    """
    return StreamingResponse(
        stream_chat_response(request, user, db),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


@router.get("/stream")
async def stream_chat_get(
    conversation_id: Optional[int] = None,
    message: str = "",
    model_name: Optional[str] = None,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = 0.7,
    user: User = Depends(require_auth),
    db: Session = Depends(get_db),
):
    """
    Stream chat responses using Server-Sent Events (SSE) - GET version for EventSource.

    This endpoint creates a new conversation or continues an existing one,
    streams the LLM response token-by-token, and saves messages to the database.

    Args:
        conversation_id: Optional conversation ID
        message: User message
        model_name: Model to use
        system_prompt: Optional system prompt
        temperature: Temperature for generation
        user: Authenticated user
        db: Database session

    Returns:
        StreamingResponse with SSE events

    Event types:
        - conversation_created: Emitted when a new conversation is created
        - message_created: Emitted when user message is saved
        - token: Emitted for each response token
        - done: Emitted when streaming is complete
        - error: Emitted on errors
    """
    # Create ChatStreamRequest from query parameters
    request = ChatStreamRequest(
        conversation_id=conversation_id,
        message=message,
        model_name=model_name,
        system_prompt=system_prompt,
        temperature=temperature,
    )

    return StreamingResponse(
        stream_chat_response(request, user, db),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


@router.post("/search", response_model=list[MessageSearchResponse])
async def search_messages(
    search_request: MessageSearchRequest,
    user: User = Depends(require_auth),
    db: Session = Depends(get_db),
):
    """
    Search for messages across conversations or within a specific conversation.

    Args:
        search_request: Search parameters
        user: Authenticated user
        db: Database session

    Returns:
        List of matching messages with snippets

    Raises:
        HTTPException: If search fails
    """
    try:
        # Build base query for user's messages
        query = (
            db.query(Message, Conversation)
            .join(Conversation, Message.conversation_id == Conversation.id)
            .filter(Conversation.user_id == user.id)
        )

        # Filter by conversation if specified
        if search_request.conversation_id:
            query = query.filter(Message.conversation_id == search_request.conversation_id)

        # Search in message content (case-insensitive)
        query = query.filter(Message.content.ilike(f"%{search_request.query}%"))

        # Order by most recent first
        query = query.order_by(Message.created_at.desc())

        # Limit results
        results = query.limit(50).all()

        # Build response with snippets
        search_results = []
        for message, conversation in results:
            # Create snippet with context around the match
            content = message.content
            query_lower = search_request.query.lower()
            content_lower = content.lower()

            # Find query position
            pos = content_lower.find(query_lower)
            if pos >= 0:
                # Extract snippet with context
                start = max(0, pos - 50)
                end = min(len(content), pos + len(search_request.query) + 50)
                snippet = content[start:end]

                if start > 0:
                    snippet = "..." + snippet
                if end < len(content):
                    snippet = snippet + "..."
            else:
                # Fallback to first 100 chars
                snippet = content[:100] + ("..." if len(content) > 100 else "")

            search_results.append(
                MessageSearchResponse(
                    conversation_id=message.conversation_id,
                    message_id=message.id,
                    role=message.role,
                    content=message.content,
                    created_at=message.created_at.isoformat(),
                    snippet=snippet,
                )
            )

        logger.info(
            f"Search completed for user {user.id}: query='{search_request.query}', "
            f"results={len(search_results)}"
        )

        return search_results

    except Exception as e:
        logger.error(f"Error searching messages: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to search messages: {str(e)}",
        )
