"""Export and import routes for conversations."""
import logging
import json
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status, Response
from fastapi.responses import JSONResponse, PlainTextResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.user import User
from app.models.conversation import Conversation
from app.models.message import Message
from app.middleware.auth import require_auth
from app.schemas.export import (
    ExportConversationSchema,
    ExportMessageSchema,
    ImportConversationRequest,
    ImportConversationResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/export", tags=["export"])


@router.get("/conversations/{conversation_id}/json")
async def export_conversation_json(
    conversation_id: int,
    user: User = Depends(require_auth),
    db: Session = Depends(get_db),
):
    """
    Export a conversation to JSON format.

    Args:
        conversation_id: ID of the conversation to export
        user: Authenticated user
        db: Database session

    Returns:
        JSON export of the conversation

    Raises:
        HTTPException: If conversation not found or export fails
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
                f"Conversation {conversation_id} not found for export by user {user.id}"
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found",
            )

        # Build export data
        messages = [
            ExportMessageSchema(
                role=msg.role,
                content=msg.content,
                tokens_used=msg.tokens_used,
                created_at=msg.created_at.isoformat(),
            )
            for msg in conversation.messages
        ]

        export_data = ExportConversationSchema(
            id=conversation.id,
            title=conversation.title,
            model_name=conversation.model_name,
            system_prompt=conversation.system_prompt,
            created_at=conversation.created_at.isoformat(),
            updated_at=conversation.updated_at.isoformat(),
            messages=messages,
            exported_at=datetime.utcnow().isoformat(),
        )

        logger.info(
            f"Exported conversation {conversation_id} to JSON for user {user.id} "
            f"({len(messages)} messages)"
        )

        # Return as downloadable JSON file
        json_content = export_data.model_dump_json(indent=2)

        return Response(
            content=json_content,
            media_type="application/json",
            headers={
                "Content-Disposition": f'attachment; filename="conversation_{conversation_id}.json"'
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting conversation to JSON: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export conversation: {str(e)}",
        )


@router.get("/conversations/{conversation_id}/markdown")
async def export_conversation_markdown(
    conversation_id: int,
    user: User = Depends(require_auth),
    db: Session = Depends(get_db),
):
    """
    Export a conversation to Markdown format.

    Args:
        conversation_id: ID of the conversation to export
        user: Authenticated user
        db: Database session

    Returns:
        Markdown export of the conversation

    Raises:
        HTTPException: If conversation not found or export fails
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
                f"Conversation {conversation_id} not found for export by user {user.id}"
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found",
            )

        # Build markdown content
        markdown_lines = [
            f"# {conversation.title or 'Conversation'}",
            "",
            f"**Model:** {conversation.model_name}  ",
            f"**Created:** {conversation.created_at.strftime('%Y-%m-%d %H:%M:%S')}  ",
            f"**Updated:** {conversation.updated_at.strftime('%Y-%m-%d %H:%M:%S')}  ",
            "",
        ]

        if conversation.system_prompt:
            markdown_lines.extend([
                "## System Prompt",
                "",
                f"> {conversation.system_prompt}",
                "",
            ])

        markdown_lines.extend([
            "## Messages",
            "",
        ])

        for msg in conversation.messages:
            role_display = {
                "user": "User",
                "assistant": "Assistant",
                "system": "System",
            }.get(msg.role, msg.role.capitalize())

            timestamp = msg.created_at.strftime("%Y-%m-%d %H:%M:%S")

            markdown_lines.extend([
                f"### {role_display} ({timestamp})",
                "",
                msg.content,
                "",
            ])

            if msg.tokens_used:
                markdown_lines.append(f"*Tokens: {msg.tokens_used}*\n")

        markdown_lines.extend([
            "---",
            f"*Exported: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC*",
        ])

        markdown_content = "\n".join(markdown_lines)

        logger.info(
            f"Exported conversation {conversation_id} to Markdown for user {user.id} "
            f"({len(conversation.messages)} messages)"
        )

        return Response(
            content=markdown_content,
            media_type="text/markdown",
            headers={
                "Content-Disposition": f'attachment; filename="conversation_{conversation_id}.md"'
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting conversation to Markdown: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export conversation: {str(e)}",
        )


@router.post("/conversations/import", response_model=ImportConversationResponse)
async def import_conversation(
    import_data: ImportConversationRequest,
    user: User = Depends(require_auth),
    db: Session = Depends(get_db),
):
    """
    Import a conversation from JSON data.

    Args:
        import_data: Conversation data to import
        user: Authenticated user
        db: Database session

    Returns:
        Import result with new conversation ID

    Raises:
        HTTPException: If import fails or data is invalid
    """
    try:
        # Sanitize and validate data
        title = import_data.title or f"Imported Chat - {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}"

        # Create new conversation
        new_conversation = Conversation(
            user_id=user.id,
            title=title,
            model_name=import_data.model_name,
            system_prompt=import_data.system_prompt,
        )

        db.add(new_conversation)
        db.flush()  # Get the conversation ID without committing

        # Import messages
        imported_count = 0
        for msg_data in import_data.messages:
            # Sanitize content (basic XSS prevention)
            content = msg_data.content.strip()

            if not content:
                continue

            # Create message
            message = Message(
                conversation_id=new_conversation.id,
                role=msg_data.role,
                content=content,
                tokens_used=msg_data.tokens_used,
            )

            db.add(message)
            imported_count += 1

        if imported_count == 0:
            db.rollback()
            logger.warning(f"Import failed: no valid messages for user {user.id}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid messages to import",
            )

        # Commit transaction
        db.commit()
        db.refresh(new_conversation)

        logger.info(
            f"Imported conversation with {imported_count} messages for user {user.id}, "
            f"new conversation ID: {new_conversation.id}"
        )

        return ImportConversationResponse(
            success=True,
            message=f"Successfully imported conversation with {imported_count} messages",
            conversation_id=new_conversation.id,
            imported_messages=imported_count,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error importing conversation: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to import conversation: {str(e)}",
        )
