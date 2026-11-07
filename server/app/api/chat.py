"""
Chat & Conversation API Endpoints:
- List conversations (chronological + search)
- Create new conversation
- Get conversation with full message history
- Rename conversation & toggle pin
- Delete conversation
- Real-time streaming AI response (SSE)
- Message feedback (thumbs up/down)
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import List, Optional
import json

from server.app.db.session import get_db
from server.app.models.entities import User, Conversation, Message
from server.app.schemas.schemas import (
    ConversationResponse,
    ConversationCreate,
    ConversationRename,
    MessageResponse,
    MessageCreate,
    MessageFeedback
)
from server.app.core.security import get_current_user
from server.app.services.llm_service import hosted_llm
from server.app.services.rag_service import rag_service
from server.app.core.config import settings

router = APIRouter(tags=["Chat & Conversations"])


@router.get("/conversations", response_model=List[ConversationResponse])
def get_conversations(
    q: Optional[str] = Query(None, description="Search keyword in titles"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Lists conversations for the current user, optionally filtered by keyword."""
    # Guest users have temporary/ephemeral sessions — their chats are not persisted to sidebar history
    if current_user.provider == "guest":
        return []

    query = db.query(Conversation).filter(Conversation.user_id == current_user.id)
    if q and q.strip():
        query = query.filter(Conversation.title.ilike(f"%{q.strip()}%"))

    # Pinned first, then newest updated
    convs = query.order_by(Conversation.pinned.desc(), Conversation.updated_at.desc()).all()
    # Map pinned to is_pinned
    for c in convs:
        setattr(c, "is_pinned", bool(c.pinned))
    return convs


@router.post("/conversations", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
def create_conversation(
    conv_in: ConversationCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Initializes a new persistent conversation."""
    model_name = conv_in.model or current_user.default_model or settings.DEFAULT_MODEL
    title = conv_in.title or "New Chat"
    conv = Conversation(
        user_id=current_user.id,
        title=title,
        model=model_name
    )
    db.add(conv)
    db.commit()
    db.refresh(conv)
    setattr(conv, "is_pinned", bool(conv.pinned))
    return conv


@router.get("/conversations/{conversation_id}")
def get_conversation_history(
    conversation_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Retrieves full conversation details and all message history."""
    conv = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == current_user.id
    ).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    messages = db.query(Message).filter(
        Message.conversation_id == conversation_id
    ).order_by(Message.created_at.asc()).all()

    formatted_messages = []
    for m in messages:
        sources_list = []
        if m.sources:
            try:
                sources_list = json.loads(m.sources)
            except Exception:
                sources_list = []
        
        rating_str = None
        if m.feedback == 1:
            rating_str = "like"
        elif m.feedback == -1:
            rating_str = "dislike"

        formatted_messages.append({
            "id": m.id,
            "conversation_id": m.conversation_id,
            "role": m.role,
            "content": m.content,
            "model": m.model,
            "sources": sources_list,
            "feedback": rating_str,
            "created_at": m.created_at.isoformat() if m.created_at else None
        })

    setattr(conv, "is_pinned", bool(conv.pinned))
    return {
        "conversation": {
            "id": conv.id,
            "user_id": conv.user_id,
            "title": conv.title,
            "model": conv.model,
            "is_pinned": bool(conv.pinned),
            "pinned": bool(conv.pinned),
            "created_at": conv.created_at.isoformat() if conv.created_at else None,
            "updated_at": conv.updated_at.isoformat() if conv.updated_at else None,
        },
        "messages": formatted_messages
    }


@router.put("/conversations/{conversation_id}", response_model=ConversationResponse)
def update_conversation(
    conversation_id: str,
    updates: dict,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Updates title or pin status of a conversation."""
    conv = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == current_user.id
    ).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    if "title" in updates and updates["title"]:
        conv.title = updates["title"].strip()
    if "is_pinned" in updates:
        conv.pinned = bool(updates["is_pinned"])
    if "pinned" in updates:
        conv.pinned = bool(updates["pinned"])

    db.commit()
    db.refresh(conv)
    setattr(conv, "is_pinned", bool(conv.pinned))
    return conv


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(
    conversation_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Deletes a conversation and its messages."""
    conv = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == current_user.id
    ).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    db.delete(conv)
    db.commit()
    return None


async def _handle_message_stream(
    conversation_id: str,
    msg_in: MessageCreate,
    db: Session,
    current_user: User
):
    conv = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == current_user.id
    ).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # 1. Save User Message
    user_msg = Message(
        conversation_id=conversation_id,
        role="user",
        content=msg_in.content
    )
    db.add(user_msg)
    db.commit()

    # Auto-generate title if this is the first turn
    msg_count = db.query(Message).filter(Message.conversation_id == conversation_id).count()
    if msg_count <= 1:
        snippet = msg_in.content.strip()
        conv.title = snippet[:32] + "..." if len(snippet) > 32 else snippet
        db.commit()

    # 2. Check RAG Context if requested
    rag_context = ""
    sources_list = []
    sources_json = None
    if msg_in.use_rag:
        rag_context, sources_list = rag_service.search_user_documents(
            db, current_user.id, msg_in.content, top_k=4
        )
        if sources_list:
            sources_json = json.dumps(sources_list, ensure_ascii=False)

    # 3. Assemble Prompt Messages History
    history = db.query(Message).filter(Message.conversation_id == conversation_id).order_by(Message.created_at).all()
    system_text = current_user.system_prompt or "You are a helpful AI assistant."
    if rag_context:
        system_text += f"\n\n[Relevant Reference Documents Grounding]:\n{rag_context}\n\nAnswer accurately using the above sources when relevant."

    llm_messages = [{"role": "system", "content": system_text}]
    for m in history[-8:]:  # Sliding window of last 8 turns
        llm_messages.append({"role": m.role, "content": m.content})

    chosen_model = msg_in.model or conv.model or current_user.default_model or settings.DEFAULT_MODEL

    # If image data is attached for Vision/OCR
    if msg_in.image_data and llm_messages:
        chosen_model = "llama-3.2-11b-vision-preview"
        prompt_text = msg_in.content or "Please perform OCR: read and transcribe all text from this image and analyze its visual content."
        llm_messages[-1] = {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": msg_in.image_data}}
            ]
        }

    # 4. Stream response generator and save assistant message upon completion
    async def sse_event_generator():
        full_content = []
        # Send sources metadata event if RAG retrieved sources
        if sources_list:
            yield f"data: {json.dumps({'sources': sources_list})}\n\n"

        async for chunk in hosted_llm.stream_chat_completion(
            messages=llm_messages,
            model=chosen_model,
            temperature=current_user.temperature or 0.7
        ):
            full_content.append(chunk)
            # Send text delta
            yield f"data: {json.dumps({'delta': chunk})}\n\n"

        complete_response = "".join(full_content)
        # Save assistant message into DB
        asst_msg = Message(
            conversation_id=conversation_id,
            role="assistant",
            content=complete_response,
            model=chosen_model,
            sources=sources_json
        )
        db.add(asst_msg)
        db.commit()
        db.refresh(asst_msg)

        # Send completion
        yield "data: [DONE]\n\n"

    return StreamingResponse(sse_event_generator(), media_type="text/event-stream")


@router.post("/messages/stream")
async def stream_message_global(
    msg_in: MessageCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Streams assistant response for a given conversation_id."""
    if not msg_in.conversation_id:
        raise HTTPException(status_code=400, detail="conversation_id is required")
    return await _handle_message_stream(msg_in.conversation_id, msg_in, db, current_user)


@router.post("/conversations/{conversation_id}/messages/stream")
async def stream_message_nested(
    conversation_id: str,
    msg_in: MessageCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Streams assistant response for a nested conversation route."""
    return await _handle_message_stream(conversation_id, msg_in, db, current_user)


@router.post("/messages/{message_id}/feedback")
def set_message_feedback(
    message_id: str,
    feedback_in: MessageFeedback,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Sets thumbs up/down feedback for a message."""
    msg = db.query(Message).join(Conversation).filter(
        Message.id == message_id,
        Conversation.user_id == current_user.id
    ).first()
    if not msg:
        raise HTTPException(status_code=404, detail="Message not found")

    if feedback_in.rating:
        msg.feedback = 1 if feedback_in.rating == "like" else -1
    elif feedback_in.feedback is not None:
        msg.feedback = feedback_in.feedback

    db.commit()
    return {"status": "success", "feedback": msg.feedback}
