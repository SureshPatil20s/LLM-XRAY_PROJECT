"""
Pydantic schemas for request validation and response serialization.
"""
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


# User Schemas
class UserBase(BaseModel):
    email: EmailStr
    name: Optional[str] = None


class UserCreate(UserBase):
    password: str = Field(..., min_length=8)


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class GoogleAuthRequest(BaseModel):
    id_token: Optional[str] = None
    email: EmailStr
    name: Optional[str] = None
    avatar_url: Optional[str] = None


class UserUpdate(BaseModel):
    name: Optional[str] = None
    system_prompt: Optional[str] = None
    default_model: Optional[str] = None
    temperature: Optional[float] = None
    avatar_url: Optional[str] = None


class UserResponse(UserBase):
    id: str
    avatar_url: Optional[str] = None
    provider: str
    system_prompt: Optional[str] = None
    default_model: Optional[str] = None
    temperature: Optional[float] = None
    created_at: datetime

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse


# Chat & Conversation Schemas
class MessageCreate(BaseModel):
    conversation_id: Optional[str] = None
    content: str
    model: Optional[str] = None
    use_rag: Optional[bool] = True
    image_data: Optional[str] = None


class MessageResponse(BaseModel):
    id: str
    conversation_id: str
    role: str
    content: str
    model: Optional[str] = None
    sources: Optional[str] = None
    feedback: Optional[int] = 0
    created_at: datetime

    class Config:
        from_attributes = True


class MessageFeedback(BaseModel):
    feedback: Optional[int] = None  # 1 for up, -1 for down, 0 for clear
    rating: Optional[str] = None    # "like" or "dislike"


class ConversationCreate(BaseModel):
    title: Optional[str] = "New Chat"
    model: Optional[str] = None


class ConversationRename(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)


class ConversationResponse(BaseModel):
    id: str
    title: str
    pinned: bool = False
    is_pinned: bool = False
    model: str
    created_at: datetime
    updated_at: datetime
    messages: Optional[List[MessageResponse]] = []

    class Config:
        from_attributes = True

    @classmethod
    def model_validate(cls, obj, **kwargs):
        if hasattr(obj, "pinned") and not hasattr(obj, "is_pinned"):
            setattr(obj, "is_pinned", getattr(obj, "pinned"))
        return super().model_validate(obj, **kwargs)


# Document & RAG Schemas
class DocumentResponse(BaseModel):
    id: str
    filename: str
    file_type: str
    file_size: int
    total_pages: int = 1
    total_chunks: int = 0
    chunk_count: int = 0
    created_at: datetime

    class Config:
        from_attributes = True

    @classmethod
    def model_validate(cls, obj, **kwargs):
        if hasattr(obj, "total_chunks") and not hasattr(obj, "chunk_count"):
            setattr(obj, "chunk_count", getattr(obj, "total_chunks"))
        return super().model_validate(obj, **kwargs)

