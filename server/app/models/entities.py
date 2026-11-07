"""
SQLAlchemy Database Models for SURESH.GPT Platform:
- Users (auth, Google OAuth, settings)
- Conversations (title, pinned, timestamps, foreign key to user)
- Messages (role, content, model, tokens, sources, feedback)
- Documents (uploaded files, chunks, page metadata for RAG)
"""
from sqlalchemy import Column, String, Integer, Float, Text, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
import uuid
from server.app.db.session import Base


def generate_uuid() -> str:
    return str(uuid.uuid4())


class User(Base):
    __tablename__ = "users"

    id = Column(String(64), primary_key=True, default=generate_uuid)
    email = Column(String(255), unique=True, index=True, nullable=False)
    name = Column(String(255), nullable=True)
    hashed_password = Column(String(255), nullable=True)
    avatar_url = Column(String(512), nullable=True)
    provider = Column(String(32), default="email")  # 'email' or 'google'
    google_id = Column(String(128), unique=True, index=True, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    # User Preferences
    system_prompt = Column(Text, default="You are SURESH.GPT, a knowledgeable, concise, and helpful AI assistant.")
    default_model = Column(String(64), default="llama-3.3-70b-versatile")
    temperature = Column(Float, default=0.7)

    # Relationships
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")
    documents = relationship("Document", back_populates="user", cascade="all, delete-orphan")


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(String(64), primary_key=True, default=generate_uuid)
    user_id = Column(String(64), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String(255), default="New Chat")
    pinned = Column(Boolean, default=False)
    model = Column(String(64), default="llama-3.3-70b-versatile")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), index=True)

    # Relationships
    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan", order_by="Message.created_at")


class Message(Base):
    __tablename__ = "messages"

    id = Column(String(64), primary_key=True, default=generate_uuid)
    conversation_id = Column(String(64), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String(32), nullable=False)  # 'user', 'assistant', 'system'
    content = Column(Text, nullable=False)
    model = Column(String(64), nullable=True)
    sources = Column(Text, nullable=True)  # JSON-encoded array of retrieved RAG documents & pages
    feedback = Column(Integer, default=0)  # 0: None, 1: thumbs_up, -1: thumbs_down
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    # Relationships
    conversation = relationship("Conversation", back_populates="messages")


class Document(Base):
    __tablename__ = "documents"

    id = Column(String(64), primary_key=True, default=generate_uuid)
    user_id = Column(String(64), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    filename = Column(String(255), nullable=False)
    file_type = Column(String(32), nullable=False)  # 'pdf', 'docx', 'txt', 'csv', 'json'
    file_size = Column(Integer, default=0)
    total_pages = Column(Integer, default=1)
    total_chunks = Column(Integer, default=1)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    user = relationship("User", back_populates="documents")
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(String(64), primary_key=True, default=generate_uuid)
    document_id = Column(String(64), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    page_number = Column(Integer, default=1)
    chunk_index = Column(Integer, default=0)
    content = Column(Text, nullable=False)
    embedding_json = Column(Text, nullable=True)  # JSON-encoded vector embedding for semantic search

    # Relationships
    document = relationship("Document", back_populates="chunks")

