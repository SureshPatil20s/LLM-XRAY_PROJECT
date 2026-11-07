"""
Database connection and session handling.
Compatible with PostgreSQL (Neon/Supabase) and SQLite.
"""
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from server.app.core.config import settings

db_url = settings.DATABASE_URL or "sqlite:///./data/conversations/suresh_gpt_platform.db"
# Handle postgres:// to postgresql:// dialect conversion if provided by providers like Neon/Supabase/Render
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

connect_args = {}
if db_url.startswith("sqlite"):
    connect_args["check_same_thread"] = False
    # Ensure dir exists
    db_dir = os.path.dirname(db_url.replace("sqlite:///", "").replace("sqlite://./", "./"))
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

engine = create_engine(
    db_url,
    connect_args=connect_args,
    pool_pre_ping=True
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """Dependency that provides an independent database session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

