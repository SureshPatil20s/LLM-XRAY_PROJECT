"""
FastAPI Main Application Entry Point for SURESH.GPT Platform.
- Configures CORS for Next.js frontend
- Registers Authentication, Chat & Document routers
- Auto-initializes database tables
- Exposes model catalog and health checks
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

from server.app.core.config import settings
from server.app.db.session import engine, Base
from server.app.api.auth import router as auth_router
from server.app.api.chat import router as chat_router
from server.app.api.documents import router as doc_router

# Auto-create tables (SQLite or PostgreSQL)
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Multi-user ChatGPT-style Platform powered by FastAPI and Hosted Groq LLMs",
    version="3.0.0"
)

# Configure CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows local Next.js dev server & Vercel deployments
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routers
app.include_router(auth_router, prefix=settings.API_V1_STR)
app.include_router(chat_router, prefix=settings.API_V1_STR)
app.include_router(doc_router, prefix=settings.API_V1_STR)


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "default_model": settings.DEFAULT_MODEL,
        "groq_configured": bool(settings.GROQ_API_KEY)
    }


@app.get(f"{settings.API_V1_STR}/models")
def list_available_models():
    """Returns available hosted models for the model picker."""
    return settings.AVAILABLE_MODELS


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.app.main:app", host="0.0.0.0", port=8000, reload=True)

