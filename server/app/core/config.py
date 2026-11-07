"""
Server Configuration & Settings.
Supports PostgreSQL (Neon/Supabase) and SQLite fallback for seamless local/cloud dev.
Configures Groq Hosted LLM endpoint with default Llama-3.3-70b-versatile and Qwen-2.5-32b.
"""
from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    PROJECT_NAME: str = "SURESH.GPT Platform"
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = os.getenv("SECRET_KEY", "suresh_gpt_super_secret_jwt_key_2026_change_in_prod")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days

    # Database: Defaults to local SQLite if DATABASE_URL is not set or empty, otherwise PostgreSQL (Neon/Supabase)
    DATABASE_URL: str = os.getenv("DATABASE_URL") or "sqlite:///./data/conversations/suresh_gpt_platform.db"

    # Hosted LLM Inference via Groq (Free Tier)
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "llama-3.3-70b-versatile")

    # Available Hosted Models
    AVAILABLE_MODELS: list = [
        {"id": "llama-3.3-70b-versatile", "name": "Llama 3.3 70B (Fast & Intelligent)", "provider": "Groq", "description": "Flagship 70B reasoning and general coding"},
        {"id": "llama-3.2-11b-vision-preview", "name": "Llama 3.2 11B Vision & OCR", "provider": "Groq Vision", "description": "Reads & extracts text from images, scans, and diagrams"},
        {"id": "flux-image-gen", "name": "AI Image Generator (Free Turbo)", "provider": "Pollinations Turbo", "description": "High-speed 1024x1024 AI image generation (free & unlimited)"},
        {"id": "qwen-2.5-32b", "name": "Qwen 2.5 32B (Advanced Reasoning)", "provider": "Groq", "description": "Math, multilingual, and deep problem solving"},
        {"id": "mixtral-8x7b-32768", "name": "Mixtral 8x7B (Large Context)", "provider": "Groq", "description": "32k context window MoE model"}
    ]

    # CORS settings
    CORS_ORIGINS: list = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://*.vercel.app"
    ]

    class Config:
        env_file = ".env"
        extra = "allow"


settings = Settings()

