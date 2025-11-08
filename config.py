"""
Configuration and environment variables for WhatsApp AI Service
"""

from functools import lru_cache
from typing import Optional
from dotenv import load_dotenv
load_dotenv(override=True)
import os
class Settings():
    """Application settings loaded from environment variables"""

    # API Configuration
    APP_NAME: str = "WhatsApp AI Service"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = False

    # Google AI (Gemini)
    GOOGLE_API_KEY: str =  os.getenv("GOOGLE_API_KEY")
    GEMINI_MODEL: str = "gemini-2.5-flash"
    GEMINI_TEMPERATURE: float = 0.1

    # Supabase / Postgres
    SUPABASE_URL: str =  os.getenv("SUPABASE_URL")
    SUPABASE_KEY: str =  os.getenv("SUPABASE_KEY")
    POSTGRES_URI: Optional[str] = os.getenv("POSTGRES_URI",None)  # Optional for LangGraph checkpointing

    # OpenAI (for embeddings)
    OPENAI_API_KEY:  str =  os.getenv("OPENAI_API_KEY")
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"  # 1536 dimensions

    # Qdrant Vector Store
    QDRANT_URL:  str =  os.getenv("QDRANT_URL")
    QDRANT_API_KEY: str =  os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION: str = "product_text"
    QDRANT_TOP_K: int = 5

    # Airtable (Optional)
    AIRTABLE_API_KEY: str =  os.getenv("AIRTABLE_API_KEY")
    AIRTABLE_BASE_ID: Optional[str] = None

    # Agent Settings
    AGENT_MAX_MESSAGES: int = 20  # Memory window size
    AGENT_TIMEOUT_SECONDS: int = 30

    # CORS Settings
    CORS_ORIGINS: list = ["*"]  # Update for production


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Global settings instance
settings = get_settings()
