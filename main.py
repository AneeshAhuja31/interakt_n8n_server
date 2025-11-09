"""
WhatsApp AI Service - FastAPI Application
Combines Intent Classification + LangGraph Agent for Star Trade

Version 2.0.0 - Refactored with routers and LangGraph integration
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import intent_router, agent_router
from config import settings
from contextlib import asynccontextmanager




# ============================================================
# LIFESPAN EVENTS
# ============================================================



# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    description="Intent Classification + LangGraph Agent for WhatsApp customer service",
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(intent_router)
app.include_router(agent_router)



# ============================================================
# ROOT ENDPOINTS
# ============================================================


@app.get("/")
async def root():
    """
    Root endpoint - Service information and health check
    """
    return {
        "status": "running",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "endpoints": {
            "intent_classification": "/classify-intent",
            "intent_test": "/test-classification",
            "agent_availability": "/agent/availability",
            "agent_availability_n8n": "/agent/availability/n8n",
            "agent_health": "/agent/health",
            "docs": "/docs",
            "redoc": "/redoc",
        },
        "models": {
            "intent_classifier": "gemini-2.0-flash-exp",
            "availability_agent": settings.GEMINI_MODEL,
        },
        "integrations": {
            "vector_store": "Qdrant",
            "database": "Supabase (PostgreSQL)",
            "graph_framework": "LangGraph",
        },
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
    }



# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":
    import uvicorn,os
    from dotenv import load_dotenv
    load_dotenv()
    # Use import string so uvicorn can enable the --reload feature
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT",8000)),
        reload=True,
        log_level="info",
    )
