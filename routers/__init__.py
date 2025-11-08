"""FastAPI routers for WhatsApp AI Service"""

from .intent_router import router as intent_router
from .agent_router import router as agent_router

__all__ = ["intent_router", "agent_router"]
