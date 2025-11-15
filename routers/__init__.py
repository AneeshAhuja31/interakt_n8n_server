"""FastAPI routers for WhatsApp AI Service"""

from .intent_router import router as intent_router
from .agent_router import router as agent_router
from .product_identification_router import router as product_identification_router

__all__ = ["intent_router", "agent_router", "product_identification_router"]
