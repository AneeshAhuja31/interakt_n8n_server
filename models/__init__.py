"""Pydantic models for WhatsApp AI Service"""

from .requests import AgentRequest, AgentRequestFromN8N
from .responses import (
    ProductMatch,
    AvailabilityStatus,
    WhatsAppTemplatePayload,
    AgentResponse,
)

__all__ = [
    "AgentRequest",
    "AgentRequestFromN8N",
    "ProductMatch",
    "AvailabilityStatus",
    "WhatsAppTemplatePayload",
    "AgentResponse",
]
