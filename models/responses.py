"""
Pydantic response models for WhatsApp AI Service
"""

from pydantic import BaseModel, Field, HttpUrl
from typing import Literal, Optional, Dict, Any, List


class ProductMatch(BaseModel):
    """Single product match from vector search"""

    name: str = Field(..., description="Product name")
    brand: str = Field(..., description="Brand name")
    price: str = Field(..., description="Price in INR")
    discount: str = Field(..., description="Discount info (e.g., '15% off' or 'No discount')")
    specs: str = Field(..., description="Short product specifications")
    product_url: str = Field(..., description="Product page URL")
    image_url: str = Field(..., description="Product image URL")
    similarity_score: float = Field(..., description="Vector similarity score (0-1)", ge=0, le=1)

    class Config:
        json_schema_extra = {
            "example": {
                "name": "Pneumatic Stapler F30",
                "brand": "Kaymo",
                "price": "₹4,500",
                "discount": "15% off",
                "specs": "Lightweight, 8-bar pressure, 30mm crown",
                "product_url": "https://startrade.in/products/pneumatic-stapler-f30",
                "image_url": "https://cdn.startrade.in/pneumatic-stapler.jpg",
                "similarity_score": 0.92,
            }
        }


class AvailabilityStatus(BaseModel):
    """Structured availability decision from LLM"""

    status: Literal["in_stock", "out_of_stock", "alternate_available"] = Field(
        ..., description="Product availability status"
    )

    matched_product: str = Field(..., description="Primary matched product name")
    brand: str = Field(..., description="Product brand")
    price: str = Field(..., description="Price in INR")
    discount: str = Field(..., description="Discount information")
    specs: str = Field(..., description="Product specifications")
    product_url: str = Field(..., description="Product page URL")
    image_url: str = Field(..., description="Product image URL")

    output: str = Field(
        ...,
        description="Hinglish summary message for WhatsApp",
        examples=[
            "Haan, Pneumatic Stapler F30 available hai! ₹4,500 mein with 15% discount. Details: https://..."
        ],
    )

    class Config:
        json_schema_extra = {
            "example": {
                "status": "in_stock",
                "matched_product": "Pneumatic Stapler F30",
                "brand": "Kaymo",
                "price": "₹4,500",
                "discount": "15% off",
                "specs": "Lightweight, 8-bar pressure, 30mm crown",
                "product_url": "https://startrade.in/products/pneumatic-stapler-f30",
                "image_url": "https://cdn.startrade.in/pneumatic-stapler.jpg",
                "output": "Haan, Pneumatic Stapler F30 available hai! Price: ₹4,500 with 15% discount. Check details: https://startrade.in/products/pneumatic-stapler-f30",
            }
        }


class WhatsAppTemplatePayload(BaseModel):
    """
    Final output matching n8n WhatsApp template format
    Matches the structure expected by n8n's WhatsApp send node
    """

    countryCode: str = Field(default="+91", description="Country code")
    phoneNumber: str = Field(..., description="Phone number without country code")
    type: Literal["Template"] = Field(default="Template", description="Message type")
    callbackData: str = Field(default="availability_check", description="Callback identifier")

    template: Dict[str, Any] = Field(
        ...,
        description="WhatsApp template structure",
        examples=[
            {
                "name": "availability_check_with_link",
                "languageCode": "en",
                "bodyValues": [
                    "Pneumatic Stapler F30",
                    "₹4,500",
                    "15% off",
                    "Kaymo",
                    "Lightweight, 8-bar pressure",
                    "https://startrade.in/products/pneumatic-stapler-f30",
                ],
            }
        ],
    )

    image_url: str = Field(..., description="Image URL to send with message")
    summary_output: str = Field(..., description="Hinglish summary text")
    status: Literal["in_stock", "out_of_stock", "alternate_available"] = Field(
        ..., description="Availability status"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "countryCode": "+91",
                "phoneNumber": "9643524080",
                "type": "Template",
                "callbackData": "availability_check",
                "template": {
                    "name": "availability_check_with_link",
                    "languageCode": "en",
                    "bodyValues": [
                        "Pneumatic Stapler F30",
                        "₹4,500",
                        "15% off",
                        "Kaymo",
                        "Lightweight, 8-bar pressure",
                        "https://startrade.in/products/pneumatic-stapler-f30",
                    ],
                },
                "image_url": "https://cdn.startrade.in/pneumatic-stapler.jpg",
                "summary_output": "Haan, Pneumatic Stapler F30 available hai! Price: ₹4,500 with 15% discount.",
                "status": "in_stock",
            }
        }


class AgentResponse(BaseModel):
    """Complete agent response with metadata"""

    result: WhatsAppTemplatePayload = Field(..., description="WhatsApp template payload")
    session_id: str = Field(..., description="Conversation session ID")

    retrieved_products: List[ProductMatch] = Field(
        default_factory=list, description="Top K products retrieved from vector search"
    )

    processing_time_ms: int = Field(..., description="Total processing time in milliseconds", ge=0)

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (top product, LLM calls, etc.)",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "result": {
                    "countryCode": "+91",
                    "phoneNumber": "9643524080",
                    "type": "Template",
                    "callbackData": "availability_check",
                    "template": {
                        "name": "availability_check_with_link",
                        "languageCode": "en",
                        "bodyValues": [
                            "Pneumatic Stapler F30",
                            "₹4,500",
                            "15% off",
                            "Kaymo",
                            "Lightweight, 8-bar pressure",
                            "https://startrade.in/products/pneumatic-stapler-f30",
                        ],
                    },
                    "image_url": "https://cdn.startrade.in/pneumatic-stapler.jpg",
                    "summary_output": "Haan, Pneumatic Stapler F30 available hai!",
                    "status": "in_stock",
                },
                "session_id": "whatsapp_+919643524080",
                "retrieved_products": [
                    {
                        "name": "Pneumatic Stapler F30",
                        "brand": "Kaymo",
                        "price": "₹4,500",
                        "discount": "15% off",
                        "specs": "Lightweight, 8-bar pressure",
                        "product_url": "https://startrade.in/products/pneumatic-stapler-f30",
                        "image_url": "https://cdn.startrade.in/pneumatic-stapler.jpg",
                        "similarity_score": 0.92,
                    }
                ],
                "processing_time_ms": 1250,
                "metadata": {"llm_calls": 1, "qdrant_search_time_ms": 120},
            }
        }
