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
    original_price: Optional[int] = Field(None, description="Original price before discount (int)")
    discount_percent: Optional[int] = Field(None, description="Discount percentage (0-100)")
    discount_amount: Optional[int] = Field(None, description="Discount amount in INR")
    final_price: Optional[int] = Field(None, description="Final price after discount (int)")

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
                "original_price": 4500,
                "discount_percent": 15,
                "discount_amount": 675,
                "final_price": 3825,
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
    original_price: Optional[int] = Field(None, description="Original price before discount (int)")
    discount_percent: Optional[int] = Field(None, description="Discount percentage (0-100)")
    discount_amount: Optional[int] = Field(None, description="Discount amount in INR")
    final_price: Optional[int] = Field(None, description="Final price after discount (int)")

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
                "original_price": 4500,
                "discount_percent": 15,
                "discount_amount": 675,
                "final_price": 3825,
            }
        }


class AgentResponse(BaseModel):
    """Complete agent response with metadata"""

    result: WhatsAppTemplatePayload = Field(..., description="WhatsApp template payload")
    session_id: str = Field(..., description="Conversation session ID")
    originalRequest: str = Field(..., description="Original customer query/message")

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
                "originalRequest": "PNEUMATIC STAPLER",
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


class SaveMessageResponse(BaseModel):
    """Response from save-message endpoint"""

    success: bool = Field(..., description="Whether message was saved successfully")
    session_id: str = Field(..., description="Session ID for the message")
    message_id: Optional[int] = Field(None, description="Database ID of saved message")
    message: str = Field(default="Message saved successfully", description="Status message")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "session_id": "whatsapp_+919643524080",
                "message_id": 123,
                "message": "Message saved successfully",
            }
        }


class OrderItem(BaseModel):
    """Individual item in an order"""

    product_name: str = Field(..., description="Product name from conversation")
    quantity: int = Field(default=1, description="Number of items", ge=1)
    unit_price: str = Field(..., description="Original price per unit before discount")
    discount: str = Field(default="No discount", description="Discount information (e.g., '15% off')")
    subtotal: str = Field(..., description="Final amount to pay for this item (after discount × quantity)")
    item_id: Optional[str] = Field(None, description="Generated item ID")
    discount_percent: Optional[int] = Field(None, description="Discount percentage (0-100)")
    discount_amount: Optional[int] = Field(None, description="Discount amount per unit")
    final_unit_price: Optional[int] = Field(None, description="Final price per unit after discount")
    item_discount_amount: Optional[int] = Field(None, description="Total discount for this item (discount × quantity)")
    price_verified: Optional[bool] = Field(None, description="Whether price was extracted from conversation")

    class Config:
        json_schema_extra = {
            "example": {
                "product_name": "Air Stapler",
                "quantity": 2,
                "unit_price": "2499",
                "discount": "15% off",
                "discount_percent": 15,
                "discount_amount": 375,
                "final_unit_price": 2124,
                "subtotal": "4248",
                "item_discount_amount": 750,
                "item_id": "ITEM_20250109_001",
                "price_verified": True,
            }
        }


class OrderSummary(BaseModel):
    """Extracted order details from chat history (supports multiple items)"""

    items: List[OrderItem] = Field(..., description="List of items in the order")
    total_price: str = Field(..., description="Total amount to pay (after all discounts)")
    order_id: Optional[str] = Field(None, description="Generated order ID")
    total_discount: Optional[int] = Field(None, description="Total discount amount across all items")
    original_price: Optional[int] = Field(None, description="Original price before discounts")

    class Config:
        json_schema_extra = {
            "example": {
                "items": [
                    {
                        "product_name": "Air Stapler",
                        "quantity": 2,
                        "unit_price": "2499",
                        "discount": "15% off",
                        "discount_percent": 15,
                        "final_unit_price": 2124,
                        "subtotal": "4248",
                        "item_discount_amount": 750,
                    },
                    {
                        "product_name": "Nails Box",
                        "quantity": 1,
                        "unit_price": "500",
                        "discount": "No discount",
                        "subtotal": "500",
                    },
                ],
                "total_price": "4748",
                "total_discount": 750,
                "original_price": 5498,
                "order_id": "ORD_20250109_001",
            }
        }


class OrderConfirmationResponse(BaseModel):
    """Response from order-confirmation endpoint (supports multiple items)"""

    success: bool = Field(..., description="Whether order was processed successfully")
    order: OrderSummary = Field(..., description="Extracted order details with multiple items")
    template_body_values: List[str] = Field(
        ..., description="Values to populate WhatsApp template"
    )
    session_id: str = Field(..., description="Conversation session ID")
    message: str = Field(
        default="Order confirmed successfully", description="Status message"
    )
    items_extracted: bool = Field(
        default=True, description="Whether items were successfully extracted from conversation (False when using fallback)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "order": {
                    "items": [
                        {
                            "product_name": "Air Stapler",
                            "quantity": 2,
                            "unit_price": "2499",
                            "discount": "15% off",
                            "subtotal": "4998",
                        },
                        {
                            "product_name": "Nails Box",
                            "quantity": 1,
                            "unit_price": "500",
                            "discount": "No discount",
                            "subtotal": "500",
                        },
                    ],
                    "total_price": "5498",
                    "order_id": "ORD_20250109_001",
                },
                "template_body_values": ["Air Stapler (x2), Nails Box (x1)", "₹5498", "ORD_20250109_001"],
                "session_id": "whatsapp_+919643524080",
                "message": "Order confirmed successfully",
                "items_extracted": True,
            }
        }


class LatestOrderResponse(BaseModel):
    """Response for getting latest order by phone number"""

    success: bool = Field(..., description="Whether operation was successful")
    order: OrderSummary = Field(..., description="Latest order details with multiple items")
    order_summary_text: str = Field(..., description="Order summary as single-line text without newlines")
    message: str = Field(default="Latest order retrieved successfully", description="Status message")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "order": {
                    "items": [
                        {
                            "product_name": "Air Stapler",
                            "quantity": 2,
                            "unit_price": "2499",
                            "discount": "15% off",
                            "subtotal": "4998",
                            "item_id": "ITEM_20250110_143052_789_001",
                        },
                        {
                            "product_name": "Nails Box",
                            "quantity": 1,
                            "unit_price": "500",
                            "discount": "No discount",
                            "subtotal": "500",
                            "item_id": "ITEM_20250110_143052_789_002",
                        },
                    ],
                    "total_price": "5498",
                    "order_id": "ORD_20250110_143052_789",
                },
                "order_summary_text": "Order ID: ORD_20250110_143052_789 | Items: Air Stapler x2 (₹2499 each, 15% off), Nails Box x1 (₹500 each, No discount) | Total: ₹5498",
                "message": "Latest order retrieved successfully",
            }
        }


class CustomerFormSubmission(BaseModel):
    """Customer form submission data"""

    form_id: str = Field(..., description="Unique form ID with WhatsApp phone + timestamp")
    whatsapp_phone_number: str = Field(..., description="WhatsApp phone number")
    entered_name: str = Field(..., description="Name entered in form")
    entered_email: str = Field(..., description="Email entered in form")
    entered_phone_number: str = Field(..., description="Phone number entered in form")
    created_at: Optional[str] = Field(None, description="Form submission timestamp")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "form_id": "FORM_9643524080_20250110_143052_789",
                "whatsapp_phone_number": "+919643524080",
                "entered_name": "Aneesh Ahuja",
                "entered_email": "aneesh@startrade.com",
                "entered_phone_number": "+919876543210",
                "created_at": "2025-01-10T14:30:52.789Z",
                "metadata": {"source": "whatsapp_form"},
            }
        }


class CustomerFormResponse(BaseModel):
    """Response from customer form endpoints"""

    success: bool = Field(..., description="Whether operation was successful")
    form_submission: CustomerFormSubmission = Field(..., description="Customer form submission data")
    message: str = Field(default="Form saved successfully", description="Status message")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "form_submission": {
                    "form_id": "FORM_9643524080_20250110_143052_789",
                    "whatsapp_phone_number": "+919643524080",
                    "entered_name": "Aneesh Ahuja",
                    "entered_email": "aneesh@startrade.com",
                    "entered_phone_number": "+919876543210",
                    "created_at": "2025-01-10T14:30:52.789Z",
                },
                "message": "Form saved successfully",
            }
        }


class CustomerLocationSubmission(BaseModel):
    """Customer location submission data"""

    location_id: str = Field(..., description="Unique location ID with WhatsApp phone + timestamp")
    whatsapp_phone_number: str = Field(..., description="WhatsApp phone number")
    address: str = Field(..., description="Full address")
    city: Optional[str] = Field(None, description="City")
    state: Optional[str] = Field(None, description="State")
    pincode: Optional[str] = Field(None, description="PIN code")
    landmark: Optional[str] = Field(None, description="Landmark")
    location_type: str = Field(default="delivery", description="Location type")
    created_at: Optional[str] = Field(None, description="Location submission timestamp")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "location_id": "LOCATION_9643524080_20250110_143052_789",
                "whatsapp_phone_number": "+919643524080",
                "address": "123, MG Road, Sector 14",
                "city": "Gurgaon",
                "state": "Haryana",
                "pincode": "122001",
                "landmark": "Near Cyber Hub",
                "location_type": "delivery",
                "created_at": "2025-01-10T14:30:52.789Z",
                "metadata": {"verified": True},
            }
        }


class CustomerLocationFormResponse(BaseModel):
    """Response from customer location form endpoints"""

    success: bool = Field(..., description="Whether operation was successful")
    location_submission: CustomerLocationSubmission = Field(..., description="Customer location submission data")
    message: str = Field(default="Location saved successfully", description="Status message")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "location_submission": {
                    "location_id": "LOCATION_9643524080_20250110_143052_789",
                    "whatsapp_phone_number": "+919643524080",
                    "address": "123, MG Road, Sector 14",
                    "city": "Gurgaon",
                    "state": "Haryana",
                    "pincode": "122001",
                    "landmark": "Near Cyber Hub",
                    "location_type": "delivery",
                    "created_at": "2025-01-10T14:30:52.789Z",
                },
                "message": "Location saved successfully",
            }
        }
