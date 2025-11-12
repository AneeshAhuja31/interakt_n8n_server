"""
Pydantic request models for WhatsApp AI Service
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, Literal


class AgentRequest(BaseModel):
    """Input request to LangGraph availability agent"""

    message: str = Field(
        ...,
        description="Customer's product query message",
        min_length=1,
        examples=["PNEUMATIC STAPLER", "Air compressor", "Brad nails"],
    )

    phone_number: str = Field(
        ...,
        description="Customer WhatsApp phone number with country code",
        pattern=r"^\+?[1-9]\d{1,14}$",  # E.164 format
        examples=["+919643524080", "9643524080"],
    )

    session_id: Optional[str] = Field(
        None,
        description="Session ID for conversation continuity (auto-generated if not provided)",
        examples=["whatsapp_+919643524080"],
    )

    context: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional context from n8n workflow or webhook",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "message": "PNEUMATIC STAPLER",
                "phone_number": "+919643524080",
                "session_id": "whatsapp_+919643524080",
                "context": {
                    "source": "whatsapp",
                    "user_name": "Aneesh Ahuja"
                },
            }
        }


class AgentRequestFromN8N(BaseModel):
    """
    Alternative input format matching n8n webhook structure
    For backward compatibility with existing n8n workflows
    """

    webhook_payload: Dict[str, Any] = Field(
        ..., description="Full webhook payload from n8n"
    )

    contact_record: Optional[Dict[str, Any]] = Field(
        None, description="Contact record from Supabase"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "webhook_payload": {
                    "body": {
                        "data": {
                            "customer": {
                                "phone_number": "9643524080",
                                "country_code": "+91",
                                "traits": {"name": "Aneesh Ahuja"},
                            },
                            "message": {"message": "PNEUMATIC STAPLER"},
                        }
                    }
                },
                "contact_record": {
                    "phone_number": "9643524080",
                    "active_template_flow": "none",
                    "last_message": "",
                },
            }
        }

    def to_agent_request(self) -> AgentRequest:
        """Convert n8n format to standard AgentRequest"""
        try:
            data = self.webhook_payload.get("body", {}).get("data", {})
            customer = data.get("customer", {})
            message_data = data.get("message", {})

            phone = customer.get("phone_number", "")
            country_code = customer.get("country_code", "+91")

            # Ensure phone has country code
            if not phone.startswith("+"):
                phone = f"{country_code}{phone}"

            message = message_data.get("message", "")

            return AgentRequest(
                message=message,
                phone_number=phone,
                context={
                    "contact_record": self.contact_record or {},
                    "customer_traits": customer.get("traits", {}),
                },
            )
        except Exception as e:
            raise ValueError(f"Invalid n8n webhook payload structure: {str(e)}")


class SaveMessageRequest(BaseModel):
    """Request to save a message to chat history"""

    phone_number: str = Field(
        ...,
        description="Customer WhatsApp phone number with country code",
        pattern=r"^\+?[1-9]\d{1,14}$",
        examples=["+919643524080", "9643524080"],
    )

    role: Literal["human", "ai", "system"] = Field(
        ...,
        description="Message sender role",
    )

    content: str = Field(
        ...,
        description="Message content/text",
        min_length=1,
    )

    message_type: str = Field(
        default="text",
        description="Type of message",
        examples=["text", "audio", "image", "button_click", "template"],
    )

    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional metadata for the message",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "phone_number": "+919643524080",
                "role": "human",
                "content": "Do you have air stapler?",
                "message_type": "text",
                "metadata": {"source": "whatsapp"},
            }
        }


class OrderConfirmationRequest(BaseModel):
    """Request to generate order confirmation from chat history"""

    message: str = Field(
        ...,
        description="Current user message (e.g., 'Confirm my order', 'Yes', etc.)",
        min_length=1,
        examples=["Confirm my order", "Yes, I want this", "I'll take 2"],
    )

    phone_number: str = Field(
        ...,
        description="Customer WhatsApp phone number with country code",
        pattern=r"^\+?[1-9]\d{1,14}$",
        examples=["+919643524080", "9643524080"],
    )

    session_id: Optional[str] = Field(
        None,
        description="Session ID (auto-generated if not provided)",
        examples=["whatsapp_+919643524080"],
    )

    class Config:
        json_schema_extra = {
            "example": {
                "message": "Confirm my order",
                "phone_number": "+919643524080",
                "session_id": "whatsapp_+919643524080",
            }
        }


class CustomerFormRequest(BaseModel):
    """Request to save customer form submission (name, email, phone)"""

    whatsapp_phone_number: str = Field(
        ...,
        description="WhatsApp phone number of the person filling the form (from webhook)",
        pattern=r"^\+?[1-9]\d{1,14}$",
        examples=["+919643524080", "9643524080"],
    )

    entered_name: str = Field(
        ...,
        description="Name entered in the form by the user",
        min_length=1,
        examples=["Aneesh Ahuja", "John Doe"],
    )

    entered_email: str = Field(
        ...,
        description="Email address entered in the form by the user",
        pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
        examples=["aneesh@example.com", "john.doe@gmail.com"],
    )

    entered_phone_number: str = Field(
        ...,
        description="Phone number entered in the form by the user (may be different from WhatsApp number)",
        pattern=r"^\+?[1-9]\d{1,14}$",
        examples=["+919876543210", "9876543210"],
    )

    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional metadata for the form submission",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "whatsapp_phone_number": "+919643524080",
                "entered_name": "Aneesh Ahuja",
                "entered_email": "aneesh@startrade.com",
                "entered_phone_number": "+919876543210",
                "metadata": {"source": "whatsapp_form"},
            }
        }


class CustomerLocationFormRequest(BaseModel):
    """Request to save customer location form submission"""

    whatsapp_phone_number: str = Field(
        ...,
        description="WhatsApp phone number of the person filling the form (from webhook)",
        pattern=r"^\+?[1-9]\d{1,14}$",
        examples=["+919643524080", "9643524080"],
    )

    address: str = Field(
        ...,
        description="Full address entered in the form",
        min_length=5,
        examples=["123, MG Road, Sector 14", "Flat 402, Building A, Green Valley"],
    )

    city: Optional[str] = Field(
        None,
        description="City name",
        examples=["Gurgaon", "Delhi", "Mumbai"],
    )

    state: Optional[str] = Field(
        None,
        description="State name",
        examples=["Haryana", "Delhi", "Maharashtra"],
    )

    pincode: Optional[str] = Field(
        None,
        description="PIN code",
        pattern=r"^\d{6}$",
        examples=["122001", "110001"],
    )

    landmark: Optional[str] = Field(
        None,
        description="Nearby landmark for easy navigation",
        examples=["Near City Mall", "Opposite Metro Station"],
    )

    location_type: str = Field(
        default="delivery",
        description="Type of location (delivery, billing, etc.)",
        examples=["delivery", "billing"],
    )

    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional metadata for the location submission",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "whatsapp_phone_number": "+919643524080",
                "address": "123, MG Road, Sector 14",
                "city": "Gurgaon",
                "state": "Haryana",
                "pincode": "122001",
                "landmark": "Near Cyber Hub",
                "location_type": "delivery",
                "metadata": {"verified": True},
            }
        }


class OrderModificationRequest(BaseModel):
    """Request for modifying an existing order"""

    message: str = Field(..., description="Customer's modification request message")
    phone_number: str = Field(..., description="Customer phone number")
    session_id: Optional[str] = Field(None, description="Session ID")

    class Config:
        json_schema_extra = {
            "example": {
                "message": "Change quantity to 2",
                "phone_number": "+919643524080",
                "session_id": "whatsapp_+919643524080",
            }
        }
