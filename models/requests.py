"""
Pydantic request models for WhatsApp AI Service
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


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
