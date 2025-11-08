"""
Intent Classification Router
Extracted from original main.py - handles WhatsApp message intent detection
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Literal, Optional, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from config import settings

router = APIRouter(prefix="", tags=["Intent Classification"])


class IntentClassification(BaseModel):
    """Structured output model for intent classification using Literal types"""

    detected_intent: Literal[
        "welcome_greeting",
        "availability_check_with_link",
        "order_confirmation_simple",
        "order_address_collect",
        "order_payment_choice",
        "order_payment_confirmed",
        "alternate_suggestion_simple",
        "out_of_stock_simple",
        "order_confirmation_approval",
        "general_text_message",
        "unknown",
    ] = Field(description="The classified intent of the user's message")


class ContactRecord(BaseModel):
    """Contact information from Supabase"""

    phone_number: Optional[str] = None
    active_template_flow: Optional[str] = "none"
    last_message: Optional[str] = ""


class MessageData(BaseModel):
    """WhatsApp message data"""

    message_content_type: Optional[str] = "text"
    message: Optional[str] = ""
    button_text: Optional[str] = None
    button_payload: Optional[Dict[str, Any]] = None
    message_context: Optional[Dict[str, Any]] = None


class CustomerData(BaseModel):
    """Customer information from webhook"""

    phone_number: Optional[str] = None


class WebhookData(BaseModel):
    """Webhook payload structure"""

    customer: Optional[CustomerData] = None
    message: Optional[MessageData] = None


class WebhookBody(BaseModel):
    """Webhook body wrapper"""

    data: Optional[WebhookData] = None


class WebhookPayload(BaseModel):
    """Full webhook payload"""

    body: Optional[WebhookBody] = None


class IntentRequest(BaseModel):
    """Input request matching Code1 node input structure"""

    contact_record: Optional[ContactRecord] = None
    webhook_payload: Optional[WebhookPayload] = None


class IntentResponse(BaseModel):
    """Output response matching Code1 node output structure"""

    phone_number: str
    user_message: str
    message_type: str
    detected_intent: str
    active_template_flow: str
    last_message: str
    full_context: Dict[str, Any]
    check_if_quick_message: bool


# ============================================================
# INTENT CLASSIFICATION LOGIC
# ============================================================


def get_gemini_llm():
    """Initialize Gemini LLM via LangChain"""
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", google_api_key=settings.GOOGLE_API_KEY
    )


def classify_intent_with_ai(
    user_message: str, message_type: str, active_flow: str, last_message: str
) -> IntentClassification:
    """
    Use LangChain + Gemini to classify user intent with structured output
    """

    # Create the prompt template using ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert intent classifier for a WhatsApp-based e-commerce customer service system for "Star Trade" company.

Your job is to analyze the customer's message and classify it into one of the predefined intents.

**Intent Definitions:**

1. **welcome_greeting**: User is greeting or starting conversation (hi, hello, hey, namaste, good morning/evening, asking "is this Star Trade?")

2. **availability_check_with_link**: User asking about product availability, price, rate, discount, or if a product/tool is available

3. **order_confirmation_simple**: User wants to confirm order, place order, book something (includes "order kardo", "book", etc.)

4. **order_address_collect**: User is providing or being asked for delivery address, pincode, city, full name for delivery

5. **order_payment_choice**: User discussing payment method (COD, UPI, bank transfer, cash on delivery, pay on delivery)

6. **order_payment_confirmed**: User confirming payment is done ("paid", "payment done", "transfer ho gaya", "I have paid")

7. **alternate_suggestion_simple**: User wants alternate products, similar items, other options ("kuch aur", "similar", "alternate")

8. **out_of_stock_simple**: Discussion about out of stock items, product not available

9. **product_availability_query**: When user is already in availability_check_with_link flow and sends a product name or inquiry

10. **order_confirmation_approval**: When user is in order_confirmation_simple flow and responds with yes/confirm/ok/done

11. **general_text_message**: Any other text message that doesn't fit the above categories

12. **unknown**: When message type is not text or intent cannot be determined

**Context-Aware Rules:**
- If active_flow is "availability_check_with_link" and message_type is "text", likely intent is "product_availability_query"
- If active_flow is "order_confirmation_simple" and user says yes/confirm/ok/done, intent is "order_confirmation_approval"
- If active_flow is "none" and no other intent matches, use "general_text_message"

**Language Support:**
- Support both English and Hindi/Hinglish text

Classify the intent based on the context and message above."""),
        ("user", """**Context Information:**
- Current conversation flow: {active_flow}
- Previous message: {last_message}
- Message type: {message_type}
- User message: {user_message}""")
    ])

    # Initialize LLM with structured output
    llm = get_gemini_llm()
    structured_llm = llm.with_structured_output(IntentClassification)

    # Execute classification using direct invoke (no LCEL chain)
    result = structured_llm.invoke(
        prompt.format_messages(
            user_message=user_message,
            message_type=message_type,
            active_flow=active_flow,
            last_message=last_message,
        )
    )

    return result


# ============================================================
# API ENDPOINTS
# ============================================================


@router.post("/classify-intent", response_model=IntentResponse)
async def classify_intent(request: IntentRequest):
    """
    Main endpoint to classify WhatsApp message intent
    Replaces the Code1 node in n8n workflow
    """

    try:
        # Extract contact record
        contact = request.contact_record or ContactRecord()

        # Extract webhook payload
        webhook = request.webhook_payload
        if not webhook or not webhook.body or not webhook.body.data:
            raise HTTPException(
                status_code=400, detail="Invalid webhook payload structure"
            )

        data = webhook.body.data
        msg = data.message or MessageData()
        customer = data.customer or CustomerData()

        # Prepare variables
        message_type = "unknown"
        user_message = ""

        # Identify message type
        raw_type = (msg.message_content_type or "").lower()
        if "text" in raw_type:
            message_type = "text"
        elif "image" in raw_type:
            message_type = "image"
        elif "audio" in raw_type:
            message_type = "audio"
        else:
            message_type = raw_type or "unknown"

        # Extract user message
        if message_type == "text" and msg.message:
            user_message = msg.message.strip()

        # Get context info
        active_flow = contact.active_template_flow or "none"
        last_message = contact.last_message or ""

        # Check if quick message (button-based)
        check_if_quick_message = bool(
            msg.message_context or (msg.button_text and msg.button_payload)
        )

        # Use AI to classify intent
        if message_type == "text" and user_message:
            classification = classify_intent_with_ai(
                user_message=user_message,
                message_type=message_type,
                active_flow=active_flow,
                last_message=last_message,
            )
            detected_intent = classification.detected_intent
        else:
            detected_intent = "unknown"

        # Prepare response matching Code1 output structure
        response = IntentResponse(
            phone_number=customer.phone_number or contact.phone_number or "unknown",
            user_message=user_message,
            message_type=message_type,
            detected_intent=detected_intent,
            active_template_flow=active_flow,
            last_message=last_message,
            full_context={
                "contact_record": contact.dict() if contact else {},
                "webhook_payload": webhook.dict() if webhook else {},
            },
            check_if_quick_message=check_if_quick_message,
        )

        return response

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error classifying intent: {str(e)}"
        )


@router.post("/test-classification")
async def test_classification(
    user_message: str, active_flow: str = "none", last_message: str = ""
):
    """
    Test endpoint for quick intent classification testing
    """
    try:
        classification = classify_intent_with_ai(
            user_message=user_message,
            message_type="text",
            active_flow=active_flow,
            last_message=last_message,
        )
        return classification
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
