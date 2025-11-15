"""
Intent Classification Router
Extracted from original main.py - handles WhatsApp message intent detection
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Literal, Optional, Dict, Any, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from config import settings
from db.supabase_client import get_supabase_client
import logging

# Initialize logger
logger = logging.getLogger(__name__)

# Initialize Supabase client (lazy-load)
supabase_client = None
try:
    supabase_client = get_supabase_client()
except Exception as e:
    logger.warning(f"Failed to initialize Supabase client: {e}")
    supabase_client = None

router = APIRouter(prefix="", tags=["Intent Classification"])


class IntentClassification(BaseModel):
    """Structured output model for intent classification using Literal types"""

    detected_intent: Literal[
        "welcome_greeting",
        "availability_check_with_link",
        "order_confirmation_simple",
        "order_modification",
        "order_address_collect",
        "order_payment_choice",
        "order_payment_confirmed",
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
    user_message: str,
    message_type: str,
    active_flow: str,
    last_message: str,
    chat_history: Optional[List[Dict[str, Any]]] = None
) -> IntentClassification:
    """
    Use LangChain + Gemini to classify user intent with structured output

    Args:
        user_message: The current user message
        message_type: Type of message (text, image, audio, etc.)
        active_flow: The current active conversation flow
        last_message: The last message sent
        chat_history: Optional list of previous messages from agent_messages table
    """

    # Format chat history for the prompt
    chat_context = ""
    if chat_history and len(chat_history) > 0:
        chat_context = "\n**Previous Conversation History (recent messages, newest first):**\n"
        # Reverse to show oldest first for better context
        for msg in reversed(chat_history[-10:]):  # Last 10 messages
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            chat_context += f"- {role}: {content}\n"
    else:
        chat_context = "\n**Previous Conversation History:** No previous messages available.\n"

    # Create the prompt template using ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert intent classifier for a WhatsApp-based e-commerce customer service system for "Star Trade" company.

Your job is to analyze the customer's message and classify it into one of the predefined intents.

**Intent Definitions:**

1. **welcome_greeting**: User is greeting or starting conversation (hi, hello, hey, namaste, good morning/evening, asking "is this Star Trade?")

2. **availability_check_with_link**: User asking about product availability, price, rate, discount, if a product/tool is available, OR requesting alternate products, similar items, other options ("kuch aur", "similar", "alternate"). This intent first checks if the product is available.

3. **order_confirmation_simple**: User wants to confirm order, place order, book something (includes "order kardo", "book", etc.)

4. **order_modification**: User wants to modify their existing order. **IMPORTANT**: Only classify as this intent when user EXPLICITLY mentions changing specific order details:
   - Changing quantity of an item ("change quantity to 2", "make it 3 pieces", "reduce to 1")
   - Removing a specific product from order ("remove the drill", "cancel the stapler item", "don't want the wrench")
   - Canceling entire order ("cancel my order", "cancel the whole order", "don't want anything")
   - **DO NOT** classify as order_modification if user just says "modify", "change", "edit" without specifying WHAT to change
   - **DO NOT** classify as order_modification if user wants to ADD new items (that should be availability_check_with_link)

5. **order_address_collect**: User is providing or being asked for delivery address, pincode, city, full name for delivery

6. **order_payment_choice**: User discussing payment method (COD, UPI, bank transfer, cash on delivery, pay on delivery)

7. **order_payment_confirmed**: User confirming payment is done ("paid", "payment done", "transfer ho gaya", "I have paid")

8. **out_of_stock_simple**: Discussion about out of stock items, product not available

9. **product_availability_query**: When user is already in availability_check_with_link flow and sends a product name or inquiry

10. **order_confirmation_approval**: When user is in order_confirmation_simple flow and responds with yes/confirm/ok/done

11. **general_text_message**: Any other text message that doesn't fit the above categories

12. **unknown**: When message type is not text or intent cannot be determined

**Context-Aware Rules:**
- Use the conversation history to understand the context of the current message
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

{chat_context}

**Current User Message:** {user_message}

Based on the conversation history and current context, classify the intent of the current user message.""")
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
            chat_context=chat_context,
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
    Now includes previous chat context from agent_messages table
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

        # Get phone number and generate session_id
        phone_number = customer.phone_number or contact.phone_number or "unknown"
        if phone_number != "unknown" and not phone_number.startswith("+"):
            phone_number = f"+91{phone_number}"
        session_id = f"whatsapp_{phone_number}"

        # Retrieve recent chat history from agent_messages table
        chat_history = []
        if supabase_client and phone_number != "unknown":
            try:
                # Get or create session
                session = await supabase_client.get_or_create_session(phone_number)
                logger.info(f"Session retrieved/created: {session.get('session_id')}")

                # Retrieve recent messages (last 10 for context)
                chat_history = await supabase_client.get_recent_messages(
                    session_id=session_id, limit=10
                )
                logger.info(f"Retrieved {len(chat_history)} messages from history for intent classification")
            except Exception as e:
                logger.warning(f"Failed to retrieve chat history for intent classification: {e}")
                chat_history = []

        # Check if quick message (button-based)
        check_if_quick_message = bool(
            msg.message_context or (msg.button_text and msg.button_payload)
        )

        # Use AI to classify intent with chat history
        if message_type == "text" and user_message:
            classification = classify_intent_with_ai(
                user_message=user_message,
                message_type=message_type,
                active_flow=active_flow,
                last_message=last_message,
                chat_history=chat_history,
            )
            detected_intent = classification.detected_intent
        else:
            detected_intent = "unknown"

        # Prepare response matching Code1 output structure
        response = IntentResponse(
            phone_number=phone_number,
            user_message=user_message,
            message_type=message_type,
            detected_intent=detected_intent,
            active_template_flow=active_flow,
            last_message=last_message,
            full_context={
                "contact_record": contact.model_dump() if contact else {},
                "webhook_payload": webhook.model_dump() if webhook else {},
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
