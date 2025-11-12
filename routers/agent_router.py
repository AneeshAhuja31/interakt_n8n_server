"""
Agent Router - LangGraph Availability Checking Agent
Replaces the n8n AI Agent cluster (8 nodes) with a single HTTP endpoint
"""

from fastapi import APIRouter, HTTPException
from models.requests import (
    AgentRequest,
    AgentRequestFromN8N,
    SaveMessageRequest,
    OrderConfirmationRequest,
    CustomerFormRequest,
    CustomerLocationFormRequest,
)
from models.responses import (
    AgentResponse,
    WhatsAppTemplatePayload,
    ProductMatch,
    SaveMessageResponse,
    OrderConfirmationResponse,
    OrderSummary,
    LatestOrderResponse,
    CustomerFormSubmission,
    CustomerFormResponse,
    CustomerLocationSubmission,
    CustomerLocationFormResponse,
)
from agents.availability_agent import get_availability_agent
from db.supabase_client import get_supabase_client
from config import settings
import time
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["Agent"])

# Initialize agent (singleton)
availability_agent = get_availability_agent()

# Lazy-load Supabase client (only when needed)
supabase_client = None
try:
    supabase_client = get_supabase_client()
except Exception as e:
    supabase_client = None


@router.post("/availability", response_model=AgentResponse)
async def check_availability(request: AgentRequest):
    """
    LangGraph-powered product availability agent

    **Replaces the n8n AI Agent cluster (8 nodes):**
    - AI Agent
    - Simple Memory
    - OpenAI Chat Model
    - Qdrant Vector Store
    - Embeddings OpenAI
    - RAG Db (Airtable)
    - F&T Brad Nails (Airtable)
    - Prepare Availability Template Payload

    **Workflow:**
    1. Search product catalog using Qdrant vector similarity
    2. Retrieve top K matching products
    3. LLM decides availability status (in_stock/out_of_stock/alternate_available)
    4. Format response for WhatsApp template
    5. Save conversation state to Postgres

    **Args:**
        request: AgentRequest with message, phone_number, session_id

    **Returns:**
        AgentResponse with WhatsApp template payload and metadata
    """
    start_time = time.time()

    try:
        # Ensure phone number has country code
        phone_number = request.phone_number
        if not phone_number.startswith("+"):
            phone_number = f"+91{phone_number}"

        # Generate or use provided session ID
        session_id = request.session_id or f"whatsapp_{phone_number}"

        

        # Get or create session in database
        session = {"session_id": session_id}
        if supabase_client:
            try:
                session = await supabase_client.get_or_create_session(phone_number)
                logger.info(f"Session retrieved/created: {session.get('session_id')}")
            except Exception as e:
                logger.warning(f"Failed to create session in DB: {e}. Continuing without session tracking.")

        # Prepare initial agent state
        initial_state = {
            "customer_message": request.message,
            "phone_number": phone_number,
            "session_id": session_id,
            "messages": [],  # Will be populated from DB or empty for new sessions
            "processing_start": start_time,
            "llm_calls": 0,
            "retrieved_products": [],
            "top_product": {},
        }

        # Run the LangGraph agent
        config = {
            "configurable": {"thread_id": session_id},
            "recursion_limit": 10,
        }

        result = await availability_agent.ainvoke(initial_state, config)

        logger.info(
            f"[/agent/availability] Agent completed: status={result.get('availability_status')}, product={result.get('matched_product')}"
        )

        # Log the customer message to database
        if supabase_client:
            try:
                await supabase_client.add_message(
                    session_id=session_id,
                    role="human",
                    content=request.message,
                    message_type="text",
                )
            except Exception as e:
                print(f"Failed to log customer message: {e}")

        if supabase_client:
            try:
                await supabase_client.add_message(
                    session_id=session_id,
                    role="ai",
                    content=result.get("hinglish_output", ""),
                    message_type="text",
                )
            except Exception as e:
                logger.warning(f"Failed to log AI message: {e}")

        # Extract phone number without country code for WhatsApp template
        phone_without_code = phone_number.lstrip("+").lstrip("91")

        # Format WhatsApp template payload (matching n8n format)
        template_payload = WhatsAppTemplatePayload(
            countryCode="+91",
            phoneNumber=phone_without_code,
            template={
                "name": "availability_check_with_link",
                "languageCode": "en",
                "bodyValues": [
                    result.get("matched_product", "Unknown Product"),
                    result.get("price", "N/A"),
                    result.get("discount", "No discount"),
                    result.get("brand", "Unknown"),
                    result.get("specs", ""),
                    result.get("product_url", ""),
                ],
            },
            image_url=result.get("image_url", ""),
            summary_output=result.get("hinglish_output", ""),
            status=result.get("availability_status", "out_of_stock"),
        )

        # Convert retrieved_products to ProductMatch models
        product_matches = [
            ProductMatch(
                name=p.get("name", "Unknown"),
                brand=p.get("brand", "Unknown"),
                price=p.get("price", "N/A"),
                discount=p.get("discount", "No discount"),
                specs=p.get("specs", ""),
                product_url=p.get("product_url", ""),
                image_url=p.get("image_url", ""),
                similarity_score=p.get("similarity_score", 0.0),
            )
            for p in result.get("retrieved_products", [])
        ]

        # Calculate processing time
        processing_time_ms = int((time.time() - start_time) * 1000)

        # Log product query to analytics
        if supabase_client:
            try:
                await supabase_client.log_product_query(
                    session_id=session_id,
                    phone_number=phone_number,
                    query_text=request.message,
                    matched_product=result.get("matched_product"),
                    brand=result.get("brand"),
                    availability_status=result.get("availability_status"),
                    retrieved_products=[p.dict() for p in product_matches],
                    response_time_ms=processing_time_ms,
                    success=not bool(result.get("error")),
                    error_message=result.get("error"),
                    metadata={
                        "llm_calls": result.get("llm_calls", 0),
                        "qdrant_search_time_ms": result.get("qdrant_search_time_ms", 0),
                    },
                )
            except Exception as e:
                logger.warning(f"Failed to log query analytics: {e}")

        # Prepare final response
        response = AgentResponse(
            result=template_payload,
            session_id=session_id,
            originalRequest=request.message,
            retrieved_products=product_matches,
            processing_time_ms=processing_time_ms,
            metadata={
                "top_product": result.get("top_product", {}),
                "llm_calls": result.get("llm_calls", 0),
                "qdrant_search_time_ms": result.get("qdrant_search_time_ms", 0),
                "reasoning": result.get("reasoning", ""),
                "error": result.get("error"),
            },
        )

        

        return response

    except Exception as e:
        logger.error(f"[/agent/availability] Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")


@router.post("/availability/n8n", response_model=AgentResponse)
async def check_availability_n8n_format(request: AgentRequestFromN8N):
    """
    Alternative endpoint that accepts n8n webhook format

    **Args:**
        request: AgentRequestFromN8N with webhook_payload and contact_record

    **Returns:**
        Same as /availability endpoint

    **Example n8n usage:**
    ```json
    {
      "webhook_payload": {
        "body": {
          "data": {
            "customer": {"phone_number": "9643524080"},
            "message": {"message": "PNEUMATIC STAPLER"}
          }
        }
      },
      "contact_record": {
        "phone_number": "9643524080",
        "active_template_flow": "none"
      }
    }
    ```
    """
    try:
        # Convert n8n format to AgentRequest
        agent_request = request.to_agent_request()

        # Call the main availability endpoint
        return await check_availability(agent_request)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"[/agent/availability/n8n] Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")


@router.get("/health")
async def agent_health_check():
    """
    Health check endpoint for the agent service

    **Returns:**
        Status information about the agent
    """
    return {
        "status": "healthy",
        "service": "LangGraph Availability Agent",
        "version": "2.0.0",
        "model": settings.GEMINI_MODEL,
        "vector_store": "Qdrant",
        "collection": settings.QDRANT_COLLECTION,
    }


@router.post("/save-message", response_model=SaveMessageResponse)
async def save_message(request: SaveMessageRequest):
    """
    Save a message to chat history

    **Purpose:**
    Store user and AI messages to maintain conversation context across all workflow paths.
    This endpoint should be called from n8n HTTP Request nodes throughout the workflow.

    **Workflow Integration Points:**
    1. After user sends any message (text/audio/image)
    2. After user clicks any button (quick reply)
    3. After AI sends any template response

    **Args:**
        request: SaveMessageRequest with phone_number, role, content, message_type, metadata

    **Returns:**
        SaveMessageResponse with success status and message ID
    """
    try:
        # Ensure phone number has country code
        phone_number = request.phone_number
        if not phone_number.startswith("+"):
            phone_number = f"+91{phone_number}"

        # Generate session ID
        session_id = f"whatsapp_{phone_number}"

        if not supabase_client:
            raise HTTPException(
                status_code=503,
                detail="Database client not available. Check Supabase configuration.",
            )

        # Get or create session
        try:
            session = await supabase_client.get_or_create_session(phone_number)
            logger.info(f"Session retrieved/created: {session.get('session_id')}")
        except Exception as e:
            logger.warning(f"Failed to create session: {e}")
            # Continue anyway with generated session_id

        # Save message to database
        try:
            await supabase_client.add_message(
                session_id=session_id,
                role=request.role,
                content=request.content,
                message_type=request.message_type,
                metadata=request.metadata,
            )
            logger.info(
                f"[/agent/save-message] Saved {request.role} message for {phone_number}: {request.content[:50]}..."
            )

            return SaveMessageResponse(
                success=True,
                session_id=session_id,
                message_id=None,  # Supabase client doesn't return ID currently
                message="Message saved successfully",
            )

        except Exception as e:
            logger.error(f"Failed to save message: {e}", exc_info=True)
            raise HTTPException(
                status_code=500, detail=f"Failed to save message: {str(e)}"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[/agent/save-message] Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


def preprocess_order_context(messages: list, current_message: str) -> dict:
    """
    Preprocess chat history to extract only order-relevant context with intelligent filtering

    This function filters and processes conversation history to identify:
    1. Product discussions (availability, pricing, specs)
    2. Order intent messages
    3. Removes greetings, unrelated queries, and general messages
    4. **NEW**: Intelligently identifies the CURRENT order context by:
       - Finding the most recent product discussion
       - Detecting conversation boundaries (previous order completions)
       - Identifying explicit confirmation signals
       - Excluding items from previous orders or inquiries

    Args:
        messages: List of message dicts from agent_messages table (ordered newest to oldest)
        current_message: Current user message confirming order

    Returns:
        dict with filtered_messages, product_context, and order_relevant_history
    """
    # Define keywords that indicate order-relevant messages
    order_keywords = [
        "price", "rate", "cost", "available", "stock", "discount", "₹", "rs",
        "order", "book", "confirm", "quantity", "specs", "product", "tool",
        "stapler", "nail", "gun", "hammer", "drill", "wrench", "bolt"
    ]

    greeting_keywords = [
        "hello", "hi", "hey", "namaste", "good morning", "good evening"
    ]

    # Keywords that indicate order confirmation/completion (conversation boundaries)
    order_completion_keywords = [
        "order confirmed", "order placed", "confirmed your order",
        "order summary", "order has been", "thank you for your order",
        "order details", "order id", "order number"
    ]

    # Keywords that indicate user wants to proceed with current order
    confirmation_signals = [
        "yes", "yeah", "sure", "okay", "ok", "confirm", "book",
        "i want", "i'll take", "proceed", "go ahead", "correct",
        "right", "that's correct", "sounds good"
    ]

    # Step 1: Find conversation boundary (last order completion)
    # Messages are ordered newest to oldest, so we need to reverse to find boundaries
    messages_oldest_first = list(reversed(messages))
    last_order_boundary_idx = -1

    for idx, msg in enumerate(messages_oldest_first):
        content = msg.get("content", "").lower()
        role = msg.get("role", "")

        # Look for AI messages indicating previous order completion
        if role == "ai" and any(keyword in content for keyword in order_completion_keywords):
            last_order_boundary_idx = idx
            logger.info(f"[preprocess] Found order boundary at index {idx}: {content[:50]}...")

    # Only consider messages AFTER the last order boundary
    if last_order_boundary_idx >= 0:
        messages_oldest_first = messages_oldest_first[last_order_boundary_idx + 1:]
        logger.info(f"[preprocess] Filtered out {last_order_boundary_idx + 1} messages before last order boundary")

    # Step 2: Find the most recent confirmation signal from user
    last_confirmation_idx = -1
    for idx, msg in enumerate(messages_oldest_first):
        content = msg.get("content", "").lower()
        role = msg.get("role", "")

        # Look for user confirmation signals
        if role == "human" and any(signal in content for signal in confirmation_signals):
            last_confirmation_idx = idx
            logger.info(f"[preprocess] Found confirmation signal at index {idx}: {content}")

    # Step 3: Filter messages based on relevance
    filtered_messages = []

    for idx, msg in enumerate(messages_oldest_first):
        content = msg.get("content", "").lower()
        role = msg.get("role", "")

        # Skip empty messages
        if not content.strip():
            continue

        # Skip pure greeting messages
        if role == "human":
            is_greeting_only = any(
                greeting in content and len(content.split()) <= 4
                for greeting in greeting_keywords
            )
            if is_greeting_only:
                continue

        # Check if message contains order-relevant keywords
        is_relevant = any(keyword in content for keyword in order_keywords)

        # Strategy: Prioritize messages AFTER the last confirmation signal
        # If there's a confirmation signal, we assume the product discussion just before it
        # is what the user wants to order
        if last_confirmation_idx >= 0:
            # Look back a few messages before confirmation to get product context
            context_window_start = max(0, last_confirmation_idx - 5)
            if idx >= context_window_start:
                if role == "ai" or is_relevant:
                    filtered_messages.append(msg)
        else:
            # No explicit confirmation found, use recency-based filtering
            # Prioritize the last 7-8 messages (most recent product discussion)
            if len(messages_oldest_first) - idx <= 8:
                if role == "ai" or is_relevant:
                    filtered_messages.append(msg)

    # If we have very few filtered messages, fall back to broader filtering
    if len(filtered_messages) < 2:
        logger.warning("[preprocess] Too few filtered messages, using fallback filtering")
        filtered_messages = []
        for msg in messages_oldest_first:
            content = msg.get("content", "").lower()
            role = msg.get("role", "")

            if not content.strip():
                continue

            if role == "human":
                is_greeting_only = any(
                    greeting in content and len(content.split()) <= 4
                    for greeting in greeting_keywords
                )
                if is_greeting_only:
                    continue

            is_relevant = any(keyword in content for keyword in order_keywords)
            if role == "ai" or is_relevant:
                filtered_messages.append(msg)

    # Extract product mentions from AI messages
    product_context = []
    for msg in filtered_messages:
        if msg.get("role") == "ai":
            content = msg.get("content", "")
            if content and len(content) > 10:
                product_context.append(content)

    # Build order-relevant history string (maintain chronological order)
    order_relevant_history = "\n".join(
        [
            f"{msg.get('role', 'unknown').upper()}: {msg.get('content', '')}"
            for msg in filtered_messages
        ]
    )

    logger.info(
        f"[preprocess_order_context] Filtered {len(messages)} messages down to {len(filtered_messages)} order-relevant messages"
    )
    logger.info(
        f"[preprocess_order_context] Found {len(product_context)} AI product responses"
    )

    return {
        "filtered_messages": filtered_messages,
        "product_context": product_context,
        "order_relevant_history": order_relevant_history,
        "message_count": len(filtered_messages),
        "has_confirmation_signal": last_confirmation_idx >= 0,
        "has_order_boundary": last_order_boundary_idx >= 0,
    }


@router.post("/order-confirmation", response_model=OrderConfirmationResponse)
async def order_confirmation(request: OrderConfirmationRequest):
    """
    Generate order confirmation from chat history using LLM with intelligent preprocessing

    **Purpose:**
    Analyze recent conversation history to extract order details (product, quantity, price)
    and generate a structured order summary for the WhatsApp template.

    **Workflow:**
    1. Retrieve last 15 messages from chat history (agent_messages table)
    2. Save user's confirmation message
    3. **Preprocess chat history** to filter only order-relevant messages:
       - Removes greetings and unrelated queries
       - Keeps product discussions, availability checks, pricing info
       - Focuses on AI responses (product details) and human queries (order intent)
    4. Use LLM to extract order details from filtered conversation
    5. Generate order summary with multiple items support
    6. Save order to customer_orders table (header + items)
    7. Return template body values for n8n

    **Args:**
        request: OrderConfirmationRequest with message, phone_number, session_id

    **Returns:**
        OrderConfirmationResponse with order details and template values
    """
    start_time = time.time()

    try:
        # Ensure phone number has country code
        phone_number = request.phone_number
        if not phone_number.startswith("+"):
            phone_number = f"+91{phone_number}"

        # Generate or use provided session ID
        session_id = request.session_id or f"whatsapp_{phone_number}"

        if not supabase_client:
            raise HTTPException(
                status_code=503,
                detail="Database client not available. Check Supabase configuration.",
            )

        # Get or create session
        try:
            session = await supabase_client.get_or_create_session(phone_number)
            logger.info(f"Session retrieved/created: {session.get('session_id')}")
        except Exception as e:
            logger.warning(f"Failed to get session: {e}")

        # Save user's order confirmation message
        try:
            await supabase_client.add_message(
                session_id=session_id,
                role="human",
                content=request.message,
                message_type="text",
            )
        except Exception as e:
            logger.warning(f"Failed to save user message: {e}")

        # Retrieve recent chat history
        try:
            messages = await supabase_client.get_recent_messages(
                session_id=session_id, limit=15
            )
            logger.info(f"Retrieved {len(messages)} messages from history")
        except Exception as e:
            logger.warning(f"Failed to retrieve message history: {e}")
            messages = []

        # Import required modules
        from langchain_google_genai import ChatGoogleGenerativeAI
        from pydantic import BaseModel, Field as PydanticField
        from typing import Optional, List

        # Define structured output models
        class OrderItemExtraction(BaseModel):
            """Single item in an order"""
            product_name: str = PydanticField(
                ...,
                description="Product name"
            )
            quantity: int = PydanticField(
                1,
                description="Number of items (default 1 if not mentioned)"
            )
            unit_price: str = PydanticField(
                ...,
                description="Unit price in INR (numbers only, no symbols)"
            )
            discount: str = PydanticField(
                "No discount",
                description="Discount information (e.g., '15% off' or 'No discount')"
            )

        class OrderExtraction(BaseModel):
            """Extracted order details from conversation (supports multiple items)"""
            items: List[OrderItemExtraction] = PydanticField(
                ...,
                description="List of items being ordered. Each item should have product_name, quantity (default 1), unit_price, and discount."
            )

        # Preprocess conversation history to extract only order-relevant context
        preprocessed_context = preprocess_order_context(
            messages=messages,
            current_message=request.message
        )

        conversation_history = preprocessed_context["order_relevant_history"]
        filtered_message_count = preprocessed_context["message_count"]

        logger.info(
            f"[order-confirmation] Preprocessed {len(messages)} messages -> {filtered_message_count} order-relevant messages"
        )

        # Check if we have any order-relevant conversation history
        if not conversation_history.strip():
            logger.warning("No order-relevant conversation history found")
            raise HTTPException(
                status_code=400,
                detail="No product discussion found in recent conversation. Please ask about a product first before confirming an order."
            )

        # Use LLM with structured output to extract order details
        llm = ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            temperature=0,
            google_api_key=settings.GOOGLE_API_KEY,
        )

        # Create structured output LLM
        structured_llm = llm.with_structured_output(OrderExtraction)

        # Create extraction prompt for multiple items
        extraction_prompt = f"""Analyze the ORDER-RELEVANT conversation history below and extract order items the customer wants to order in their CURRENT order.

**CRITICAL INSTRUCTIONS:**
1. The conversation history has been INTELLIGENTLY PRE-FILTERED to include ONLY the most recent product discussion
2. Previous orders and old product inquiries have been EXCLUDED from this history
3. Only extract items that the customer is ordering RIGHT NOW (in this current order confirmation)
4. DO NOT extract items from old inquiries that are no longer relevant
5. Focus on the MOST RECENT product discussion and the customer's LATEST confirmation intent

**Extraction Rules:**
Extract MULTIPLE items ONLY if the customer is explicitly ordering more than one product in the CURRENT order. Each item should have:
- product_name: Name of the product (extract from AI responses which contain product details)
- quantity: Number of items (default to 1 if not mentioned for that specific product)
- unit_price: Unit price in INR (numbers only, no ₹ symbol) - look in AI responses for pricing
- discount: Discount information if mentioned (e.g., "15% off" or "No discount") - look in AI responses

**How to Identify CURRENT Order Items:**
- Look for the MOST RECENT AI product response(s) in the conversation
- Match it with the MOST RECENT human confirmation/quantity message
- If multiple products appear, only include those the customer CONFIRMED in their recent messages
- Ignore products mentioned earlier that the customer didn't follow up on

**Examples:**
Example 1 - Single Product:
HUMAN: What's the price of air stapler?
AI: Air Stapler is available at ₹2499 with 10% discount
HUMAN: I want 2
→ Extract: [{{product_name: "Air Stapler", quantity: 2, unit_price: "2499", discount: "10% discount"}}]

Example 2 - Multiple Products (Explicit):
HUMAN: I want 2 air staplers and 3 nail boxes
AI: Air Stapler: ₹2499, Nail Box: ₹500
HUMAN: Yes, confirm
→ Extract: [{{product_name: "Air Stapler", quantity: 2, unit_price: "2499", discount: "No discount"}}, {{product_name: "Nail Box", quantity: 3, unit_price: "500", discount: "No discount"}}]

Example 3 - DO NOT include old products:
(This scenario should NOT appear in pre-filtered history, but if it does:)
HUMAN: Tell me about hammer [OLD - should be filtered out]
AI: Hammer costs ₹800 [OLD - should be filtered out]
HUMAN: What about pneumatic stapler? [RECENT]
AI: Pneumatic Stapler costs ₹4500 with 15% off [RECENT]
HUMAN: I'll take one [RECENT]
→ Extract ONLY: [{{product_name: "Pneumatic Stapler", quantity: 1, unit_price: "4500", discount: "15% off"}}]
→ DO NOT extract: Hammer (it was from an old inquiry)

**Order-Relevant Conversation History (Pre-filtered for CURRENT order only):**
{conversation_history}

**Current User Message:** {request.message}

Now extract ONLY the items from the CURRENT order. Focus on the most recent AI product response(s) and the customer's latest confirmation."""

        try:
            # Get structured output from LLM
            order_data = await structured_llm.ainvoke(extraction_prompt)
            logger.info(f"Extracted order data: {order_data}")

            # Validate that we have items
            if not order_data.items or len(order_data.items) == 0:
                raise ValueError("No items extracted from conversation")

            # Process each item and calculate subtotals
            order_items = []
            total_price = 0

            for item in order_data.items:
                # Calculate subtotal for this item
                try:
                    unit_price_int = int(item.unit_price) if item.unit_price.isdigit() else 0
                    quantity = item.quantity or 1
                    subtotal = unit_price_int * quantity
                except (ValueError, AttributeError):
                    subtotal = 0

                order_items.append({
                    "product_name": item.product_name,
                    "quantity": item.quantity or 1,
                    "unit_price": item.unit_price,
                    "discount": item.discount or "No discount",
                    "subtotal": str(subtotal),
                })

                total_price += subtotal

            logger.info(f"Processed {len(order_items)} items, total price: {total_price}")

        except Exception as e:
            logger.error(f"LLM extraction failed: {e}", exc_info=True)
            # Fallback: Create default single item
            order_items = [{
                "product_name": "Product",
                "quantity": 1,
                "unit_price": "0",
                "discount": "No discount",
                "subtotal": "0",
            }]
            total_price = 0

        # Generate order ID and item IDs with millisecond precision
        from datetime import datetime
        import uuid

        # Format: ORD_YYYYMMDD_HHMMSS_mmm (e.g., ORD_20250110_143052_789)
        now = datetime.now()
        timestamp = now.strftime('%Y%m%d_%H%M%S')
        milliseconds = now.strftime('%f')[:3]  # First 3 digits of microseconds
        order_id = f"ORD_{timestamp}_{milliseconds}"

        # Create OrderItem objects
        from models.responses import OrderItem
        order_item_objects = []

        for idx, item in enumerate(order_items):
            item_id = f"ITEM_{timestamp}_{milliseconds}_{idx+1:03d}"
            order_item_objects.append(
                OrderItem(
                    product_name=item["product_name"],
                    quantity=item["quantity"],
                    unit_price=item["unit_price"],
                    discount=item["discount"],
                    subtotal=item["subtotal"],
                    item_id=item_id,
                )
            )
            # Add item_id to the dict for database insertion
            item["item_id"] = item_id

        # Create OrderSummary object with multiple items
        order_summary = OrderSummary(
            items=order_item_objects,
            total_price=str(total_price),
            order_id=order_id,
        )

        # Save order to database (new multi-item structure)
        try:
            # Create order header
            await supabase_client.create_order_header(
                order_id=order_id,
                session_id=session_id,
                phone_number=phone_number,
                total_price=str(total_price),
                metadata={"item_count": len(order_items)},
            )
            logger.info(f"Order header saved to database: {order_id}")

            # Create order items
            await supabase_client.create_order_items(
                order_id=order_id,
                items=order_items,
                phone_number=phone_number,
            )
            logger.info(f"Saved {len(order_items)} items to database for order {order_id}")

        except Exception as e:
            logger.warning(f"Failed to save order to database: {e}")

        # Save order summary as AI message (with all items)
        items_text = ", ".join([
            f"{item.product_name} x{item.quantity}"
            for item in order_summary.items
        ])
        order_summary_text = f"Order Summary: {items_text} | Total: ₹{order_summary.total_price}"
        try:
            await supabase_client.add_message(
                session_id=session_id,
                role="ai",
                content=order_summary_text,
                message_type="order_summary",
                metadata={"order_id": order_id, "item_count": len(order_summary.items)},
            )
        except Exception as e:
            logger.warning(f"Failed to save order summary message: {e}")

        # Prepare template body values for WhatsApp
        # Format: "Product1 (x2), Product2 (x1)", "Total", "Order ID"
        items_summary = ", ".join([
            f"{item.product_name} (x{item.quantity})"
            for item in order_summary.items
        ])
        template_body_values = [
            items_summary,
            f"₹{order_summary.total_price}",
            order_id,
        ]

        processing_time_ms = int((time.time() - start_time) * 1000)
        logger.info(
            f"[/agent/order-confirmation] Order processed in {processing_time_ms}ms: {order_id}"
        )

        return OrderConfirmationResponse(
            success=True,
            order=order_summary,
            template_body_values=template_body_values,
            session_id=session_id,
            message="Order confirmed successfully",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[/agent/order-confirmation] Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Order confirmation error: {str(e)}")


@router.get("/latest-order/{phone_number}", response_model=LatestOrderResponse)
async def get_latest_order(phone_number: str):
    """
    Get the latest order for a customer by phone number

    **Purpose:**
    Retrieve the most recent order with all items and generate a single-line summary.

    **Workflow:**
    1. Ensure phone number has country code
    2. Get latest order header from order_headers table
    3. Get all items for that order from order_items table
    4. Build OrderSummary with multiple items
    5. Generate single-line order summary text (no newlines)

    **Args:**
        phone_number: Customer phone number

    **Returns:**
        LatestOrderResponse with order details and summary text

    **Example:**
        GET /agent/latest-order/+919643524080
        GET /agent/latest-order/9643524080
    """
    try:
        # Ensure phone number has country code
        if not phone_number.startswith("+"):
            phone_number = f"+91{phone_number}"

        if not supabase_client:
            raise HTTPException(
                status_code=503,
                detail="Database client not available. Check Supabase configuration.",
            )

        # Get latest order header
        order_headers = await supabase_client.get_customer_order_headers(
            phone_number=phone_number,
            limit=1
        )

        if not order_headers or len(order_headers) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No orders found for phone number {phone_number}"
            )

        latest_order_header = order_headers[0]
        order_id = latest_order_header.get("order_id")

        logger.info(f"[/agent/latest-order] Found order {order_id} for {phone_number}")

        # Get all items for this order
        response = (
            supabase_client.client.table("order_items")
            .select("*")
            .eq("order_id", order_id)
            .order("created_at", desc=False)
            .execute()
        )

        if not response.data or len(response.data) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No items found for order {order_id}"
            )

        order_items_data = response.data

        # Build OrderItem objects
        from models.responses import OrderItem
        order_items = []
        for item_data in order_items_data:
            order_items.append(
                OrderItem(
                    product_name=item_data.get("product_name", ""),
                    quantity=item_data.get("quantity", 1),
                    unit_price=str(item_data.get("unit_price", "0")),
                    discount=item_data.get("discount", "No discount"),
                    subtotal=str(item_data.get("subtotal", "0")),
                    item_id=item_data.get("item_id"),
                )
            )

        # Build OrderSummary
        total_price = latest_order_header.get("total_price", "0")
        order_summary = OrderSummary(
            items=order_items,
            total_price=str(total_price),
            order_id=order_id,
        )

        # Generate single-line summary text (no newlines)
        items_summary_parts = []
        for item in order_items:
            item_text = f"{item.product_name} x{item.quantity} (₹{item.unit_price} each"
            if item.discount and item.discount != "No discount":
                item_text += f", {item.discount}"
            item_text += ")"
            items_summary_parts.append(item_text)

        items_summary = ", ".join(items_summary_parts)
        order_summary_text = f"Order ID: {order_id} | Items: {items_summary} | Total: ₹{total_price}"

        logger.info(f"[/agent/latest-order] Generated summary for {order_id}")

        return LatestOrderResponse(
            success=True,
            order=order_summary,
            order_summary_text=order_summary_text,
            message="Latest order retrieved successfully",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[/agent/latest-order] Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@router.post("/customer-form", response_model=CustomerFormResponse)
async def save_customer_form(request: CustomerFormRequest):
    """
    Save customer form submission (name, email, phone number)

    **Purpose:**
    Store form data submitted by WhatsApp user with timestamped form ID.

    **Form Fields:**
    - Name (entered by user)
    - Email (entered by user)
    - Phone number (entered by user - may differ from WhatsApp number)

    **Unique ID Format:**
    FORM_{last10digitsOfWhatsAppNumber}_{YYYYMMDD_HHMMSS_mmm}
    Example: FORM_9643524080_20250110_143052_789

    **Args:**
        request: CustomerFormRequest with whatsapp_phone_number, entered_name, entered_email, entered_phone_number

    **Returns:**
        CustomerFormResponse with form submission details and form_id
    """
    try:
        # Ensure WhatsApp phone number has country code
        whatsapp_phone = request.whatsapp_phone_number
        if not whatsapp_phone.startswith("+"):
            whatsapp_phone = f"+91{whatsapp_phone}"

        if not supabase_client:
            raise HTTPException(
                status_code=503,
                detail="Database client not available. Check Supabase configuration.",
            )

        # Save form to database
        try:
            form_record = await supabase_client.save_customer_form(
                whatsapp_phone_number=whatsapp_phone,
                entered_name=request.entered_name,
                entered_email=request.entered_email,
                entered_phone_number=request.entered_phone_number,
                metadata=request.metadata,
            )

            logger.info(
                f"[/agent/customer-form] Form saved: {form_record.get('form_id')} for WhatsApp {whatsapp_phone}"
            )

            # Convert to CustomerFormSubmission model
            form_submission = CustomerFormSubmission(
                form_id=form_record.get("form_id"),
                whatsapp_phone_number=form_record.get("whatsapp_phone_number"),
                entered_name=form_record.get("entered_name"),
                entered_email=form_record.get("entered_email"),
                entered_phone_number=form_record.get("entered_phone_number"),
                created_at=form_record.get("created_at"),
                metadata=form_record.get("metadata"),
            )

            return CustomerFormResponse(
                success=True,
                form_submission=form_submission,
                message="Form saved successfully",
            )

        except Exception as e:
            logger.error(f"Failed to save form: {e}", exc_info=True)
            raise HTTPException(
                status_code=500, detail=f"Failed to save form: {str(e)}"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[/agent/customer-form] Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@router.get("/customer-form/{whatsapp_phone_number}", response_model=CustomerFormResponse)
async def get_customer_form(whatsapp_phone_number: str):
    """
    Get the most recent form submission by WhatsApp phone number

    **Args:**
        whatsapp_phone_number: WhatsApp phone number

    **Returns:**
        CustomerFormResponse with most recent form submission
    """
    try:
        # Ensure phone number has country code
        if not whatsapp_phone_number.startswith("+"):
            whatsapp_phone_number = f"+91{whatsapp_phone_number}"

        if not supabase_client:
            raise HTTPException(
                status_code=503,
                detail="Database client not available. Check Supabase configuration.",
            )

        # Get form from database
        form_record = await supabase_client.get_customer_form(whatsapp_phone_number)

        if not form_record:
            raise HTTPException(
                status_code=404, detail=f"No form found for WhatsApp {whatsapp_phone_number}"
            )

        # Convert to CustomerFormSubmission model
        form_submission = CustomerFormSubmission(
            form_id=form_record.get("form_id"),
            whatsapp_phone_number=form_record.get("whatsapp_phone_number"),
            entered_name=form_record.get("entered_name"),
            entered_email=form_record.get("entered_email"),
            entered_phone_number=form_record.get("entered_phone_number"),
            created_at=form_record.get("created_at"),
            metadata=form_record.get("metadata"),
        )

        return CustomerFormResponse(
            success=True,
            form_submission=form_submission,
            message="Form retrieved successfully",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[/agent/customer-form] Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@router.post("/customer-location-form", response_model=CustomerLocationFormResponse)
async def save_customer_location_form(request: CustomerLocationFormRequest):
    """
    Save customer location form submission

    **Purpose:**
    Store location data submitted by WhatsApp user with timestamped location ID.

    **Unique ID Format:**
    LOCATION_{last10digitsOfWhatsAppNumber}_{YYYYMMDD_HHMMSS_mmm}
    Example: LOCATION_9643524080_20250110_143052_789

    **Args:**
        request: CustomerLocationFormRequest with whatsapp_phone_number, address, city, state, etc.

    **Returns:**
        CustomerLocationFormResponse with location submission details and location_id
    """
    try:
        # Ensure WhatsApp phone number has country code
        whatsapp_phone = request.whatsapp_phone_number
        if not whatsapp_phone.startswith("+"):
            whatsapp_phone = f"+91{whatsapp_phone}"

        if not supabase_client:
            raise HTTPException(
                status_code=503,
                detail="Database client not available. Check Supabase configuration.",
            )

        # Save location to database
        try:
            location_record = await supabase_client.save_customer_location_form(
                whatsapp_phone_number=whatsapp_phone,
                address=request.address,
                city=request.city,
                state=request.state,
                pincode=request.pincode,
                landmark=request.landmark,
                location_type=request.location_type,
                metadata=request.metadata,
            )

            logger.info(
                f"[/agent/customer-location-form] Location saved: {location_record.get('location_id')} for WhatsApp {whatsapp_phone}"
            )

            # Convert to CustomerLocationSubmission model
            location_submission = CustomerLocationSubmission(
                location_id=location_record.get("location_id"),
                whatsapp_phone_number=location_record.get("whatsapp_phone_number"),
                address=location_record.get("address"),
                city=location_record.get("city"),
                state=location_record.get("state"),
                pincode=location_record.get("pincode"),
                landmark=location_record.get("landmark"),
                location_type=location_record.get("location_type"),
                created_at=location_record.get("created_at"),
                metadata=location_record.get("metadata"),
            )

            return CustomerLocationFormResponse(
                success=True,
                location_submission=location_submission,
                message="Location saved successfully",
            )

        except Exception as e:
            logger.error(f"Failed to save location: {e}", exc_info=True)
            raise HTTPException(
                status_code=500, detail=f"Failed to save location: {str(e)}"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[/agent/customer-location-form] Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@router.get("/customer-location-form/{whatsapp_phone_number}", response_model=CustomerLocationFormResponse)
async def get_customer_location_form(whatsapp_phone_number: str, location_type: str = "delivery"):
    """
    Get the most recent location submission by WhatsApp phone number

    **Args:**
        whatsapp_phone_number: WhatsApp phone number
        location_type: Type of location (delivery, billing, etc.)

    **Returns:**
        CustomerLocationFormResponse with most recent location submission
    """
    try:
        # Ensure phone number has country code
        if not whatsapp_phone_number.startswith("+"):
            whatsapp_phone_number = f"+91{whatsapp_phone_number}"

        if not supabase_client:
            raise HTTPException(
                status_code=503,
                detail="Database client not available. Check Supabase configuration.",
            )

        # Get location from database
        location_record = await supabase_client.get_customer_location_form(
            whatsapp_phone_number, location_type
        )

        if not location_record:
            raise HTTPException(
                status_code=404,
                detail=f"No {location_type} location found for WhatsApp {whatsapp_phone_number}",
            )

        # Convert to CustomerLocationSubmission model
        location_submission = CustomerLocationSubmission(
            location_id=location_record.get("location_id"),
            whatsapp_phone_number=location_record.get("whatsapp_phone_number"),
            address=location_record.get("address"),
            city=location_record.get("city"),
            state=location_record.get("state"),
            pincode=location_record.get("pincode"),
            landmark=location_record.get("landmark"),
            location_type=location_record.get("location_type"),
            created_at=location_record.get("created_at"),
            metadata=location_record.get("metadata"),
        )

        return CustomerLocationFormResponse(
            success=True,
            location_submission=location_submission,
            message="Location retrieved successfully",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[/agent/customer-location-form] Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")
