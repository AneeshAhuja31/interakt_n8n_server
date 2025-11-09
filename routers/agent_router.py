"""
Agent Router - LangGraph Availability Checking Agent
Replaces the n8n AI Agent cluster (8 nodes) with a single HTTP endpoint
"""

from fastapi import APIRouter, HTTPException
from models.requests import AgentRequest, AgentRequestFromN8N, SaveMessageRequest, OrderConfirmationRequest
from models.responses import (
    AgentResponse,
    WhatsAppTemplatePayload,
    ProductMatch,
    SaveMessageResponse,
    OrderConfirmationResponse,
    OrderSummary,
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


@router.post("/order-confirmation", response_model=OrderConfirmationResponse)
async def order_confirmation(request: OrderConfirmationRequest):
    """
    Generate order confirmation from chat history using LLM

    **Purpose:**
    Analyze recent conversation history to extract order details (product, quantity, price)
    and generate a structured order summary for the WhatsApp template.

    **Workflow:**
    1. Retrieve last 15 messages from chat history
    2. Save user's confirmation message
    3. Use LLM to extract order details from conversation
    4. Generate order summary
    5. Save order to customer_orders table
    6. Return template body values for n8n

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
        from typing import Optional

        # Define structured output model
        class OrderExtraction(BaseModel):
            """Extracted order details from conversation"""
            items: Optional[str] = PydanticField(
                None,
                description="Product name or items being ordered"
            )
            price: Optional[str] = PydanticField(
                None,
                description="Unit price in INR (numbers only, no symbols)"
            )
            quantity: Optional[int] = PydanticField(
                1,
                description="Number of items (default 1 if not mentioned)"
            )
            discount: Optional[str] = PydanticField(
                "No discount",
                description="Discount information (e.g., '15% off' or 'No discount')"
            )
            subtotal: Optional[str] = PydanticField(
                None,
                description="Total price (price × quantity, numbers only)"
            )

        # Build conversation history for LLM
        conversation_history = "\n".join(
            [
                f"{msg.get('role', 'unknown').upper()}: {msg.get('content', '')}"
                for msg in messages
            ]
        )

        # Check if we have any conversation history
        if not conversation_history.strip():
            logger.warning("No conversation history found for order confirmation")
            raise HTTPException(
                status_code=400,
                detail="No conversation history found. Please ask about a product first before confirming an order."
            )

        # Use LLM with structured output to extract order details
        llm = ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            temperature=0,
            google_api_key=settings.GOOGLE_API_KEY,
        )

        # Create structured output LLM
        structured_llm = llm.with_structured_output(OrderExtraction)

        # Create extraction prompt
        extraction_prompt = f"""Analyze the conversation history and extract order details.

Extract the following information:
- items: Product name or items the customer wants to order
- price: Unit price in INR (numbers only, no ₹ symbol)
- quantity: Number of items (default to 1 if not mentioned)
- discount: Discount information if mentioned (e.g., "15% off" or "No discount")
- subtotal: Total price (price × quantity, numbers only)

**Conversation History:**
{conversation_history}

**Current User Message:** {request.message}

Extract the order details from the conversation above."""

        try:
            # Get structured output from LLM
            order_data = await structured_llm.ainvoke(extraction_prompt)
            logger.info(f"Extracted order data: {order_data}")

            # Convert to dict for easier handling
            order_dict = {
                "product_name": order_data.items or "Product",
                "quantity": order_data.quantity or 1,
                "unit_price": order_data.price or "0",
                "discount": order_data.discount or "No discount",
                "total_price": order_data.subtotal or (
                    str(int(order_data.price or "0") * (order_data.quantity or 1))
                    if order_data.price and order_data.price.isdigit()
                    else "0"
                ),
            }

        except Exception as e:
            logger.error(f"LLM extraction failed: {e}", exc_info=True)
            # Fallback: Create default order data
            order_dict = {
                "product_name": "Product",
                "quantity": 1,
                "unit_price": "0",
                "discount": "No discount",
                "total_price": "0",
            }

        # Generate order ID
        from datetime import datetime

        order_id = f"ORD_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Create OrderSummary object
        order_summary = OrderSummary(
            product_name=order_dict.get("product_name", "Product"),
            quantity=int(order_dict.get("quantity", 1)),
            unit_price=str(order_dict.get("unit_price", "0")),
            discount=order_dict.get("discount", "No discount"),
            total_price=str(order_dict.get("total_price", "0")),
            order_id=order_id,
        )

        # Save order to database
        try:
            await supabase_client.create_order(
                session_id=session_id,
                phone_number=phone_number,
                product_name=order_summary.product_name,
                quantity=order_summary.quantity,
                unit_price=order_summary.unit_price,
                total_price=order_summary.total_price,
                discount=order_summary.discount,
                order_id=order_id,
            )
            logger.info(f"Order saved to database: {order_id}")
        except Exception as e:
            logger.warning(f"Failed to save order to database: {e}")

        # Save order summary as AI message
        order_summary_text = f"Order Summary: {order_summary.product_name} x {order_summary.quantity} = ₹{order_summary.total_price} ({order_summary.discount})"
        try:
            await supabase_client.add_message(
                session_id=session_id,
                role="ai",
                content=order_summary_text,
                message_type="order_summary",
                metadata={"order_id": order_id},
            )
        except Exception as e:
            logger.warning(f"Failed to save order summary message: {e}")

        # Prepare template body values for WhatsApp
        template_body_values = [
            order_summary.product_name,
            order_summary.unit_price,
            f"({order_summary.discount})",
            str(order_summary.quantity),
            f"₹{order_summary.total_price}",
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
