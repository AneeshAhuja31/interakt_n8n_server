"""
Agent Router - LangGraph Availability Checking Agent
Replaces the n8n AI Agent cluster (8 nodes) with a single HTTP endpoint
"""

from fastapi import APIRouter, HTTPException
from models.requests import AgentRequest, AgentRequestFromN8N
from models.responses import AgentResponse, WhatsAppTemplatePayload, ProductMatch
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
