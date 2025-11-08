"""
LangGraph Availability Checking Agent
Replaces the n8n AI Agent cluster with a stateful LangGraph workflow
"""

from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from agents.state import AvailabilityAgentState
from agents.tools.qdrant_tool import search_products
from models.responses import AvailabilityStatus
from config import settings
import time
import logging

logger = logging.getLogger(__name__)


# ============================================================
# NODE DEFINITIONS
# ============================================================


def query_qdrant_node(state: AvailabilityAgentState) -> Dict[str, Any]:
    """
    Node 1: Query Qdrant vector store for product matches

    Args:
        state: Current agent state

    Returns:
        Updated state with retrieved_products and top_product
    """
    logger.info(f"[query_qdrant_node] Searching for: {state['customer_message']}")

    start_time = time.time()

    try:
        # Search products using Qdrant
        products = search_products(
            query=state["customer_message"],
            top_k=settings.QDRANT_TOP_K,
            score_threshold=0.5,  # Only return products with >50% similarity
        )

        search_time_ms = int((time.time() - start_time) * 1000)

        if not products:
            logger.warning("No products found in Qdrant search")
            # Return empty results but continue workflow
            return {
                "retrieved_products": [],
                "top_product": {},
                "qdrant_search_time_ms": search_time_ms,
                "current_node": "query_qdrant",
            }

        # Get top product (highest similarity score)
        top_product = products[0]

        logger.info(
            f"[query_qdrant_node] Found {len(products)} products. Top match: {top_product['name']} (score: {top_product['similarity_score']:.2f})"
        )

        return {
            "retrieved_products": products,
            "top_product": top_product,
            "qdrant_search_time_ms": search_time_ms,
            "current_node": "query_qdrant",
        }

    except Exception as e:
        logger.error(f"[query_qdrant_node] Error: {str(e)}")
        return {
            "error": f"Qdrant search failed: {str(e)}",
            "retrieved_products": [],
            "top_product": {},
            "current_node": "query_qdrant",
        }


def llm_decision_node(state: AvailabilityAgentState) -> Dict[str, Any]:
    """
    Node 2: LLM makes availability decision and formats output

    Args:
        state: Current agent state

    Returns:
        Updated state with availability_status and all output fields
    """
    logger.info("[llm_decision_node] Making availability decision")

    # Check if we have products from Qdrant
    if not state.get("top_product"):
        # No products found - return out_of_stock
        logger.warning("[llm_decision_node] No products found, returning out_of_stock")
        return {
            "availability_status": "out_of_stock",
            "matched_product": "Product not found",
            "brand": "N/A",
            "price": "N/A",
            "discount": "No discount",
            "specs": "Product not available in catalog",
            "product_url": "",
            "image_url": "",
            "hinglish_output": f"Sorry, '{state['customer_message']}' abhi available nahi hai. Koi aur product dekhna chahenge?",
            "reasoning": "No matching products found in vector search",
            "current_node": "llm_decision",
        }

    try:
        # Initialize LLM
        llm = ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            temperature=settings.GEMINI_TEMPERATURE,
            google_api_key=settings.GOOGLE_API_KEY,
        )

        # Use with_structured_output instead of PydanticOutputParser
        structured_llm = llm.with_structured_output(AvailabilityStatus)

        # Create prompt using ChatPromptTemplate
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert product availability agent for "Star Trade" - a pneumatic tools and hardware company in India.

**Your Task:**
1. Determine availability status:
   - "in_stock": The matched product is a good match (similarity > 0.75) and available
   - "out_of_stock": No good match found (similarity < 0.60)
   - "alternate_available": Match is moderate (0.60-0.75) or customer might want alternatives

2. Generate a friendly Hinglish (Hindi + English mix) response message for WhatsApp
   - Use conversational tone
   - Include product name, price, and link
   - Examples:
     * "Haan, {{product_name}} available hai! Price: {{price}} with {{discount}}. Check details: {{url}}"
     * "Sorry, exact product abhi nahi hai, but similar product available: {{name}} at {{price}}. Link: {{url}}"

**Important:**
- Return structured data with all required fields
- Always include all fields
- Make output message natural and friendly
- Use Hinglish (Hindi-English mix) that's easy to read on WhatsApp"""),
            ("user", """**Customer Query:** {customer_message}

**Top Matched Product from Database:**
- Name: {product_name}
- Brand: {product_brand}
- Price: {product_price}
- Discount: {product_discount}
- Specs: {product_specs}
- Product URL: {product_url}
- Image URL: {product_image}
- Similarity Score: {similarity_score}

**Other Similar Products:** {other_products}""")
        ])

        # Prepare other products summary
        other_products_summary = ""
        if len(state.get("retrieved_products", [])) > 1:
            other_products_summary = "\n".join(
                [
                    f"- {p['name']} ({p['brand']}) - {p['price']} (score: {p['similarity_score']:.2f})"
                    for p in state["retrieved_products"][1:3]  # Show top 2 alternatives
                ]
            )

        # Execute LLM decision using direct invoke (no LCEL chain)
        top_product = state["top_product"]
        result = structured_llm.invoke(
            prompt.format_messages(
                customer_message=state["customer_message"],
                product_name=top_product.get("name", "Unknown"),
                product_brand=top_product.get("brand", "Unknown"),
                product_price=top_product.get("price", "Price not available"),
                product_discount=top_product.get("discount", "No discount"),
                product_specs=top_product.get("specs", ""),
                product_url=top_product.get("product_url", ""),
                product_image=top_product.get("image_url", ""),
                similarity_score=top_product.get("similarity_score", 0),
                other_products=other_products_summary or "None",
            )
        )

        logger.info(
            f"[llm_decision_node] Decision: {result.status}, Product: {result.matched_product}"
        )

        return {
            "availability_status": result.status,
            "matched_product": result.matched_product,
            "brand": result.brand,
            "price": result.price,
            "discount": result.discount,
            "specs": result.specs,
            "product_url": result.product_url,
            "image_url": result.image_url,
            "hinglish_output": result.output,
            "reasoning": f"LLM decision based on similarity score: {top_product.get('similarity_score', 0):.2f}",
            "current_node": "llm_decision",
            "llm_calls": state.get("llm_calls", 0) + 1,
        }

    except Exception as e:
        logger.error(f"[llm_decision_node] Error: {str(e)}")

        # Fallback to rule-based decision
        top_product = state["top_product"]
        similarity_score = top_product.get("similarity_score", 0)

        if similarity_score >= 0.75:
            status = "in_stock"
            message = f"Haan, {top_product['name']} available hai! Price: {top_product['price']}. Check: {top_product['product_url']}"
        elif similarity_score >= 0.60:
            status = "alternate_available"
            message = f"Exact match nahi mila, but similar product available: {top_product['name']} at {top_product['price']}. Link: {top_product['product_url']}"
        else:
            status = "out_of_stock"
            message = f"Sorry, '{state['customer_message']}' abhi available nahi hai."

        return {
            "availability_status": status,
            "matched_product": top_product.get("name", "Unknown"),
            "brand": top_product.get("brand", "Unknown"),
            "price": top_product.get("price", "N/A"),
            "discount": top_product.get("discount", "No discount"),
            "specs": top_product.get("specs", ""),
            "product_url": top_product.get("product_url", ""),
            "image_url": top_product.get("image_url", ""),
            "hinglish_output": message,
            "reasoning": f"Fallback rule-based decision (LLM error). Error: {str(e)}",
            "error": f"LLM failed, used fallback: {str(e)}",
            "current_node": "llm_decision",
        }


# ============================================================
# GRAPH CONSTRUCTION
# ============================================================


def create_availability_agent(postgres_uri: Optional[str] = None):
    """
    Create the LangGraph availability checking agent

    Args:
        postgres_uri: PostgreSQL connection string for checkpointing
                     If None, uses settings.POSTGRES_URI

    Returns:
        Compiled LangGraph workflow
    """
    logger.info("Creating availability agent workflow...")

    # Build the state graph
    workflow = StateGraph(AvailabilityAgentState)

    # Add nodes
    workflow.add_node("query_qdrant", query_qdrant_node)
    workflow.add_node("llm_decision", llm_decision_node)

    # Define edges (workflow flow)
    workflow.set_entry_point("query_qdrant")
    workflow.add_edge("query_qdrant", "llm_decision")
    workflow.add_edge("llm_decision", END)

    # Setup checkpointing if Postgres URI provided
    if postgres_uri or settings.POSTGRES_URI:
        try:
            checkpointer = PostgresSaver.from_conn_string(
                postgres_uri or settings.POSTGRES_URI
            )
            compiled_graph = workflow.compile(checkpointer=checkpointer)
            compiled_graph.get_graph().draw_mermaid_png("mermaid.png")
        except Exception as e:
            
            compiled_graph = workflow.compile()
    else:
        compiled_graph = workflow.compile()

    return compiled_graph


# ============================================================
# GLOBAL AGENT INSTANCE (singleton)
# ============================================================

_agent_instance = None


def get_availability_agent():
    """Get or create the global availability agent instance"""
    global _agent_instance

    if _agent_instance is None:
        _agent_instance = create_availability_agent()

    return _agent_instance
