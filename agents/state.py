"""
LangGraph Agent State Definitions
Defines the state structure for the availability checking agent
"""

from typing import TypedDict, Annotated, Sequence, Literal, List, Dict, Any, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AvailabilityAgentState(TypedDict):
    """
    State for the LangGraph availability checking agent

    This state is passed between nodes in the LangGraph workflow
    and manages the complete conversation and decision-making process
    """

    # ============================================================
    # CORE INPUTS (from request)
    # ============================================================

    customer_message: str
    """The customer's product query (e.g., 'PNEUMATIC STAPLER')"""

    phone_number: str
    """Customer WhatsApp phone number with country code"""

    session_id: str
    """Session identifier for conversation continuity"""

    # ============================================================
    # CONVERSATION HISTORY (managed by LangGraph)
    # ============================================================

    messages: Annotated[Sequence[BaseMessage], add_messages]
    """
    Conversation message history
    The add_messages annotation enables automatic message appending
    """

    # ============================================================
    # VECTOR SEARCH RESULTS
    # ============================================================

    query_embedding: Optional[List[float]]
    """Embedding vector of the customer query"""

    retrieved_products: List[Dict[str, Any]]
    """
    Top K products retrieved from Qdrant vector search
    Each product contains: name, brand, price, specs, url, etc.
    """

    top_product: Dict[str, Any]
    """
    The best matched product (highest similarity score)
    Used as primary candidate for availability decision
    """

    qdrant_search_time_ms: Optional[int]
    """Time taken for Qdrant search in milliseconds"""

    # ============================================================
    # LLM DECISION
    # ============================================================

    availability_status: Literal["in_stock", "out_of_stock", "alternate_available"]
    """
    Final availability decision made by the LLM
    - in_stock: Product is available
    - out_of_stock: Product not available, no alternatives
    - alternate_available: Product not available, but similar products found
    """

    reasoning: Optional[str]
    """LLM's reasoning for the availability decision (for debugging)"""

    # ============================================================
    # STRUCTURED OUTPUT FIELDS (for WhatsApp template)
    # ============================================================

    matched_product: str
    """Final product name to show customer"""

    brand: str
    """Product brand"""

    price: str
    """Price in INR (formatted, e.g., '₹4,500')"""

    discount: str
    """Discount information (e.g., '15% off' or 'No discount')"""

    specs: str
    """Short product specifications"""

    product_url: str
    """Product page URL"""

    image_url: str
    """Product image URL"""

    hinglish_output: str
    """
    Hinglish summary message for WhatsApp
    Example: "Haan, Pneumatic Stapler available hai! ₹4,500 mein..."
    """

    # ============================================================
    # METADATA & TRACKING
    # ============================================================

    processing_start: Optional[float]
    """Timestamp when processing started (for performance tracking)"""

    current_node: Optional[str]
    """Current node in the LangGraph workflow (for debugging)"""

    error: Optional[str]
    """Error message if any step fails"""

    llm_calls: Optional[int]
    """Number of LLM API calls made"""

    total_tokens: Optional[int]
    """Total tokens used across all LLM calls"""


# Type alias for optional state fields (used during initialization)
AvailabilityAgentInput = Dict[str, Any]
