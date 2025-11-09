"""
Supabase client for database operations
Handles connections to Postgres/Supabase for agent state persistence
"""

from supabase import create_client, Client
from functools import lru_cache
from typing import Optional, Dict, Any, List
from datetime import datetime
from config import settings
import logging

logger = logging.getLogger(__name__)


class SupabaseClient:
    """Wrapper around Supabase client with helper methods"""

    def __init__(self, url: str, key: str):
        self.client: Client = create_client(url, key)
        logger.info("Supabase client initialized")

    # ============================================================
    # SESSION MANAGEMENT
    # ============================================================

    async def get_or_create_session(self, phone_number: str) -> Dict[str, Any]:
        """
        Get existing session or create new one for a phone number

        Args:
            phone_number: WhatsApp phone number (e.g., "+919643524080")

        Returns:
            Session record as dictionary
        """
        session_id = f"whatsapp_{phone_number}"

        # Check if session exists
        response = (
            self.client.table("agent_sessions")
            .select("*")
            .eq("session_id", session_id)
            .execute()
        )

        if response.data and len(response.data) > 0:
            # Update last_interaction_at
            session = response.data[0]
            self.client.table("agent_sessions").update(
                {
                    "last_interaction_at": datetime.now().isoformat(),
                    "is_active": True,
                }
            ).eq("session_id", session_id).execute()

            return session

        # Create new session
        new_session = {
            "session_id": session_id,
            "phone_number": phone_number,
            "country_code": phone_number[:3] if phone_number.startswith("+") else "+91",
            "total_messages": 0,
            "is_active": True,
        }

        response = self.client.table("agent_sessions").insert(new_session).execute()

        if response.data and len(response.data) > 0:
            return response.data[0]

        raise Exception(f"Failed to create session for {phone_number}")

    # ============================================================
    # MESSAGE HISTORY
    # ============================================================

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        message_type: str = "text",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Add a message to the conversation history

        Args:
            session_id: Session identifier
            role: 'human' or 'ai' or 'system'
            content: Message content
            message_type: Type of message (text, image, audio)
            metadata: Additional metadata

        Returns:
            Created message record
        """
        message_data = {
            "session_id": session_id,
            "role": role,
            "content": content,
            "message_type": message_type,
            "metadata": metadata or {},
        }

        response = self.client.table("agent_messages").insert(message_data).execute()

        # Update session message count
        self.client.rpc(
            "increment_session_messages", {"p_session_id": session_id}
        ).execute()

        if response.data and len(response.data) > 0:
            return response.data[0]

        raise Exception("Failed to add message")

    async def get_recent_messages(
        self, session_id: str, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get recent messages for a session (for memory window)

        Args:
            session_id: Session identifier
            limit: Number of messages to retrieve (default: 20)

        Returns:
            List of message records ordered by timestamp DESC
        """
        response = (
            self.client.table("agent_messages")
            .select("role, content, timestamp, metadata")
            .eq("session_id", session_id)
            .order("timestamp", desc=True)
            .limit(limit)
            .execute()
        )

        return response.data if response.data else []

    async def trim_message_history(self, session_id: str, keep_count: int = 20) -> int:
        """
        Trim message history to keep only the most recent N messages

        Args:
            session_id: Session identifier
            keep_count: Number of messages to keep

        Returns:
            Number of messages deleted
        """
        # Use the SQL function defined in schemas.sql
        response = self.client.rpc(
            "trim_message_history",
            {"p_session_id": session_id, "p_keep_count": keep_count},
        ).execute()

        return response.data if response.data else 0

    # ============================================================
    # CHECKPOINTING (for LangGraph)
    # ============================================================

    async def save_checkpoint(
        self, session_id: str, checkpoint_id: str, state_data: Dict[str, Any], node_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Save agent state checkpoint

        Args:
            session_id: Session identifier
            checkpoint_id: Unique checkpoint identifier
            state_data: Full agent state as dictionary
            node_name: Current node in LangGraph

        Returns:
            Created checkpoint record
        """
        checkpoint_data = {
            "session_id": session_id,
            "checkpoint_id": checkpoint_id,
            "state_data": state_data,
            "node_name": node_name,
        }

        response = (
            self.client.table("agent_state_checkpoints")
            .insert(checkpoint_data)
            .execute()
        )

        if response.data and len(response.data) > 0:
            return response.data[0]

        raise Exception(f"Failed to save checkpoint {checkpoint_id}")

    async def get_latest_checkpoint(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the most recent checkpoint for a session

        Args:
            session_id: Session identifier

        Returns:
            Checkpoint record or None if not found
        """
        response = (
            self.client.table("agent_state_checkpoints")
            .select("*")
            .eq("session_id", session_id)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )

        if response.data and len(response.data) > 0:
            return response.data[0]

        return None

    # ============================================================
    # QUERY LOGGING (for analytics)
    # ============================================================

    async def log_product_query(
        self,
        session_id: str,
        phone_number: str,
        query_text: str,
        matched_product: Optional[str] = None,
        brand: Optional[str] = None,
        availability_status: Optional[str] = None,
        retrieved_products: Optional[List[Dict[str, Any]]] = None,
        response_time_ms: Optional[int] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Log a product availability query for analytics

        Args:
            session_id: Session identifier
            phone_number: Customer phone number
            query_text: Original query text
            matched_product: Best matched product name
            brand: Product brand
            availability_status: in_stock/out_of_stock/alternate_available
            retrieved_products: Top K products from vector search
            response_time_ms: Response time in milliseconds
            success: Whether query was successful
            error_message: Error message if failed
            metadata: Additional metadata

        Returns:
            Created query log record
        """
        log_data = {
            "session_id": session_id,
            "phone_number": phone_number,
            "query_text": query_text,
            "matched_product": matched_product,
            "brand": brand,
            "availability_status": availability_status,
            "retrieved_products": retrieved_products or [],
            "response_time_ms": response_time_ms,
            "success": success,
            "error_message": error_message,
            "metadata": metadata or {},
        }

        response = self.client.table("product_query_logs").insert(log_data).execute()

        if response.data and len(response.data) > 0:
            return response.data[0]

        logger.warning("Failed to log product query")
        return {}

    # ============================================================
    # ANALYTICS
    # ============================================================

    async def get_session_stats(self, phone_number: str) -> Optional[Dict[str, Any]]:
        """
        Get statistics for a customer session

        Args:
            phone_number: Customer phone number

        Returns:
            Session statistics from the session_stats view
        """
        response = (
            self.client.table("session_stats")
            .select("*")
            .eq("phone_number", phone_number)
            .execute()
        )

        if response.data and len(response.data) > 0:
            return response.data[0]

        return None

    async def get_daily_metrics(self, limit: int = 30) -> List[Dict[str, Any]]:
        """
        Get daily query metrics

        Args:
            limit: Number of days to retrieve

        Returns:
            List of daily metrics from daily_query_metrics view
        """
        response = (
            self.client.table("daily_query_metrics")
            .select("*")
            .limit(limit)
            .execute()
        )

        return response.data if response.data else []

    # ============================================================
    # ORDER MANAGEMENT
    # ============================================================

    async def create_order(
        self,
        session_id: str,
        phone_number: str,
        product_name: str,
        quantity: int,
        unit_price: str,
        total_price: str,
        discount: str = "No discount",
        order_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create a new customer order

        Args:
            session_id: Session identifier
            phone_number: Customer phone number
            product_name: Name of the product ordered
            quantity: Number of items
            unit_price: Price per unit (as string)
            total_price: Total order price (as string)
            discount: Discount information
            order_id: Optional custom order ID
            metadata: Additional metadata

        Returns:
            Created order record
        """
        order_data = {
            "session_id": session_id,
            "phone_number": phone_number,
            "product_name": product_name,
            "quantity": quantity,
            "unit_price": unit_price,
            "total_price": total_price,
            "discount": discount,
            "order_status": "pending",
            "metadata": metadata or {},
        }

        if order_id:
            order_data["order_id"] = order_id

        response = self.client.table("customer_orders").insert(order_data).execute()

        if response.data and len(response.data) > 0:
            return response.data[0]

        raise Exception("Failed to create order")

    async def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """
        Get order by ID

        Args:
            order_id: Order ID

        Returns:
            Order record or None if not found
        """
        response = (
            self.client.table("customer_orders")
            .select("*")
            .eq("order_id", order_id)
            .execute()
        )

        if response.data and len(response.data) > 0:
            return response.data[0]

        return None

    async def update_order_status(
        self, order_id: str, status: str
    ) -> Dict[str, Any]:
        """
        Update order status

        Args:
            order_id: Order ID
            status: New status (pending, confirmed, cancelled)

        Returns:
            Updated order record
        """
        response = (
            self.client.table("customer_orders")
            .update({"order_status": status})
            .eq("order_id", order_id)
            .execute()
        )

        if response.data and len(response.data) > 0:
            return response.data[0]

        raise Exception(f"Failed to update order {order_id}")

    async def get_customer_orders(
        self, phone_number: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get orders for a customer

        Args:
            phone_number: Customer phone number
            limit: Number of orders to retrieve

        Returns:
            List of order records
        """
        response = (
            self.client.table("customer_orders")
            .select("*")
            .eq("phone_number", phone_number)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )

        return response.data if response.data else []


@lru_cache()
def get_supabase_client() -> SupabaseClient:
    """Get cached Supabase client instance"""
    return SupabaseClient(url=settings.SUPABASE_URL, key=settings.SUPABASE_KEY)
