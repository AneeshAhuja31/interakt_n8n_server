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

    # ============================================================
    # ORDER HEADERS & ITEMS (NEW MULTI-ITEM ORDERS)
    # ============================================================

    async def create_order_header(
        self,
        order_id: str,
        session_id: str,
        phone_number: str,
        total_price: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create a new order header (for multi-item orders)

        Args:
            order_id: Unique order ID
            session_id: Session identifier
            phone_number: Customer phone number
            total_price: Total order price (sum of all items)
            metadata: Additional metadata

        Returns:
            Created order header record
        """
        order_header_data = {
            "order_id": order_id,
            "session_id": session_id,
            "phone_number": phone_number,
            "total_price": total_price,
            "order_status": "pending",
            "metadata": metadata or {},
        }

        response = self.client.table("order_headers").insert(order_header_data).execute()

        if response.data and len(response.data) > 0:
            return response.data[0]

        raise Exception("Failed to create order header")

    async def create_order_items(
        self,
        order_id: str,
        items: List[Dict[str, Any]],
        phone_number: str,
    ) -> List[Dict[str, Any]]:
        """
        Create multiple order items (batch insert)

        Args:
            order_id: Order ID to associate items with
            items: List of item dictionaries with keys:
                - item_id: Unique item ID
                - product_name: Product name
                - quantity: Number of items (default 1)
                - unit_price: Price per unit
                - discount: Discount information
                - subtotal: Item subtotal (unit_price × quantity)
                - metadata: Optional additional data
            phone_number: Customer phone number (for easy querying by customer)

        Returns:
            List of created order item records
        """
        # Add order_id and phone_number to each item
        items_data = []
        for item in items:
            item_data = {
                "order_id": order_id,
                "item_id": item.get("item_id"),
                "product_name": item.get("product_name"),
                "quantity": item.get("quantity", 1),
                "unit_price": item.get("unit_price"),
                "discount": item.get("discount", "No discount"),
                "subtotal": item.get("subtotal"),
                "phone_number": phone_number,
                "metadata": item.get("metadata", {}),
            }
            items_data.append(item_data)

        response = self.client.table("order_items").insert(items_data).execute()

        if response.data and len(response.data) > 0:
            return response.data

        raise Exception("Failed to create order items")

    async def get_order_with_items(self, order_id: str) -> Optional[Dict[str, Any]]:
        """
        Get order header with all associated items

        Args:
            order_id: Order ID

        Returns:
            Dict with order header and items, or None if not found
        """
        # Get order header
        header_response = (
            self.client.table("order_headers")
            .select("*")
            .eq("order_id", order_id)
            .execute()
        )

        if not header_response.data or len(header_response.data) == 0:
            return None

        order_header = header_response.data[0]

        # Get order items
        items_response = (
            self.client.table("order_items")
            .select("*")
            .eq("order_id", order_id)
            .execute()
        )

        return {
            "header": order_header,
            "items": items_response.data if items_response.data else [],
        }

    async def update_order_header_status(
        self, order_id: str, status: str
    ) -> Dict[str, Any]:
        """
        Update order header status

        Args:
            order_id: Order ID
            status: New status (pending, confirmed, cancelled, completed)

        Returns:
            Updated order header record
        """
        response = (
            self.client.table("order_headers")
            .update({"order_status": status})
            .eq("order_id", order_id)
            .execute()
        )

        if response.data and len(response.data) > 0:
            return response.data[0]

        raise Exception(f"Failed to update order header {order_id}")

    async def get_customer_orders(
        self, phone_number: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get orders for a customer (from old customer_orders table)

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

    async def get_customer_order_headers(
        self, phone_number: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get order headers for a customer (from new order_headers table)

        Args:
            phone_number: Customer phone number
            limit: Number of orders to retrieve

        Returns:
            List of order header records
        """
        response = (
            self.client.table("order_headers")
            .select("*")
            .eq("phone_number", phone_number)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )

        return response.data if response.data else []

    async def get_customer_order_items(
        self, phone_number: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get order items for a customer (from order_items table)
        Useful for quickly retrieving recent orders by phone number

        Args:
            phone_number: Customer phone number
            limit: Number of order items to retrieve

        Returns:
            List of order item records ordered by creation time
        """
        response = (
            self.client.table("order_items")
            .select("*")
            .eq("phone_number", phone_number)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )

        return response.data if response.data else []

    async def get_recent_order_by_id_prefix(
        self, phone_number: str, order_id_prefix: str = "ORD_"
    ) -> Optional[Dict[str, Any]]:
        """
        Get the most recent order for a customer matching an order ID prefix

        Args:
            phone_number: Customer phone number
            order_id_prefix: Order ID prefix to match (default: "ORD_")

        Returns:
            Most recent order header or None if not found
        """
        response = (
            self.client.table("order_headers")
            .select("*")
            .eq("phone_number", phone_number)
            .like("order_id", f"{order_id_prefix}%")
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )

        if response.data and len(response.data) > 0:
            return response.data[0]

        return None

    # ============================================================
    # CUSTOMER FORM SUBMISSIONS (Name, Email, Phone)
    # ============================================================

    async def save_customer_form(
        self,
        whatsapp_phone_number: str,
        entered_name: str,
        entered_email: str,
        entered_phone_number: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Save customer form submission (name, email, phone number)

        The form is filled by a WhatsApp user, and they provide their details.
        Unique ID is based on WhatsApp phone number + timestamp.

        Args:
            whatsapp_phone_number: WhatsApp phone number (from webhook)
            entered_name: Name entered in the form
            entered_email: Email entered in the form
            entered_phone_number: Phone number entered in the form
            metadata: Additional metadata

        Returns:
            Created form submission record with form_id
        """
        from datetime import datetime

        # Generate form ID with WhatsApp phone + timestamp
        now = datetime.now()
        timestamp = now.strftime('%Y%m%d_%H%M%S')
        milliseconds = now.strftime('%f')[:3]
        # Clean phone number for ID (remove + and country code for readability)
        clean_phone = whatsapp_phone_number.lstrip("+").lstrip("91")[-10:]  # Last 10 digits
        form_id = f"FORM_{clean_phone}_{timestamp}_{milliseconds}"

        form_data = {
            "form_id": form_id,
            "whatsapp_phone_number": whatsapp_phone_number,
            "entered_name": entered_name,
            "entered_email": entered_email,
            "entered_phone_number": entered_phone_number,
            "metadata": metadata or {},
        }

        response = self.client.table("customer_form_submissions").insert(form_data).execute()

        if response.data and len(response.data) > 0:
            logger.info(f"Customer form saved: {form_id} for WhatsApp {whatsapp_phone_number}")
            return response.data[0]

        raise Exception("Failed to save customer form")

    async def get_customer_form(
        self, whatsapp_phone_number: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get the most recent form submission for a WhatsApp number

        Args:
            whatsapp_phone_number: WhatsApp phone number

        Returns:
            Most recent form submission record or None if not found
        """
        response = (
            self.client.table("customer_form_submissions")
            .select("*")
            .eq("whatsapp_phone_number", whatsapp_phone_number)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )

        if response.data and len(response.data) > 0:
            return response.data[0]

        return None

    async def get_customer_form_history(
        self, whatsapp_phone_number: str, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get form submission history for a WhatsApp number

        Args:
            whatsapp_phone_number: WhatsApp phone number
            limit: Number of records to retrieve

        Returns:
            List of form submissions ordered by creation time (newest first)
        """
        response = (
            self.client.table("customer_form_submissions")
            .select("*")
            .eq("whatsapp_phone_number", whatsapp_phone_number)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )

        return response.data if response.data else []

    # ============================================================
    # CUSTOMER LOCATION SUBMISSIONS
    # ============================================================

    async def save_customer_location_form(
        self,
        whatsapp_phone_number: str,
        address: str,
        city: Optional[str] = None,
        state: Optional[str] = None,
        pincode: Optional[str] = None,
        landmark: Optional[str] = None,
        location_type: str = "delivery",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Save customer location form submission

        The location form is filled by a WhatsApp user.
        Unique ID is based on WhatsApp phone number + timestamp.

        Args:
            whatsapp_phone_number: WhatsApp phone number (from webhook)
            address: Full address entered in form
            city: City name
            state: State name
            pincode: PIN code
            landmark: Nearby landmark
            location_type: Type of location (delivery, billing, etc.)
            metadata: Additional metadata

        Returns:
            Created location submission record with location_id
        """
        from datetime import datetime

        # Generate location ID with WhatsApp phone + timestamp
        now = datetime.now()
        timestamp = now.strftime('%Y%m%d_%H%M%S')
        milliseconds = now.strftime('%f')[:3]
        # Clean phone number for ID (remove + and country code for readability)
        clean_phone = whatsapp_phone_number.lstrip("+").lstrip("91")[-10:]  # Last 10 digits
        location_id = f"LOCATION_{clean_phone}_{timestamp}_{milliseconds}"

        location_data = {
            "location_id": location_id,
            "whatsapp_phone_number": whatsapp_phone_number,
            "address": address,
            "city": city,
            "state": state,
            "pincode": pincode,
            "landmark": landmark,
            "location_type": location_type,
            "metadata": metadata or {},
        }

        response = self.client.table("customer_location_submissions").insert(location_data).execute()

        if response.data and len(response.data) > 0:
            logger.info(f"Customer location saved: {location_id} for WhatsApp {whatsapp_phone_number}")
            return response.data[0]

        raise Exception("Failed to save customer location")

    async def get_customer_location_form(
        self, whatsapp_phone_number: str, location_type: str = "delivery"
    ) -> Optional[Dict[str, Any]]:
        """
        Get the most recent location submission for a WhatsApp number

        Args:
            whatsapp_phone_number: WhatsApp phone number
            location_type: Type of location (delivery, billing, etc.)

        Returns:
            Most recent location submission record or None if not found
        """
        response = (
            self.client.table("customer_location_submissions")
            .select("*")
            .eq("whatsapp_phone_number", whatsapp_phone_number)
            .eq("location_type", location_type)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )

        if response.data and len(response.data) > 0:
            return response.data[0]

        return None

    async def get_customer_location_form_history(
        self, whatsapp_phone_number: str, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get location submission history for a WhatsApp number

        Args:
            whatsapp_phone_number: WhatsApp phone number
            limit: Number of records to retrieve

        Returns:
            List of location submissions ordered by creation time (newest first)
        """
        response = (
            self.client.table("customer_location_submissions")
            .select("*")
            .eq("whatsapp_phone_number", whatsapp_phone_number)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )

        return response.data if response.data else []


@lru_cache()
def get_supabase_client() -> SupabaseClient:
    """Get cached Supabase client instance"""
    return SupabaseClient(url=settings.SUPABASE_URL, key=settings.SUPABASE_KEY)
