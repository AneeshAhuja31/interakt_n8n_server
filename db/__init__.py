"""Database module for WhatsApp AI Service"""

from .supabase_client import get_supabase_client, SupabaseClient

__all__ = ["get_supabase_client", "SupabaseClient"]
