"""
Qdrant Vector Store Tool for Product Search
Provides semantic search capabilities for the availability agent
"""

from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from config import settings
import logging
import time
import re

logger = logging.getLogger(__name__)


class QdrantProductSearch:
    """
    Wrapper around Qdrant vector store for product search
    """

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        """
        Initialize Qdrant client and embeddings

        Args:
            url: Qdrant server URL (defaults to settings.QDRANT_URL)
            api_key: Qdrant API key (defaults to settings.QDRANT_API_KEY)
            collection_name: Collection name (defaults to settings.QDRANT_COLLECTION)
        """
        self.url = url or settings.QDRANT_URL
        self.api_key = api_key or settings.QDRANT_API_KEY
        self.collection_name = collection_name or settings.QDRANT_COLLECTION

        # Initialize Qdrant client
        self.client = QdrantClient(url=self.url, api_key=self.api_key)

        # Initialize OpenAI embeddings (1536 dimensions - matches existing Qdrant collection)
        self.embeddings = OpenAIEmbeddings(
            model=settings.OPENAI_EMBEDDING_MODEL,
            openai_api_key=settings.OPENAI_API_KEY,
        )

        # Create Qdrant vector store instance
        self.vectorstore = Qdrant(
            client=self.client,
            collection_name=self.collection_name,
            embeddings=self.embeddings,
        )

        logger.info(
            f"Qdrant tool initialized: collection='{self.collection_name}', url='{self.url}'"
        )

    def search_products(
        self, query: str, top_k: int = 5, score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for products using semantic similarity

        Args:
            query: Customer's product query (e.g., "PNEUMATIC STAPLER")
            top_k: Number of top results to return (default: 5)
            score_threshold: Minimum similarity score (0-1) to include result

        Returns:
            List of product dictionaries with metadata and similarity scores
        """
        start_time = time.time()

        try:
            logger.debug(f"[Qdrant Search] Starting search...")
            logger.debug(f"[Qdrant Search] Query: '{query}'")
            logger.debug(f"[Qdrant Search] Top K: {top_k}, Score threshold: {score_threshold}")

            # Generate query embedding
            logger.debug(f"[Qdrant Search] Generating query embedding...")
            query_vector = self.embeddings.embed_query(query)
            logger.debug(f"[Qdrant Search] Query vector dimensions: {len(query_vector)}")

            # Perform direct Qdrant client search (bypasses LangChain Document validation)
            logger.debug(f"[Qdrant Search] Calling direct Qdrant client search...")
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k
            )
            logger.debug(f"[Qdrant Search] Raw results count: {len(results)}")

            products = []
            for idx, result in enumerate(results):
                logger.debug(f"[Qdrant Search] Processing result {idx + 1}/{len(results)}")
                logger.debug(f"[Qdrant Search] Score: {result.score}")
                logger.debug(f"[Qdrant Search] Payload keys: {list(result.payload.keys())}")

                # Skip if below threshold
                if score_threshold and result.score < score_threshold:
                    logger.debug(f"[Qdrant Search] Skipping result {idx + 1} - score {result.score} below threshold {score_threshold}")
                    continue

                # Extract metadata from payload (matching debron_ingestion.py schema)
                payload = result.payload

                # Parse price and discount to calculate final price
                price_str = payload.get("price_inr", "Price not available")
                discount_str = payload.get("discount_info", "No discount")

                # Initialize price fields
                original_price = 0
                discount_percent = 0
                discount_amount = 0
                final_price = 0

                # Parse original price (remove ₹, INR, commas, etc.)
                price_match = re.search(r'[\d,]+', price_str)
                if price_match:
                    original_price = int(price_match.group().replace(',', ''))

                # Parse discount percentage from discount text (e.g., "25% off" -> 25)
                discount_match = re.search(r'(\d+)\s*%', discount_str)
                if discount_match:
                    discount_percent = int(discount_match.group(1))

                # Calculate discount amount and final price
                if original_price > 0 and discount_percent > 0:
                    discount_amount = int(original_price * discount_percent / 100)
                    final_price = original_price - discount_amount
                else:
                    final_price = original_price

                product = {
                    "name": payload.get("product_name", "Unknown Product"),
                    "brand": payload.get("brand", "Unknown Brand"),
                    "price": price_str,  # Original price string as-is
                    "discount": discount_str,  # Discount string as-is
                    "original_price": original_price,  # Parsed original price (int)
                    "discount_percent": discount_percent,  # Discount percentage (int)
                    "discount_amount": discount_amount,  # Calculated discount amount (int)
                    "final_price": final_price,  # Price after discount (int)
                    "specs": payload.get("specs", ""),
                    "model": payload.get("model", ""),
                    "category": payload.get("category", ""),
                    "keywords": payload.get("keywords", ""),
                    "usage_description": payload.get("usage_description", ""),
                    "product_url": "",  # Not in ingestion schema, kept for compatibility
                    "image_url": payload.get("cloudinary_url", ""),
                    "similarity_score": float(result.score),
                    "content": payload.get("content", ""),  # Combined text used for embedding
                }

                logger.debug(f"[Qdrant Search] Added product: {product['name']} (score: {result.score:.4f})")
                products.append(product)

            search_time_ms = int((time.time() - start_time) * 1000)

            logger.info(
                f"[Qdrant Search] COMPLETED: query='{query}', valid_results={len(products)}/{len(results)}, time={search_time_ms}ms"
            )

            if products:
                logger.debug(f"[Qdrant Search] Top result: {products[0]['name']} (score: {products[0]['similarity_score']:.4f})")
            else:
                logger.warning(f"[Qdrant Search] No valid products found!")

            return products

        except Exception as e:
            logger.error(f"[Qdrant Search] FAILED: {str(e)}", exc_info=True)
            raise

    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding vector for a text query

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        try:
            embedding = self.embeddings.embed_query(text)
            return embedding
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            raise


# ============================================================
# GLOBAL INSTANCE (singleton pattern)
# ============================================================

_qdrant_instance: Optional[QdrantProductSearch] = None


def get_qdrant_tool() -> QdrantProductSearch:
    """Get or create the global Qdrant tool instance"""
    global _qdrant_instance

    if _qdrant_instance is None:
        _qdrant_instance = QdrantProductSearch()

    return _qdrant_instance


# ============================================================
# CONVENIENCE FUNCTION (for use in LangGraph nodes)
# ============================================================


def search_products(
    query: str, top_k: int = 5, score_threshold: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Convenience function to search products
    Uses the global Qdrant tool instance

    Args:
        query: Customer's product query
        top_k: Number of results to return
        score_threshold: Minimum similarity score

    Returns:
        List of product dictionaries with metadata and scores

    Example:
        >>> products = search_products("PNEUMATIC STAPLER", top_k=5)
        >>> print(products[0]['name'])
        'Pneumatic Stapler F30'
    """
    tool = get_qdrant_tool()
    return tool.search_products(query, top_k, score_threshold)
