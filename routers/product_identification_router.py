"""
Product Identification Router - Image-based product identification using CLIP + GPT-4o
Handles image URL input for n8n workflows
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from transformers import CLIPProcessor, CLIPModel
from qdrant_client import QdrantClient
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from PIL import Image
from io import BytesIO
import torch
import base64
import requests
import logging
from config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/product-identification", tags=["Product Identification"])


# ============= PYDANTIC MODELS =============

class ImageURLRequest(BaseModel):
    """Request model for image URL input (n8n compatible)"""
    image_url: str = Field(description="URL of the image to identify")

    class Config:
        json_schema_extra = {
            "example": {
                "image_url": "https://res.cloudinary.com/your-cloud/image/upload/v123/sample.jpg"
            }
        }


class ProductMatch(BaseModel):
    """Individual product match from retrieval"""
    product_name: str = Field(description="Name of the product")
    model: str = Field(description="Model number or identifier")
    category: str = Field(description="Product category")
    brand: str = Field(description="Brand name")
    similarity_score: float = Field(description="Vector similarity score from Qdrant (0-1)")
    confidence: float = Field(description="LLM confidence that this is the correct match (0-1)")
    reasoning: str = Field(description="Explanation of why this product matches or doesn't match")
    specs: Optional[str] = Field(default=None, description="Product specifications")
    price: Optional[str] = Field(default=None, description="Price in INR")


class ProductIdentification(BaseModel):
    """Final structured output from RAG chain"""
    identified_product: ProductMatch = Field(description="The most likely product match")
    overall_confidence: float = Field(description="Overall confidence in the identification (0-1)")
    reasoning_chain: List[str] = Field(description="Step-by-step reasoning process")
    alternative_matches: List[ProductMatch] = Field(description="Other possible matches ranked by confidence")
    visual_analysis: str = Field(description="Analysis of the uploaded image's visual features")
    recommendation: str = Field(description="Final recommendation or conclusion about the product identification")


# ============= INITIALIZE MODELS (LAZY LOADING) =============

_clip_model = None
_clip_processor = None
_qdrant_client = None
_llm = None
_collection_name = "products_images"


def get_clip_model():
    """Lazy load CLIP model"""
    global _clip_model
    if _clip_model is None:
        logger.info("Loading CLIP model...")
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    return _clip_model


def get_clip_processor():
    """Lazy load CLIP processor"""
    global _clip_processor
    if _clip_processor is None:
        logger.info("Loading CLIP processor...")
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return _clip_processor


def get_qdrant_client():
    """Lazy load Qdrant client"""
    global _qdrant_client
    if _qdrant_client is None:
        logger.info("Initializing Qdrant client...")
        _qdrant_client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
    return _qdrant_client


def get_llm():
    """Lazy load OpenAI LLM"""
    global _llm
    if _llm is None:
        logger.info("Initializing OpenAI LLM...")
        _llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=settings.OPENAI_API_KEY)
    return _llm


# ============= CORE RAG FUNCTIONS =============

def embed_image(image: Image.Image) -> list:
    """
    Generate CLIP embeddings for an image

    Args:
        image: PIL Image object

    Returns:
        List of floats representing the 512-dim embedding vector
    """
    model = get_clip_model()
    processor = get_clip_processor()
    
    image = image.convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        embedding = model.get_image_features(**inputs)
    embedding = embedding / embedding.norm(dim=-1, keepdim=True)  # normalize
    return embedding[0].tolist()


def retrieve_similar_products(image_embedding: list, top_k: int = 5) -> list:
    """
    Retrieve top-k similar products from Qdrant

    Args:
        image_embedding: CLIP embedding vector
        top_k: Number of results to return

    Returns:
        List of search results with payload and score
    """
    client = get_qdrant_client()
    search_results = client.search(
        collection_name=_collection_name,
        query_vector=image_embedding,
        limit=top_k,
        with_payload=True
    )
    return search_results


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string for GPT-4o vision"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


def load_image_from_url(url: str) -> Image.Image:
    """
    Download and load image from URL (for n8n workflows)

    Args:
        url: Image URL (Cloudinary, HTTP, etc.)

    Returns:
        PIL Image object
    """
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content))
    return image


def run_rag_chain(uploaded_image: Image.Image, retrieved_products: list) -> ProductIdentification:
    """
    Multi-step RAG chain using GPT-4o to identify product from image and retrieved metadata

    Args:
        uploaded_image: The user's uploaded image
        retrieved_products: List of similar products from Qdrant

    Returns:
        ProductIdentification with structured output
    """
    llm = get_llm()

    # Prepare retrieved products context
    products_context = []
    for idx, result in enumerate(retrieved_products):
        payload = result.payload
        products_context.append({
            "rank": idx + 1,
            "similarity_score": result.score,
            "product_name": payload.get("Product Name", ""),
            "model": payload.get("Model", ""),
            "category": payload.get("Category", ""),
            "brand": payload.get("Brand", ""),
            "specs": payload.get("Specs", ""),
            "price": payload.get("Price", ""),
            "keywords": payload.get("Keywords", ""),
            "usage_description": payload.get("Usage Description", ""),
            "stock_status": payload.get("Stock Status", "")
        })

    # Convert image to base64 for GPT-4o vision
    image_base64 = image_to_base64(uploaded_image)

    # Multi-step prompt for dynamic reasoning
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert product identification AI that analyzes images and product metadata to accurately identify industrial tools and equipment.

Your task is to perform a multi-step analysis:

STEP 1: VISUAL ANALYSIS
Carefully examine the uploaded image and describe:
- What type of product/tool is visible
- Key visual features (shape, color, components, design)
- Any visible text, labels, model numbers, or brand markings
- Estimated category (e.g., power tool, hand tool, equipment, etc.)

STEP 2: RETRIEVAL ANALYSIS
Review the top 5 similar products retrieved from the database and their similarity scores.
For each product, analyze:
- How well the specs and category match the visual features
- Whether the brand and model make sense given what you see
- Correlation between vector similarity score and actual visual match

STEP 3: CONFIDENCE SCORING
Assign a confidence score (0-1) to each retrieved product based on:
- Visual similarity to uploaded image
- Spec alignment with visible features
- Vector similarity score
- Brand/model consistency

STEP 4: FINAL IDENTIFICATION
Identify the most likely product and provide:
- Clear reasoning chain showing your analysis
- Overall confidence level
- Alternative possibilities if applicable
- Final recommendation

Be thorough, analytical, and transparent in your reasoning. If no clear match exists, state that clearly."""),
        ("user", [
            {"type": "text", "text": "Here is the uploaded image to identify:"},
            {"type": "image_url", "image_url": {"url": "{image_url}"}},
            {"type": "text", "text": "\n\nHere are the top 5 similar products retrieved from the database:\n\n{products_context}\n\nPlease perform your multi-step analysis and provide structured identification results."}
        ])
    ])

    # Create structured output chain
    structured_llm = llm.with_structured_output(ProductIdentification)
    chain = prompt | structured_llm

    # Run the chain
    result = chain.invoke({
        "image_url": image_base64,
        "products_context": "\n\n".join([
            f"RANK {p['rank']} (Similarity: {p['similarity_score']:.3f})\n"
            f"Product Name: {p['product_name']}\n"
            f"Model: {p['model']}\n"
            f"Category: {p['category']}\n"
            f"Brand: {p['brand']}\n"
            f"Specs: {p['specs']}\n"
            f"Price: {p['price']}\n"
            f"Keywords: {p['keywords']}\n"
            f"Usage: {p['usage_description']}\n"
            f"Stock Status: {p['stock_status']}"
            for p in products_context
        ])
    })

    return result


# ============= ENDPOINTS =============

@router.post("/identify-product-from-url", response_model=ProductIdentification)
async def identify_product_from_url(request: ImageURLRequest):
    """
    Identify a product from an image URL using RAG chain
    Perfect for n8n workflows where images are downloaded to URLs

    Args:
        request: JSON body with image_url field (Cloudinary, HTTP, etc.)

    Returns:
        ProductIdentification with structured analysis and results

    Example:
        POST /product-identification/identify-product-from-url
        {
            "image_url": "https://res.cloudinary.com/your-cloud/image/upload/v123/sample.jpg"
        }
    """
    try:
        # Download and load image
        logger.info(f"Downloading image from URL: {request.image_url}")
        image = load_image_from_url(request.image_url)

        # Step 1: Generate embeddings
        logger.info("Generating CLIP embeddings...")
        image_embedding = embed_image(image)

        # Step 2: Retrieve similar products
        logger.info("Retrieving similar products from Qdrant...")
        retrieved_products = retrieve_similar_products(image_embedding, top_k=5)

        if not retrieved_products:
            raise HTTPException(
                status_code=404,
                detail="No similar products found in database"
            )

        # Step 3: Run RAG chain with GPT-4o
        logger.info("Running multi-step RAG chain with GPT-4o...")
        result = run_rag_chain(image, retrieved_products)

        return result

    except HTTPException:
        raise
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download image from URL: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Failed to download image from URL: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """Health check endpoint for product identification service"""
    return {
        "service": "Product Identification",
        "status": "running",
        "model": "CLIP + GPT-4o",
        "vector_db": "Qdrant Cloud",
        "collection": _collection_name
    }
