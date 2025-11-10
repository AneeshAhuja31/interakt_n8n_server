from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, PointStruct
import os
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import uuid
import math

load_dotenv(override=True)

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Use OpenAI embeddings. Ensure OPENAI_API_KEY is set in the environment (.env or system env).
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY not set. Please add it to your environment or .env file.")

# Use the smaller OpenAI embedding model (good quality / cost balance)
# Explicitly set the model and vector_dim to avoid calling the API at import time.
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small")

collection_name = "product_text"
# text-embedding-3-small produces 1536-dimensional vectors
vector_dim = 1536

if collection_name not in [c.name for c in qdrant_client.get_collections().collections]:
    qdrant_client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_dim, distance="Cosine")
    )

df = pd.read_csv(r"C:\Users\HP\OneDrive\Desktop\interakt_n8n\intent_classifier_api\ingestion\debron.csv")

text_columns = [
    "Product Name", "Model", "Category", "Specs", "Price (INR)",
    "Discount Info", "Brand", "Keywords", "Usage Description"
]
def embed_text():
    df["combined_text"] = df[text_columns].astype(str).agg(" ".join, axis=1)

    df["cloudinary_url"] = df["cloudinary_url"].fillna("")

    batch_size = 50  
    total_rows = len(df)
    num_batches = math.ceil(total_rows / batch_size)

    for batch_num in range(num_batches):
        start = batch_num * batch_size
        end = min((batch_num + 1) * batch_size, total_rows)
        batch_df = df.iloc[start:end]

        batch_texts = batch_df["combined_text"].tolist()
        batch_vectors = embeddings.embed_documents(batch_texts)

        # Helper function to safely convert pandas values to Python types
        def safe_convert(value):
            """Convert pandas values to native Python types, handling NaN/None"""
            if pd.isna(value):
                return ""
            return str(value) if value is not None else ""

        batch_points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=batch_vectors[i],
                payload={
                    "product_name": safe_convert(batch_df["Product Name"].iloc[i]),
                    "model": safe_convert(batch_df["Model"].iloc[i]),
                    "category": safe_convert(batch_df["Category"].iloc[i]),
                    "specs": safe_convert(batch_df["Specs"].iloc[i]),
                    "price_inr": safe_convert(batch_df["Price (INR)"].iloc[i]),
                    "discount_info": safe_convert(batch_df["Discount Info"].iloc[i]),
                    "brand": safe_convert(batch_df["Brand"].iloc[i]),
                    "keywords": safe_convert(batch_df["Keywords"].iloc[i]),
                    "usage_description": safe_convert(batch_df["Usage Description"].iloc[i]),
                    "cloudinary_url": safe_convert(batch_df["cloudinary_url"].iloc[i]),
                    "content": batch_texts[i],  # Add the combined text used for embedding
                },
            )
            for i in range(len(batch_df))
        ]

        qdrant_client.upsert(
            collection_name=collection_name,
            points=batch_points
        )

        print(f"Uploaded batch {batch_num+1}/{num_batches} ({len(batch_points)} items)")

    print(f"\nDone! Total {total_rows} product embeddings uploaded to Qdrant.")


def test_query(query_text: str, limit: int = 20):
    print(f"\nSearching for: '{query_text}'")

    query_vector = embeddings.embed_query(query_text)
    search_results = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=limit
    )

    if not search_results:
        print("⚠️ No matches found.")
        return

    for i, res in enumerate(search_results, 1):
        payload = res.payload
        print(f"\n{i}. {payload.get('product_name', 'N/A')} ({payload.get('brand', 'Unknown brand')})")
        print(f"   Category: {payload.get('category', 'N/A')}")
        print(f"   Model: {payload.get('model', 'N/A')}")
        print(f"   Price: {payload.get('price_inr', 'N/A')}")
        print(f"   Score: {res.score:.3f}")
        if payload.get("cloudinary_url"):
            print(f"   Image: {payload['cloudinary_url']}")

test_query("i need debron drill")
# embed_text()