import logging
import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain.schema import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from unidecode import unidecode
from rapidfuzz import process, fuzz
from dotenv import load_dotenv
import re

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Real Estate RAG API", version="1.0.0")

HF_API_KEY = os.getenv("HF_API_KEY")
MODEL_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1"  # Replace with your preferred model

logger.info("Starting Real Estate RAG API server...")


def query_huggingface_model(prompt: str, max_length: int = 200) -> str:
    if not HF_API_KEY:
        logger.error("HF_API_KEY not set")
        return "Sorry, the AI service is not configured."

    try:
        headers = {
            "Authorization": f"Bearer {HF_API_KEY}"
        }
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_length,
                "temperature": 0.7,
                "do_sample": True,
                "top_p": 0.9
            }
        }

        response = requests.post(
            f"https://api-inference.huggingface.co/models/{MODEL_ID}",
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        output = response.json()
        if isinstance(output, list) and len(output) > 0 and "generated_text" in output[0]:
            generated_text = output[0]["generated_text"]
            # Remove prompt echo if present
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            return generated_text
        else:
            logger.error(f"Unexpected response format from Hugging Face API: {output}")
            return "Sorry, I could not generate a response."
    except Exception as e:
        logger.error(f"Hugging Face API error: {e}")
        return "Sorry, I encountered an error processing your request."


# ------------------- Data Models -------------------

class PropertyMetadata(BaseModel):
    owner_id: int
    price: int = Field(gt=0)
    image: str
    address: str
    city: str
    type: str
    offer_type: str
    area: int = Field(gt=0)
    rooms: int = Field(ge=0)
    bathrooms: int = Field(ge=0)
    featured: bool
    created_at: str
    updated_at: str


class PropertyItem(BaseModel):
    id: int
    title: str
    content: str
    metadata: PropertyMetadata


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    properties: List[PropertyItem] = Field(min_items=1)


class ChatResponse(BaseModel):
    reply: str
    properties_found: int
    chunks_processed: int


# ------------------- Embeddings -------------------

class SimpleEmbeddings:
    def __init__(self, max_features: int = 1000):
        self.vectorizer = None
        self.fitted = False
        self.max_features = max_features

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        if not texts:
            raise ValueError("No texts provided for embedding")
        min_df = 1
        max_df = 1.0 if len(texts) <= 2 else 0.95
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=min_df,
            max_df=max_df
        )
        vectors = self.vectorizer.fit_transform(texts)
        self.fitted = True
        return vectors.toarray()

    def embed_query(self, text: str) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Must fit on documents first")
        if not text.strip():
            raise ValueError("Query text cannot be empty")
        return self.vectorizer.transform([text]).toarray()[0]


# ------------------- Helper Functions -------------------

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    if embeddings.size == 0:
        raise ValueError("No embeddings provided")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    logger.info(f"Built FAISS index with {embeddings.shape[0]} vectors of dimension {dim}")
    return index


def detect_city_in_query(query: str, available_cities: List[str]) -> Optional[str]:
    query_norm = unidecode(query).lower()
    cities_norm = [unidecode(city).lower() for city in available_cities]
    best_match = process.extractOne(query_norm, cities_norm, scorer=fuzz.ratio)
    if best_match and best_match[1] >= 60:
        matched_city_norm = best_match[0]
        for orig_city in available_cities:
            if unidecode(orig_city).lower() == matched_city_norm:
                return orig_city
    return None


def format_context(docs: List[LangchainDocument], max_context_length: int = 2000) -> str:
    context_parts = []
    current_length = 0
    for doc in docs:
        title = doc.metadata.get("title", "N/A")
        city = doc.metadata.get("city", "N/A")
        price = doc.metadata.get("price", "N/A")
        property_type = doc.metadata.get("type", "N/A")
        area = doc.metadata.get("area", "N/A")
        rooms = doc.metadata.get("rooms", "N/A")
        # Extract description line (usually second line in page_content)
        description = ""
        lines = doc.page_content.split("\n")
        if len(lines) > 1:
            description = lines[1].strip()
        price_str = f"${price:,}" if isinstance(price, int) else str(price)

        context_part = (
            f"Title: {title}\n"
            f"Description: {description}\n"
            f"Location: {city}\n"
            f"Type: {property_type}\n"
            f"Price: {price_str}\n"
            f"Area: {area} sqft\n"
            f"Rooms: {rooms}\n"
            "---\n"
        )
        if current_length + len(context_part) > max_context_length:
            break
        context_parts.append(context_part)
        current_length += len(context_part)
    return "\n".join(context_parts)


def format_properties_with_linebreaks(text: str) -> str:
    # Split on bullets (*, -, •) or numbered lists if present
    parts = re.split(r'[\*\-\u2022]\s*', text)
    parts = [p.strip() for p in parts if p.strip()]
    return "\n\n".join(parts)


# ------------------- Routes -------------------

@app.get("/health")
async def health_check():
    logger.info("Health check requested")
    return {"status": "healthy", "service": "Real Estate RAG API"}


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    logger.info(f"Received chat request with message: {request.message}")
    try:
        query_raw = request.message.strip()

        # Optional: clean common instruction phrases from user query to avoid echo
        query_cleaned = re.sub(r"please provide a list.*", "", query_raw, flags=re.I).strip()

        query = unidecode(query_cleaned).lower()

        available_cities = [p.metadata.city for p in request.properties]
        detected_city = detect_city_in_query(query, available_cities)

        filtered_props = [
            p for p in request.properties
            if detected_city and unidecode(p.metadata.city).lower() == unidecode(detected_city).lower()
        ] or request.properties

        docs = []
        for prop in filtered_props:
            enhanced_content = (
                f"{prop.title}\n{prop.content}\n"
                f"Location: {prop.metadata.city}\n"
                f"Type: {prop.metadata.type}\n"
                f"Price: {prop.metadata.price}\n"
                f"Area: {prop.metadata.area}\n"
                f"Rooms: {prop.metadata.rooms}"
            )
            metadata_dict = prop.metadata.model_dump() if hasattr(prop.metadata, "model_dump") else prop.metadata.dict()
            docs.append(LangchainDocument(page_content=enhanced_content, metadata={"title": prop.title, **metadata_dict}))

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        split_docs = splitter.split_documents(docs)
        texts = [doc.page_content for doc in split_docs]

        embedder = SimpleEmbeddings(max_features=min(1000, len(texts) * 10))
        doc_embeddings = embedder.embed_documents(texts).astype('float32')

        index = build_faiss_index(doc_embeddings)

        query_embedding = embedder.embed_query(query)
        norm = np.linalg.norm(query_embedding)
        query_embedding = (query_embedding / norm) if norm > 0 else np.ones_like(query_embedding) / np.sqrt(len(query_embedding))
        query_embedding = np.expand_dims(query_embedding, axis=0).astype('float32')

        k = min(5, len(split_docs))
        D, I = index.search(query_embedding, k=k)
        top_docs = [split_docs[i] for i in I[0]]

        context = format_context(top_docs, max_context_length=1000)

        # FIXED PROMPT: do NOT include the instruction phrase inside the reply
        prompt = (
            f"{context}\n"
            f"Client query: \"{query_cleaned}\"\n\n"
            "Summarize the matching properties clearly and concisely. "
            "For each property, include title, brief description, location, type, price, area, and rooms. "
            "Provide the summary only — do NOT include instructions or extra text."
        )

        ai_reply = query_huggingface_model(prompt, max_length=300)

        ai_reply_formatted = format_properties_with_linebreaks(ai_reply)

        if not ai_reply_formatted or len(ai_reply_formatted.strip()) < 15:
            city_text = f" in {detected_city}" if detected_city else ""
            reply = f"Based on the properties I found{city_text}, here is a summary:\n\n"
            for i, doc in enumerate(top_docs[:3], 1):
                title = doc.metadata.get("title", "Property")
                city = doc.metadata.get("city", "Unknown location")
                price = doc.metadata.get("price", "Price not available")
                property_type = doc.metadata.get("type", "Property")
                area = doc.metadata.get("area", "Unknown")
                rooms = doc.metadata.get("rooms", "Unknown")
                price_str = f"${price:,}" if isinstance(price, int) else str(price)
                reply += (
                    f"{i}. {title} for {price_str}. "
                    f"Located in {city}, this {property_type} has an area of {area} sqft and {rooms} rooms.\n\n"
                )
        else:
            reply = ai_reply_formatted.strip()

        logger.info("Returning reply to client")
        return ChatResponse(
            reply=reply,
            properties_found=len(top_docs),
            chunks_processed=len(split_docs)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error in /chat endpoint")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# ------------------- Run Server -------------------

if __name__ == "__main__":
    import uvicorn
    logger.info("Running server on 0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
