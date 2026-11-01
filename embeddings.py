"""
Embeddings module for LocalGPT RAG.
Uses SentenceTransformers to generate dense semantic vector embeddings.
"""
from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

_EMBEDDER_INSTANCE: Optional[SentenceTransformer] = None
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def get_embedder(model_name: str = DEFAULT_EMBEDDING_MODEL) -> SentenceTransformer:
    """Singleton getter for the SentenceTransformer model."""
    global _EMBEDDER_INSTANCE
    if _EMBEDDER_INSTANCE is None:
        _EMBEDDER_INSTANCE = SentenceTransformer(model_name)
    return _EMBEDDER_INSTANCE


def embed_texts(texts: List[str], model_name: str = DEFAULT_EMBEDDING_MODEL) -> np.ndarray:
    """
    Computes normalized embeddings for a list of texts.
    Returns float32 NumPy array with shape (N, dim).
    """
    if not texts:
        return np.empty((0, 384), dtype=np.float32)
    model = get_embedder(model_name)
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings.astype(np.float32)


def embed_query(query: str, model_name: str = DEFAULT_EMBEDDING_MODEL) -> np.ndarray:
    """
    Computes normalized embedding for a single query string.
    Returns float32 NumPy array with shape (1, dim).
    """
    model = get_embedder(model_name)
    embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    return embedding.astype(np.float32)

