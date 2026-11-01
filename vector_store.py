"""
FAISS Vector Store module for LocalGPT.
Manages vector index, chunk metadata storage, persistence, and semantic search.
"""
import os
import json
from typing import List, Dict, Any, Optional
import numpy as np
import faiss

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "documents")
INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
METADATA_PATH = os.path.join(DATA_DIR, "chunks_metadata.json")


class FAISSVectorStore:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index: faiss.Index = faiss.IndexFlatIP(dimension)
        self.chunks_metadata: List[Dict[str, Any]] = []
        self.load()

    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray) -> None:
        """Adds text chunks and their corresponding normalized vectors to the FAISS index."""
        if len(chunks) == 0 or embeddings.shape[0] == 0:
            return

        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension {embeddings.shape[1]} does not match index {self.dimension}")

        # Add to FAISS index
        self.index.add(embeddings)
        self.chunks_metadata.extend(chunks)
        self.save()

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Searches the index for top_k most similar chunks.
        Returns list of chunk dicts enriched with 'similarity_score'.
        """
        if self.index.ntotal == 0:
            return []

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks_metadata):
                continue
            chunk = dict(self.chunks_metadata[idx])
            chunk["similarity_score"] = float(score)
            results.append(chunk)

        return results

    def save(self, data_dir: str = DATA_DIR) -> None:
        """Persists the FAISS index and metadata to disk."""
        os.makedirs(data_dir, exist_ok=True)
        idx_path = os.path.join(data_dir, "faiss_index.bin")
        meta_path = os.path.join(data_dir, "chunks_metadata.json")
        
        faiss.write_index(self.index, idx_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks_metadata, f, ensure_ascii=False, indent=2)

    def load(self, data_dir: str = DATA_DIR) -> bool:
        """Loads FAISS index and metadata if available on disk."""
        idx_path = os.path.join(data_dir, "faiss_index.bin")
        meta_path = os.path.join(data_dir, "chunks_metadata.json")

        if os.path.exists(idx_path) and os.path.exists(meta_path):
            try:
                self.index = faiss.read_index(idx_path)
                with open(meta_path, "r", encoding="utf-8") as f:
                    self.chunks_metadata = json.load(f)
                return True
            except Exception as e:
                print(f"[Warning] Failed to load FAISS index: {e}")
                self.index = faiss.IndexFlatIP(self.dimension)
                self.chunks_metadata = []
                return False
        return False

    def clear(self) -> None:
        """Clears all vectors and metadata."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks_metadata = []
        idx_path = os.path.join(DATA_DIR, "faiss_index.bin")
        meta_path = os.path.join(DATA_DIR, "chunks_metadata.json")
        if os.path.exists(idx_path):
            os.remove(idx_path)
        if os.path.exists(meta_path):
            os.remove(meta_path)

    def get_indexed_documents(self) -> List[str]:
        """Returns sorted list of distinct document sources currently indexed."""
        sources = set(chunk.get("source", "Unknown") for chunk in self.chunks_metadata)
        return sorted(list(sources))

    def get_total_chunks(self) -> int:
        return self.index.ntotal


_VECTOR_STORE_INSTANCE: Optional[FAISSVectorStore] = None


def get_vector_store() -> FAISSVectorStore:
    """Singleton getter for the FAISS vector store."""
    global _VECTOR_STORE_INSTANCE
    if _VECTOR_STORE_INSTANCE is None:
        _VECTOR_STORE_INSTANCE = FAISSVectorStore()
    return _VECTOR_STORE_INSTANCE

