"""
FAISS Vector Store module for LocalGPT Pro.
Provides per-user FAISS index and chunk metadata isolation:
Path: data/users/<uid>/faiss_index/
"""
import os
import json
from typing import List, Dict, Any, Optional
import numpy as np
import faiss

BASE_USERS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "users")
LEGACY_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "documents")


def get_user_storage_paths(uid: str) -> Dict[str, str]:
    """Returns directory paths for user-specific documents, FAISS index, and exports."""
    # Sanitize uid for filesystem
    safe_uid = "".join(c for c in uid if c.isalnum() or c in ("-", "_", "@", "."))
    user_root = os.path.join(BASE_USERS_DIR, safe_uid)
    docs_dir = os.path.join(user_root, "documents")
    faiss_dir = os.path.join(user_root, "faiss_index")
    exports_dir = os.path.join(user_root, "exports")
    
    os.makedirs(docs_dir, exist_ok=True)
    os.makedirs(faiss_dir, exist_ok=True)
    os.makedirs(exports_dir, exist_ok=True)
    
    return {
        "root": user_root,
        "documents": docs_dir,
        "faiss_index": faiss_dir,
        "exports": exports_dir,
        "index_file": os.path.join(faiss_dir, "faiss_index.bin"),
        "metadata_file": os.path.join(faiss_dir, "chunks_metadata.json")
    }


class FAISSVectorStore:
    def __init__(self, dimension: int = 384, storage_dir: Optional[str] = None):
        self.dimension = dimension
        self.storage_dir = storage_dir or LEGACY_DATA_DIR
        self.index: faiss.Index = faiss.IndexFlatIP(dimension)
        self.chunks_metadata: List[Dict[str, Any]] = []
        self.load()

    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray) -> None:
        """Adds text chunks and their corresponding normalized vectors to the FAISS index."""
        if len(chunks) == 0 or embeddings.shape[0] == 0:
            return

        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension {embeddings.shape[1]} does not match index {self.dimension}")

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

    def save(self, data_dir: Optional[str] = None) -> None:
        """Persists the FAISS index and metadata to disk."""
        target_dir = data_dir or self.storage_dir
        os.makedirs(target_dir, exist_ok=True)
        idx_path = os.path.join(target_dir, "faiss_index.bin")
        meta_path = os.path.join(target_dir, "chunks_metadata.json")
        
        faiss.write_index(self.index, idx_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks_metadata, f, ensure_ascii=False, indent=2)

    def load(self, data_dir: Optional[str] = None) -> bool:
        """Loads FAISS index and metadata if available on disk."""
        target_dir = data_dir or self.storage_dir
        idx_path = os.path.join(target_dir, "faiss_index.bin")
        meta_path = os.path.join(target_dir, "chunks_metadata.json")

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

    def remove_document(self, filename: str) -> None:
        """Removes chunks belonging to a document and rebuilds the FAISS index."""
        remaining_chunks = [c for c in self.chunks_metadata if c.get("source") != filename]
        if len(remaining_chunks) == len(self.chunks_metadata):
            return
        
        # If removing all chunks
        if len(remaining_chunks) == 0:
            self.clear()
            return

        # Re-embed remaining chunks
        from embeddings import embed_texts
        self.clear()
        texts = [c["text"] for c in remaining_chunks]
        new_embeddings = embed_texts(texts)
        self.add_chunks(remaining_chunks, new_embeddings)

    def clear(self) -> None:
        """Clears all vectors and metadata."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks_metadata = []
        idx_path = os.path.join(self.storage_dir, "faiss_index.bin")
        meta_path = os.path.join(self.storage_dir, "chunks_metadata.json")
        if os.path.exists(idx_path):
            try:
                os.remove(idx_path)
            except OSError:
                pass
        if os.path.exists(meta_path):
            try:
                os.remove(meta_path)
            except OSError:
                pass

    def get_indexed_documents(self) -> List[str]:
        """Returns sorted list of distinct document sources currently indexed."""
        sources = set(chunk.get("source", "Unknown") for chunk in self.chunks_metadata)
        return sorted(list(sources))

    def get_total_chunks(self) -> int:
        return self.index.ntotal


# Per-user instances cache
_USER_VECTOR_STORES: Dict[str, FAISSVectorStore] = {}
_LEGACY_STORE: Optional[FAISSVectorStore] = None


def get_user_vector_store(uid: str) -> FAISSVectorStore:
    """Returns the isolated FAISSVectorStore instance for a given user UID."""
    global _USER_VECTOR_STORES
    if uid not in _USER_VECTOR_STORES:
        paths = get_user_storage_paths(uid)
        _USER_VECTOR_STORES[uid] = FAISSVectorStore(dimension=384, storage_dir=paths["faiss_index"])
    return _USER_VECTOR_STORES[uid]


def get_vector_store() -> FAISSVectorStore:
    """Singleton getter for default/legacy FAISS vector store."""
    global _LEGACY_STORE
    if _LEGACY_STORE is None:
        _LEGACY_STORE = FAISSVectorStore(dimension=384, storage_dir=LEGACY_DATA_DIR)
    return _LEGACY_STORE
