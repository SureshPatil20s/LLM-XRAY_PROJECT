"""
RAG (Retrieval-Augmented Generation) Pipeline for LocalGPT Pro.
Provides user-isolated document processing, vector storage, and semantic retrieval:
- Per-user document folder: data/users/<uid>/documents/
- Per-user FAISS index: data/users/<uid>/faiss_index/
- Fallback to global store if uid is omitted
"""
import os
import shutil
from typing import List, Dict, Any, Optional, Tuple
from document_loader import load_document, chunk_text, save_uploaded_file
from embeddings import embed_texts, embed_query
from vector_store import get_vector_store, get_user_vector_store, get_user_storage_paths
import database


def save_user_document_file(file_path_or_obj: Any, filename: str, uid: str) -> str:
    """Saves uploaded file to data/users/<uid>/documents/<filename>."""
    paths = get_user_storage_paths(uid)
    dest_path = os.path.join(paths["documents"], filename)

    if hasattr(file_path_or_obj, "read"):
        with open(dest_path, "wb") as f:
            f.write(file_path_or_obj.read())
    elif isinstance(file_path_or_obj, str) and os.path.exists(file_path_or_obj):
        if os.path.abspath(file_path_or_obj) != os.path.abspath(dest_path):
            shutil.copy2(file_path_or_obj, dest_path)
    return dest_path


def ingest_document(
    file_path_or_obj: Any,
    filename: str,
    uid: Optional[str] = None
) -> Dict[str, Any]:
    """
    Ingests a document into the RAG vector database for a specific user:
    1. Saves file to data/users/<uid>/documents/
    2. Loads text per page
    3. Splits text into overlapping semantic chunks
    4. Computes normalized vector embeddings
    5. Stores chunks and vectors in user's isolated FAISS index
    6. Records document in database
    """
    if uid:
        saved_path = save_user_document_file(file_path_or_obj, filename, uid)
        store = get_user_vector_store(uid)
    else:
        saved_path = save_uploaded_file(file_path_or_obj, filename)
        store = get_vector_store()

    pages_data = load_document(saved_path)
    if not pages_data:
        return {"filename": filename, "chunks": 0, "pages": 0, "status": "empty"}

    chunks = chunk_text(pages_data, chunk_size=600, chunk_overlap=100)
    if not chunks:
        return {"filename": filename, "chunks": 0, "pages": len(pages_data), "status": "no_chunks"}

    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)

    store.add_chunks(chunks, embeddings)

    # Record in database if uid provided
    if uid:
        database.add_user_document(uid, filename, len(pages_data), len(chunks))

    return {
        "filename": filename,
        "chunks": len(chunks),
        "pages": len(pages_data),
        "status": "success"
    }


def retrieve_context(
    query: str,
    top_k: int = 4,
    uid: Optional[str] = None
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Retrieves the top_k most relevant chunks for a user query from the user's isolated index.
    """
    store = get_user_vector_store(uid) if uid else get_vector_store()

    if store.get_total_chunks() == 0:
        return "", []

    query_emb = embed_query(query)
    matched_chunks = store.search(query_emb, top_k=top_k)

    if not matched_chunks:
        return "", []

    context_parts = []
    sources_seen = set()
    sources_list = []

    for chunk in matched_chunks:
        source_name = chunk.get("source", "Document")
        page_num = chunk.get("page", 1)
        text = chunk.get("text", "")

        context_parts.append(
            f"--- Document: {source_name} (Page {page_num}) ---\n{text}"
        )

        source_key = (source_name, page_num)
        if source_key not in sources_seen:
            sources_seen.add(source_key)
            sources_list.append({
                "source": source_name,
                "page": page_num,
                "snippet": text[:120] + ("..." if len(text) > 120 else "")
            })

    formatted_context_str = "\n\n".join(context_parts)
    return formatted_context_str, sources_list


def clear_user_documents(uid: str) -> None:
    """Clears all documents and FAISS index for a user."""
    store = get_user_vector_store(uid)
    store.clear()
    paths = get_user_storage_paths(uid)
    if os.path.exists(paths["documents"]):
        for f in os.listdir(paths["documents"]):
            p = os.path.join(paths["documents"], f)
            if os.path.isfile(p):
                os.remove(p)


def clear_all_documents(uid: Optional[str] = None) -> None:
    """Clears vector store and document records."""
    if uid:
        clear_user_documents(uid)
    else:
        store = get_vector_store()
        store.clear()


def delete_single_document(uid: str, doc_id: str) -> bool:
    """Deletes a specific document from user's index, disk, and database."""
    filename = database.delete_user_document(uid, doc_id)
    if not filename:
        return False

    # Remove from FAISS index
    store = get_user_vector_store(uid)
    store.remove_document(filename)

    # Remove file from disk
    paths = get_user_storage_paths(uid)
    file_path = os.path.join(paths["documents"], filename)
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
        except OSError:
            pass

    return True


def get_indexed_summary(uid: Optional[str] = None) -> Dict[str, Any]:
    """Returns summary of indexed documents for user or global."""
    if uid:
        store = get_user_vector_store(uid)
        db_docs = database.get_user_documents(uid)
        return {
            "documents": [d["filename"] for d in db_docs] or store.get_indexed_documents(),
            "total_chunks": store.get_total_chunks(),
            "records": db_docs
        }
    store = get_vector_store()
    return {
        "documents": store.get_indexed_documents(),
        "total_chunks": store.get_total_chunks(),
        "records": []
    }
