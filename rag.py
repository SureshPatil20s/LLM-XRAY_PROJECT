"""
RAG (Retrieval-Augmented Generation) Pipeline for LocalGPT.
Connects Document Loader, Semantic Embeddings, FAISS Vector Store,
and context formatting with source attribution.
"""
from typing import List, Dict, Any, Optional, Tuple
from document_loader import load_document, chunk_text, save_uploaded_file
from embeddings import embed_texts, embed_query
from vector_store import get_vector_store


def ingest_document(file_path_or_obj: Any, filename: str) -> Dict[str, Any]:
    """
    Ingests a document into the RAG vector database:
    1. Saves file to data/documents/
    2. Loads text per page
    3. Splits text into overlapping semantic chunks
    4. Computes normalized vector embeddings
    5. Stores chunks and vectors in FAISS
    """
    saved_path = save_uploaded_file(file_path_or_obj, filename)
    pages_data = load_document(saved_path)
    
    if not pages_data:
        return {"filename": filename, "chunks": 0, "pages": 0, "status": "empty"}

    chunks = chunk_text(pages_data, chunk_size=600, chunk_overlap=100)
    if not chunks:
        return {"filename": filename, "chunks": 0, "pages": len(pages_data), "status": "no_chunks"}

    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)

    store = get_vector_store()
    store.add_chunks(chunks, embeddings)

    return {
        "filename": filename,
        "chunks": len(chunks),
        "pages": len(pages_data),
        "status": "success"
    }


def retrieve_context(query: str, top_k: int = 4) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Retrieves the top_k most relevant chunks for a user query.
    Returns:
    - formatted_context_str: Cleanly structured context to inject into prompt.
    - sources_list: List of distinct source references (filename & page).
    """
    store = get_vector_store()
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


def clear_all_documents() -> None:
    """Clears the FAISS index and metadata."""
    store = get_vector_store()
    store.clear()


def get_indexed_summary() -> Dict[str, Any]:
    """Returns information on indexed documents and total chunks."""
    store = get_vector_store()
    return {
        "documents": store.get_indexed_documents(),
        "total_chunks": store.get_total_chunks()
    }

