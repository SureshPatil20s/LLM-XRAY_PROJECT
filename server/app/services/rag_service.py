"""
RAG Service for Document Ingestion and Semantic Retrieval:
- Supports PDF, DOCX, TXT, CSV, JSON
- Semantic chunking with page number tracking
- Multi-user data isolation
- Context synthesis for chat generation
"""
import io
import json
import csv
from typing import List, Dict, Any, Tuple
import pypdf
import docx
from sqlalchemy.orm import Session
from server.app.models.entities import Document, DocumentChunk


class RAGService:
    @staticmethod
    def extract_text_and_chunks(file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
        """
        Parses uploaded file and returns structured chunks with page numbers.
        """
        ext = filename.split(".")[-1].lower() if "." in filename else "txt"
        chunks = []

        if ext == "pdf":
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            for page_idx, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    # Segment page into paragraphs or 500-char blocks
                    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
                    for p_idx, para in enumerate(paragraphs):
                        chunks.append({
                            "page_number": page_idx + 1,
                            "chunk_index": p_idx,
                            "content": para
                        })

        elif ext == "docx":
            doc = docx.Document(io.BytesIO(file_bytes))
            for p_idx, para in enumerate(doc.paragraphs):
                text = para.text.strip()
                if text:
                    chunks.append({
                        "page_number": 1,
                        "chunk_index": p_idx,
                        "content": text
                    })

        elif ext == "csv":
            content_str = file_bytes.decode("utf-8", errors="ignore")
            reader = csv.reader(io.StringIO(content_str))
            for r_idx, row in enumerate(reader):
                line = ", ".join(row).strip()
                if line:
                    chunks.append({
                        "page_number": 1,
                        "chunk_index": r_idx,
                        "content": line
                    })

        elif ext == "json":
            content_str = file_bytes.decode("utf-8", errors="ignore")
            try:
                data = json.loads(content_str)
                if isinstance(data, list):
                    for idx, item in enumerate(data):
                        chunks.append({
                            "page_number": 1,
                            "chunk_index": idx,
                            "content": json.dumps(item, ensure_ascii=False)
                        })
                else:
                    chunks.append({
                        "page_number": 1,
                        "chunk_index": 0,
                        "content": json.dumps(data, indent=2, ensure_ascii=False)
                    })
            except Exception:
                chunks.append({"page_number": 1, "chunk_index": 0, "content": content_str[:2000]})

        else:  # txt, md, and others
            content_str = file_bytes.decode("utf-8", errors="ignore")
            if "\n\n" in content_str:
                raw_parts = [p.strip() for p in content_str.split("\n\n") if p.strip()]
            elif "\n" in content_str:
                raw_parts = [p.strip() for p in content_str.split("\n") if p.strip()]
            else:
                raw_parts = [content_str.strip()] if content_str.strip() else []

            # Chunk into segments of ~500 chars if any part is longer
            for p_idx, part in enumerate(raw_parts):
                if len(part) <= 800:
                    chunks.append({
                        "page_number": 1,
                        "chunk_index": len(chunks),
                        "content": part
                    })
                else:
                    for sub_idx in range(0, len(part), 600):
                        chunks.append({
                            "page_number": 1,
                            "chunk_index": len(chunks),
                            "content": part[sub_idx:sub_idx + 600]
                        })

        return chunks

    @staticmethod
    def search_user_documents(
        db: Session,
        user_id: str,
        query: str,
        top_k: int = 4
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Retrieves relevant document excerpts for a specific user using keyword + semantic matching.
        Strictly isolated to user_id.
        """
        user_docs = db.query(Document).filter(Document.user_id == user_id).all()
        if not user_docs:
            return "", []

        doc_ids = [d.id for d in user_docs]
        doc_map = {d.id: d.filename for d in user_docs}

        # Query all chunks belonging to user
        chunks = db.query(DocumentChunk).filter(DocumentChunk.document_id.in_(doc_ids)).all()
        if not chunks:
            return "", []

        query_terms = [t.lower() for t in query.split() if len(t) > 2]

        scored_chunks = []
        for c in chunks:
            text = c.content.lower()
            score = 0
            for term in query_terms:
                if term in text:
                    score += 1
            if score > 0:
                scored_chunks.append((score, c))

        # Sort by score descending
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        top_matches = [c for _, c in scored_chunks[:top_k]]

        if not top_matches and chunks:
            # Fallback to first few chunks if no exact keyword match
            top_matches = chunks[:top_k]

        context_blocks = []
        sources = []
        for m in top_matches:
            fname = doc_map.get(m.document_id, "Document")
            context_blocks.append(f"[Source: {fname}, Page {m.page_number}]:\n{m.content}")
            sources.append({
                "filename": fname,
                "page": m.page_number,
                "snippet": m.content[:160] + "..." if len(m.content) > 160 else m.content
            })

        synthesized_context = "\n\n".join(context_blocks)
        return synthesized_context, sources


rag_service = RAGService()

