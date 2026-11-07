"""
Documents & Knowledge Base API:
- Upload files (PDF, DOCX, TXT, CSV, JSON)
- Auto-extract semantic chunks with page numbering
- List user documents
- Delete document
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from sqlalchemy.orm import Session
from typing import List

from server.app.db.session import get_db
from server.app.models.entities import User, Document, DocumentChunk
from server.app.schemas.schemas import DocumentResponse
from server.app.core.security import get_current_user
from server.app.services.rag_service import rag_service

router = APIRouter(prefix="/documents", tags=["Documents & RAG"])


@router.get("", response_model=List[DocumentResponse])
def get_documents(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Lists all uploaded documents for the authenticated user."""
    docs = db.query(Document).filter(Document.user_id == current_user.id).order_by(Document.created_at.desc()).all()
    for d in docs:
        setattr(d, "chunk_count", d.total_chunks)
    return docs


@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Uploads and indexes a document (PDF, DOCX, TXT, CSV, JSON).
    Strictly isolated to current user.
    """
    file_bytes = await file.read()
    filename = file.filename or "uploaded_file.txt"
    ext = filename.split(".")[-1].lower() if "." in filename else "txt"

    allowed_exts = {"pdf", "docx", "txt", "csv", "json"}
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type .{ext}. Allowed: PDF, DOCX, TXT, CSV, JSON"
        )

    chunks_data = rag_service.extract_text_and_chunks(file_bytes, filename)
    max_pages = max([c.get("page_number", 1) for c in chunks_data], default=1)

    doc = Document(
        user_id=current_user.id,
        filename=filename,
        file_type=ext,
        file_size=len(file_bytes),
        total_pages=max_pages,
        total_chunks=len(chunks_data)
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    for item in chunks_data:
        chunk = DocumentChunk(
            document_id=doc.id,
            page_number=item.get("page_number", 1),
            chunk_index=item.get("chunk_index", 0),
            content=item.get("content", "")
        )
        db.add(chunk)
    db.commit()

    setattr(doc, "chunk_count", doc.total_chunks)
    return doc


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    document_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Deletes a document and all its indexed chunks."""
    doc = db.query(Document).filter(
        Document.id == document_id,
        Document.user_id == current_user.id
    ).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    db.delete(doc)
    db.commit()
    return None

