"""
Document Loader and Chunker for LocalGPT.
Supports PDF (pypdf), DOCX (python-docx), and plain text (TXT).
Preserves page numbers and file source provenance for RAG citations.
"""
import os
import re
from typing import List, Dict, Any, Union, BinaryIO
from pypdf import PdfReader
from docx import Document

DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "documents")


def save_uploaded_file(file_obj: Union[BinaryIO, Any, str, bytes], filename: str) -> str:
    """Saves uploaded file to data/documents/ and returns full path."""
    os.makedirs(DOCS_DIR, exist_ok=True)
    file_path = os.path.join(DOCS_DIR, filename)

    if isinstance(file_obj, str) and os.path.abspath(file_obj) == os.path.abspath(file_path):
        return file_path

    with open(file_path, "wb") as f:
        if hasattr(file_obj, "read"):
            f.write(file_obj.read())
            if hasattr(file_obj, "seek"):
                file_obj.seek(0)
        elif isinstance(file_obj, bytes):
            f.write(file_obj)
        elif isinstance(file_obj, str) and os.path.exists(file_obj):
            with open(file_obj, "rb") as src:
                f.write(src.read())
        else:
            raise ValueError(f"Unsupported file object type: {type(file_obj)}")
    return file_path


def load_pdf(file_path: str) -> List[Dict[str, Any]]:
    """Loads a PDF file and returns list of per-page texts."""
    reader = PdfReader(file_path)
    pages_data = []
    filename = os.path.basename(file_path)
    for idx, page in enumerate(reader.pages):
        page_num = idx + 1
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages_data.append({
                "source": filename,
                "page": page_num,
                "text": text
            })
    return pages_data


def load_docx(file_path: str) -> List[Dict[str, Any]]:
    """Loads a DOCX file and returns structured paragraph chunks."""
    doc = Document(file_path)
    filename = os.path.basename(file_path)
    full_text = "\n\n".join([p.text.strip() for p in doc.paragraphs if p.text.strip()])
    return [{
        "source": filename,
        "page": 1,
        "text": full_text
    }]


def load_txt(file_path: str) -> List[Dict[str, Any]]:
    """Loads a plain text file."""
    filename = os.path.basename(file_path)
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read().strip()
    return [{
        "source": filename,
        "page": 1,
        "text": text
    }]


def load_document(file_path: str) -> List[Dict[str, Any]]:
    """Dispatches document loading based on file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return load_pdf(file_path)
    elif ext in [".docx", ".doc"]:
        return load_docx(file_path)
    elif ext in [".txt", ".md", ".csv", ".json"]:
        return load_txt(file_path)
    else:
        # Fallback to plain text
        return load_txt(file_path)


def chunk_text(
    pages_data: List[Dict[str, Any]],
    chunk_size: int = 600,
    chunk_overlap: int = 100
) -> List[Dict[str, Any]]:
    """
    Splits document pages into smaller overlapping chunks for RAG embedding.
    Preserves document source and page number in each chunk metadata.
    """
    all_chunks = []
    chunk_counter = 0

    for page_item in pages_data:
        text = page_item["text"]
        source = page_item["source"]
        page_num = page_item.get("page", 1)

        # Normalize whitespace while preserving paragraphs
        clean_text = re.sub(r"[ \t]+", " ", text).strip()
        if not clean_text:
            continue

        start = 0
        text_len = len(clean_text)

        while start < text_len:
            end = min(start + chunk_size, text_len)
            
            # If not at the end of text, attempt to break cleanly at sentence or word boundary
            if end < text_len:
                boundary = clean_text.rfind(". ", start, end)
                if boundary == -1 or boundary <= start + (chunk_size // 2):
                    boundary = clean_text.rfind(" ", start, end)
                if boundary != -1 and boundary > start + (chunk_size // 2):
                    end = boundary + 1

            chunk_content = clean_text[start:end].strip()
            if len(chunk_content) > 20:  # Avoid trivial empty chunks
                chunk_counter += 1
                all_chunks.append({
                    "chunk_id": chunk_counter,
                    "text": chunk_content,
                    "source": source,
                    "page": page_num
                })

            if end >= text_len:
                break
            start = max(start + 1, end - chunk_overlap)

    return all_chunks
