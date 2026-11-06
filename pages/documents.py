"""
Document Manager Page for LocalGPT Pro.
Provides:
- Drag-and-drop document upload (PDF, TXT, DOCX) with stage progress bar
- Per-user document library listing (pages, chunks, date)
- Per-user FAISS index management & document deletion
"""
import streamlit as st
import time
import database
from auth.guards import require_login
from ui.theme import apply_theme
from ui.sidebar import render_sidebar
import rag

# Page Setup
st.set_page_config(
    page_title="Document Library — SURESH.GPT Pro",
    page_icon="📁",
    layout="wide",
    initial_sidebar_state="expanded"
)

database.init_db()

# Enforce login guard
user = require_login()
uid = user["uid"]

# Apply user theme
active_theme = st.session_state.get("theme", user.get("theme", "dark"))
apply_theme(active_theme)

# Render Sidebar
render_sidebar(user, active_conv_id=st.session_state.get("active_conv_id"), current_page="documents")


def render_documents_page():
    # Top navigation bar
    col_back, col_actions = st.columns([2.5, 7.5])
    with col_back:
        if st.button("← Back to Chat Workspace", key="doc_back_btn", type="primary", use_container_width=True):
            try:
                st.switch_page("pages/workspace.py")
            except Exception:
                st.rerun()

    st.markdown(f"""
    <div style="margin-top: 10px; margin-bottom: 24px;">
        <h1 style="font-size: 2.1rem; font-weight: 800; color: #ececec; margin: 0; letter-spacing: -0.02em;">
            📁 My Document Library (RAG)
        </h1>
        <p style="font-size: 0.92rem; color: #8e8ea0; margin-top: 4px;">
            Manage and index private reference materials. All files and FAISS embeddings are strictly isolated to your account (<code>{user.get('email')}</code>).
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Ingestion Card
    st.markdown("""
    <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 16px; padding: 22px; margin-bottom: 24px;">
        <h3 style="margin-top:0; color:#38bdf8; font-size:1.15rem;">📥 Ingest New Document</h3>
    """, unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, or DOCX",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        help="Files are parsed, chunked with semantic overlap, embedded with SentenceTransformers, and added to your private FAISS vector store."
    )

    if uploaded_files:
        if st.button("🚀 Process and Index Documents", type="primary"):
            for f in uploaded_files:
                progress_bar = st.progress(0, text=f"Starting ingestion for '{f.name}'...")
                
                # Step 1: Extract & Save
                progress_bar.progress(25, text=f"Extracting text from '{f.name}'...")
                time.sleep(0.15)
                
                # Step 2: Chunk
                progress_bar.progress(50, text="Generating overlapping semantic chunks...")
                time.sleep(0.15)

                # Step 3: Embed & Index
                progress_bar.progress(75, text="Computing dense vector embeddings with SentenceTransformers...")
                res = rag.ingest_document(f, f.name, uid=uid)
                
                progress_bar.progress(100, text=f"Done! Indexed {res.get('chunks', 0)} chunks across {res.get('pages', 0)} pages.")
                time.sleep(0.2)
                st.success(f"Indexed **{f.name}** ({res.get('chunks')} chunks).")
            
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # Document Library Listing
    summary = rag.get_indexed_summary(uid=uid)
    records = summary.get("records", [])

    st.markdown("### 📚 Indexed Documents")

    if not records:
        st.info("No documents uploaded yet. Drag and drop files above to start chatting with your private knowledge base.")
    else:
        st.caption(f"Total Chunks in FAISS Index: **{summary.get('total_chunks', 0)}**")
        
        for doc in records:
            col_info, col_del = st.columns([5, 1])
            with col_info:
                st.markdown(f"""
                <div style="background: rgba(15, 23, 42, 0.45); border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 12px; padding: 14px 18px; margin-bottom: 10px;">
                    <div style="font-weight: 700; color: #f8fafc; font-size: 1rem;">📄 {doc['filename']}</div>
                    <div style="font-size: 0.82rem; color: #94a3b8; margin-top: 4px;">
                        Pages: <b>{doc.get('pages', 1)}</b> &bull; Chunks: <b>{doc.get('chunks', 1)}</b> &bull; Uploaded: {doc.get('uploaded_at', '')[:19].replace('T', ' ')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with col_del:
                if st.button("🗑️ Delete", key=f"del_doc_{doc['doc_id']}", use_container_width=True):
                    rag.delete_single_document(uid, doc['doc_id'])
                    st.rerun()

        st.markdown("---")
        if st.button("⚠️ Clear All My Indexed Documents", type="secondary"):
            rag.clear_user_documents(uid)
            st.success("All personal documents and FAISS index cleared.")
            st.rerun()


# Execute
render_documents_page()
