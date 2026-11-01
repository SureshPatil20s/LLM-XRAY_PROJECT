"""
LocalGPT: ChatGPT-Style Local AI Assistant with RAG and LLM X-Ray
==================================================================
Project Phase 2 Continuation Upgrade

Features:
- Multi-turn ChatGPT-style conversational assistant
- Persistent conversation memory (SQLite)
- Conversation history sidebar (+ New Chat, rename, delete)
- Real-time streaming token generation (TextIteratorStreamer)
- Document RAG (PDF, DOCX, TXT) via FAISS and SentenceTransformers
- Source citation and verification links
- Message controls: Copy, Regenerate, Edit
- Integrated LLM X-Ray Inspection (Tokens, Embeddings PCA, Layers, Attention Heatmap, Hidden States, Logits)
- Full Project 1 Deep Mechanistic Lab & Phases 1-3 Claim Verification preserved
"""
import sys
import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional

import streamlit as st
import numpy as np
import pandas as pd
import torch
import plotly.graph_objects as go

# Core models and modules
from tokenizer import load_tokenizer, get_token_details
from model import (
    load_model,
    inspect_inference_internals,
    generate_step_by_step,
    analyze_computational_process,
    run_structured_explanation
)
from experiment_tracker import (
    capture_output_run,
    list_saved_captures,
    capture_phase2_explanation,
    list_phase2_explanations,
)
from visualization import (
    plot_embeddings_pca,
    plot_vector_profile,
    plot_attention_heatmap,
    plot_hidden_states_pca,
    plot_logits_bar_chart,
    plot_token_generation_timeline,
    plot_logit_lens_progression,
)

# Phase 2 LocalGPT & Phase 3 modules
import database
import document_loader
import embeddings
import vector_store
import rag
import memory
from memory import ConversationMemory
import xray
import chat
import claim_verifier

# Streamlit Page Config
st.set_page_config(
    page_title="SURESH.GPT — ChatGPT-Style AI with RAG & LLM X-Ray",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .localgpt-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .localgpt-sub {
        font-size: 0.95rem;
        color: #94a3b8;
        margin-bottom: 1.2rem;
    }
    .chat-bubble-assistant {
        background-color: #0f172a;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 8px;
    }
    .source-box {
        background-color: #1e293b;
        border-left: 4px solid #38bdf8;
        border-radius: 4px;
        padding: 8px 12px;
        margin-top: 10px;
        font-size: 0.85rem;
        color: #cbd5e1;
    }
    .action-btn-row {
        display: flex;
        gap: 8px;
        margin-top: 6px;
        margin-bottom: 16px;
    }
    .stButton>button {
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)


# Hardware Auto-Detection
has_cuda = torch.cuda.is_available()
hardware_device_name = torch.cuda.get_device_name(0) if has_cuda else "System CPU"
auto_model_index = 0 if has_cuda else 1


# Cached Model & Tokenizer Loaders
@st.cache_resource(show_spinner="Loading Hugging Face Tokenizer...")
def get_cached_tokenizer(name: str):
    return load_tokenizer(name)


@st.cache_resource(show_spinner="Loading Local LLM (First run downloads weights)...")
def get_cached_model(name: str):
    return load_model(name)


# Sidebar: App Mode Selector & Model Setup
st.sidebar.markdown("""
<div style="font-size:1.35rem; font-weight:800; color:#38bdf8; margin-bottom:4px;">
    🤖 SURESH.GPT
</div>
<div style="font-size:0.8rem; color:#94a3b8; margin-bottom:12px;">
    Local AI Assistant • RAG • LLM X-Ray
</div>
""", unsafe_allow_html=True)

nav_mode = st.sidebar.radio(
    "Navigation Mode",
    options=["💬 SURESH.GPT Assistant", "🔬 LLM X-Ray Laboratory & Claim Verifier"],
    index=0
)

# Model Selection
model_choice = st.sidebar.selectbox(
    "Model",
    options=[
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-0.5B-Instruct"
    ],
    index=auto_model_index,
    help="1.5B is recommended for accuracy; 0.5B is lightweight for CPU."
)
actual_model_name = model_choice

try:
    with st.spinner("Initializing LLM tokenizer..."):
        tokenizer = get_cached_tokenizer(actual_model_name)
    with st.spinner("Initializing LLM weights..."):
        loaded_model, device_str = get_cached_model(actual_model_name)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()


# ==============================================================================
# INTERFACE 1: LocalGPT ChatGPT-Style AI Assistant with RAG & LLM X-Ray
# ==============================================================================
def render_localgpt_interface():
    database.init_db()

    # Session state for conversations and X-Ray cache
    if "active_conv_id" not in st.session_state:
        all_convs = database.get_conversations()
        if all_convs:
            st.session_state["active_conv_id"] = all_convs[0]["id"]
        else:
            st.session_state["active_conv_id"] = database.create_conversation("New Chat")

    if "xray_cache" not in st.session_state:
        st.session_state["xray_cache"] = {}

    if "edit_mode_msg_id" not in st.session_state:
        st.session_state["edit_mode_msg_id"] = None

    conv_id = st.session_state["active_conv_id"]
    conv_meta = database.get_conversation(conv_id)
    if not conv_meta:
        conv_id = database.create_conversation("New Chat")
        st.session_state["active_conv_id"] = conv_id
        conv_meta = database.get_conversation(conv_id)

    memory_mgr = ConversationMemory(conv_id)

    # ── Sidebar: Conversation Controls & History ──────────────────────────────
    st.sidebar.markdown("---")
    col_new_chat, _ = st.sidebar.columns([3, 1])
    with col_new_chat:
        if st.sidebar.button("➕ New Chat", use_container_width=True, type="primary"):
            new_id = database.create_conversation("New Chat")
            st.session_state["active_conv_id"] = new_id
            st.session_state["edit_mode_msg_id"] = None
            st.rerun()

    # Recent Chats List
    st.sidebar.markdown("""
    <div style="font-size:0.95rem; font-weight:700; color:#cbd5e1; margin-top:10px; margin-bottom:6px;">
        📁 Recent Chats
    </div>
    """, unsafe_allow_html=True)

    conversations = database.get_conversations()
    for c in conversations:
        c_id = c["id"]
        is_active = (c_id == conv_id)
        btn_label = f"💬 {c['title']}" if not is_active else f"👉 {c['title']}"
        
        col_c_btn, col_c_del = st.sidebar.columns([4, 1])
        with col_c_btn:
            if st.button(btn_label, key=f"conv_sel_{c_id}", use_container_width=True):
                st.session_state["active_conv_id"] = c_id
                st.session_state["edit_mode_msg_id"] = None
                st.rerun()
        with col_c_del:
            if st.button("🗑️", key=f"conv_del_{c_id}", help="Delete chat"):
                database.delete_conversation(c_id)
                if c_id == conv_id:
                    remaining = database.get_conversations()
                    st.session_state["active_conv_id"] = remaining[0]["id"] if remaining else database.create_conversation("New Chat")
                st.rerun()

    # Rename active chat expander
    # Rename active chat expander (wrapped in st.form so Enter key saves immediately)
    with st.sidebar.expander("✏️ Rename Active Chat", expanded=False):
        new_title = st.text_input("New Title", value=conv_meta.get("title", ""), key="title_rename_input")
        if st.button("Save Title", key="save_title_btn"):
            if new_title.strip():
                database.update_conversation_title(conv_id, new_title.strip())
                st.rerun()
        with st.form(key=f"rename_chat_form_{conv_id}", clear_on_submit=False):
            new_title = st.text_input("New Title (Press Enter to Save)", value=conv_meta.get("title", ""), key=f"title_rename_input_{conv_id}")
            save_title_submitted = st.form_submit_button("💾 Save Title", use_container_width=True)
            if save_title_submitted:
                if new_title.strip():
                    database.update_conversation_title(conv_id, new_title.strip())
                    st.rerun()

    # ── Sidebar: Document Upload (RAG) ────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="font-size:0.95rem; font-weight:700; color:#38bdf8; margin-bottom:6px;">
        📄 Document Chat (RAG)
    </div>
    """, unsafe_allow_html=True)

    uploaded_doc = st.sidebar.file_uploader(
        "Upload PDF, TXT, or DOCX",
        type=["pdf", "txt", "docx"],
        help="Documents are chunked, embedded, and indexed into FAISS for context retrieval."
    )

    if uploaded_doc is not None:
        if st.sidebar.button("📥 Ingest Document", key="ingest_doc_btn", use_container_width=True):
            with st.spinner(f"Ingesting '{uploaded_doc.name}' into FAISS..."):
                ingest_res = rag.ingest_document(uploaded_doc, uploaded_doc.name)
                st.sidebar.success(f"Indexed {ingest_res.get('chunks', 0)} chunks from {ingest_res.get('pages', 0)} pages!")
                st.rerun()

    rag_summary = rag.get_indexed_summary()
    indexed_docs = rag_summary.get("documents", [])
    total_chunks = rag_summary.get("total_chunks", 0)

    if indexed_docs:
        st.sidebar.caption(f"📚 **Indexed:** {len(indexed_docs)} doc(s), **{total_chunks}** chunks.")
        for d in indexed_docs:
            st.sidebar.markdown(f"- 📄 `{d}`")
        if st.sidebar.button("🗑️ Clear Indexed Documents", key="clear_docs_btn"):
            rag.clear_all_documents()
            st.sidebar.info("FAISS document index cleared.")
            st.rerun()
    else:
        st.sidebar.caption("No documents currently indexed. Chat operates in standard generative mode.")

    # ── Sidebar: Settings & Generation Controls ──────────────────────────────
    st.sidebar.markdown("---")
    with st.sidebar.expander("⚙️ Assistant & Generation Settings", expanded=False):
        system_prompt = st.text_area(
            "System Instructions",
            value="You are a helpful AI assistant. Explain concepts clearly with examples.",
            help="Controls persona and behavior of the assistant."
        )
        use_rag_toggle = st.checkbox("Enable RAG context retrieval", value=True)
        enable_xray_toggle = st.checkbox("Enable LLM X-Ray Inspection", value=True)
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.5, value=0.7, step=0.05)
        top_k = st.slider("Top-K", min_value=1, max_value=100, value=50, step=5)
        top_p = st.slider("Top-P", min_value=0.1, max_value=1.0, value=0.9, step=0.05)
        max_tokens = st.slider("Max Tokens", min_value=30, max_value=1024, value=256, step=16)

    # ── Main Area: Chat Title & Header ───────────────────────────────────────
    st.markdown('<div class="localgpt-title">🤖 SURESH.GPT</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="localgpt-sub">Active Chat: <b style="color:#e2e8f0;">{conv_meta.get("title", "New Chat")}</b> • ChatGPT-Style Local AI Assistant with Document RAG &amp; LLM X-Ray</div>', unsafe_allow_html=True)

    # Fetch Conversation Message History
    history = memory_mgr.get_history()

    # Render Messages
    for msg in history:
        msg_id = msg["id"]
        role = msg["role"]
        content = msg["content"]
        sources = msg.get("sources", [])

        if role == "user":
            with st.chat_message("user"):
                st.markdown(content)
                # Message control: Edit
                col_e1, _ = st.columns([1, 10])
                with col_e1:
                    if st.button("✏️ Edit", key=f"edit_btn_{msg_id}", help="Edit this message and regenerate"):
                        st.session_state["edit_mode_msg_id"] = msg_id
                        st.session_state["edit_text"] = content
                        st.rerun()

        elif role == "assistant":
            with st.chat_message("assistant"):
                st.markdown(content)

                # Show document sources if present
                if sources:
                    st.markdown("""<div class="source-box"><b>📚 Sources:</b><br>""", unsafe_allow_html=True)
                    for s in sources:
                        doc_name = s.get("source", "Document")
                        page_num = s.get("page", 1)
                        snippet = s.get("snippet", "")
                        st.markdown(f"- 📄 **{doc_name}** — Page {page_num} *(excerpt: {snippet[:80]}...)*")
                    st.markdown("</div>", unsafe_allow_html=True)

                # Action Controls: Copy, Regenerate, X-Ray
                col_act1, col_act2, col_act3, _ = st.columns([1, 1.2, 1.2, 5])
                with col_act1:
                    if st.button("📋 Copy", key=f"copy_btn_{msg_id}"):
                        st.code(content, language="markdown")
                with col_act2:
                    if st.button("🔄 Regenerate", key=f"regen_btn_{msg_id}"):
                        # Delete this assistant response and re-trigger generation
                        memory_mgr.delete_from_message(msg_id, include_self=True)
                        # Find the preceding user message
                        updated_hist = memory_mgr.get_history()
                        if updated_hist and updated_hist[-1]["role"] == "user":
                            st.session_state["pending_user_query"] = updated_hist[-1]["content"]
                            # Also delete that user message so it is re-added cleanly in the stream runner
                            memory_mgr.delete_from_message(updated_hist[-1]["id"], include_self=True)
                        st.rerun()
                with col_act3:
                    if enable_xray_toggle:
                        is_xray_open = st.session_state.get(f"show_xray_{msg_id}", False)
                        btn_txt = "🔬 Close X-Ray" if is_xray_open else "🔍 X-Ray"
                        if st.button(btn_txt, key=f"xray_btn_{msg_id}"):
                            st.session_state[f"show_xray_{msg_id}"] = not is_xray_open
                            st.rerun()

                # Render Inline X-Ray Inspection Panel if toggled
                if st.session_state.get(f"show_xray_{msg_id}", False):
                    with st.container():
                        if msg_id not in st.session_state["xray_cache"]:
                            with st.spinner("Extracting internal activations and weights..."):
                                xray_data = xray.capture_xray(
                                    model=loaded_model,
                                    tokenizer=tokenizer,
                                    input_text=content[:250],
                                    device=device_str
                                )
                                st.session_state["xray_cache"][msg_id] = xray_data
                        else:
                            xray_data = st.session_state["xray_cache"][msg_id]

                        xray.render_xray_panel(xray_data, key_prefix=f"inline_{msg_id}")

    # Edit Message Modal / Expander if user clicked Edit
    # Edit Message Modal / Expander if user clicked Edit (wrapped in st.form so Enter key submits)
    if st.session_state["edit_mode_msg_id"]:
        edit_id = st.session_state["edit_mode_msg_id"]
        st.markdown("---")
        st.info("✏️ **Editing user message:** Submitting will regenerate the conversation from this point onwards.")
        new_msg_val = st.text_area("Edit your prompt:", value=st.session_state.get("edit_text", ""), key="edit_prompt_area")
        col_ed1, col_ed2 = st.columns([1, 5])
        with col_ed1:
            if st.button("Submit Edit & Regenerate", type="primary"):
                # Delete messages starting from edit_id
                memory_mgr.delete_from_message(edit_id, include_self=True)
                st.session_state["pending_user_query"] = new_msg_val
        with st.form(key=f"edit_form_{edit_id}", clear_on_submit=False):
            st.info("✏️ **Editing user message:** Press **Enter** or click Submit to regenerate from this prompt onwards.")
            new_msg_val = st.text_input("Edit your prompt (Press Enter to submit):", value=st.session_state.get("edit_text", ""), key=f"edit_prompt_input_{edit_id}")
            col_ed1, col_ed2 = st.columns([1, 4])
            with col_ed1:
                edit_submitted = st.form_submit_button("🚀 Submit & Regenerate", type="primary", use_container_width=True)
            with col_ed2:
                edit_cancelled = st.form_submit_button("Cancel Edit", use_container_width=True)

            if edit_submitted:
                if new_msg_val.strip():
                    memory_mgr.delete_from_message(edit_id, include_self=True)
                    st.session_state["pending_user_query"] = new_msg_val.strip()
                    st.session_state["edit_mode_msg_id"] = None
                    st.rerun()
            elif edit_cancelled:
                st.session_state["edit_mode_msg_id"] = None
                st.rerun()
        with col_ed2:
            if st.button("Cancel Edit"):
                st.session_state["edit_mode_msg_id"] = None
                st.rerun()

    # Handle pending query (from edit or regenerate)
    user_query = None
    if "pending_user_query" in st.session_state and st.session_state["pending_user_query"]:
        user_query = st.session_state.pop("pending_user_query")

    # Bottom Chat Input
    chat_input_val = st.chat_input("Ask anything... (or ask about your uploaded documents)")
    if chat_input_val:
        user_query = chat_input_val

    # Execute Streaming Response if user submitted a query
    if user_query:
        # Display User Turn
        with st.chat_message("user"):
            st.markdown(user_query)

        # Stream Assistant Turn
        with st.chat_message("assistant"):
            stream_gen, retrieved_sources = chat.stream_chat_response(
                model=loaded_model,
                tokenizer=tokenizer,
                memory=memory_mgr,
                user_query=user_query,
                system_prompt=system_prompt,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                max_new_tokens=max_tokens,
                use_rag=use_rag_toggle,
                device=device_str
            )
            full_response = st.write_stream(stream_gen)

            # Display sources if retrieved
            if retrieved_sources:
                st.markdown("""<div class="source-box"><b>📚 Sources:</b><br>""", unsafe_allow_html=True)
                for s in retrieved_sources:
                    st.markdown(f"- 📄 **{s['source']}** — Page {s['page']} *(excerpt: {s['snippet'][:80]}...)*")
                st.markdown("</div>", unsafe_allow_html=True)

            # Persist assistant response to database & memory
            asst_msg_id = memory_mgr.add_assistant_message(
                content=full_response,
                sources=retrieved_sources
            )

            # Pre-compute X-Ray cache if enabled
            if enable_xray_toggle:
                try:
                    xray_data = xray.capture_xray(
                        model=loaded_model,
                        tokenizer=tokenizer,
                        input_text=full_response[:250],
                        device=device_str
                    )
                    st.session_state["xray_cache"][asst_msg_id] = xray_data
                except Exception:
                    pass

        st.rerun()


# ==============================================================================
# INTERFACE 2: LLM X-Ray Laboratory & Phases 1-3 Claim Verification
# ==============================================================================
def render_xray_lab_interface():
    st.markdown('<div class="localgpt-title">🔬 LLM X-RAY LABORATORY</div>', unsafe_allow_html=True)
    st.markdown('<div class="localgpt-sub">Deep Mechanistic Interpretability Lab • Autoregressive Diagnostics • Phases 1, 2, and 3 Verification</div>', unsafe_allow_html=True)

    col_inp1, col_inp2 = st.columns([3, 1])
    with col_inp1:
        prompt = st.text_input("Enter prompt for mechanistic inspection:", value="What is AI?", key="lab_prompt_input")
    with col_inp2:
        max_lab_tokens = st.slider("Budget Tokens", min_value=5, max_value=60, value=25, key="lab_max_tokens")
    with st.form(key="lab_prompt_form", clear_on_submit=False):
        col_inp1, col_inp2 = st.columns([3, 1])
        with col_inp1:
            prompt = st.text_input("Enter prompt for mechanistic inspection (Press Enter to Run):", value="What is AI?", key="lab_prompt_input")
        with col_inp2:
            max_lab_tokens = st.slider("Budget Tokens", min_value=5, max_value=60, value=25, key="lab_max_tokens")

    run_lab_btn = st.button("🚀 Run Deep Lab Forward Pass", type="primary", use_container_width=True)
    run_lab_btn = st.form_submit_button("🚀 Run Deep Lab Forward Pass", type="primary", use_container_width=True)

    if run_lab_btn or "lab_results" not in st.session_state:
        with st.spinner("Executing Transformer forward pass and capturing all intermediate tensors..."):
            tok_details = get_token_details(prompt, tokenizer, use_chat_template=True)
            internals = inspect_inference_internals(loaded_model, tokenizer, prompt, device=device_str, use_chat_template=True)
            generated_text, step_history, usage_stats = generate_step_by_step(
                model=loaded_model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=max_lab_tokens,
                temperature=0.0,
                top_k=50,
                top_p=0.9,
                use_chat_template=True,
                device=device_str
            )
            process_audit = analyze_computational_process(prompt, generated_text, step_history, usage_stats)
            phase1_capture = capture_output_run(
                prompt=prompt,
                response=generated_text,
                model_name=actual_model_name,
                model=loaded_model,
                formatted_prompt=tok_details.get("formatted_text"),
                max_new_tokens=max_lab_tokens,
                prompt_tokens=tok_details.get("total_tokens", len(tok_details.get("tokens", []))),
                generated_tokens=len(step_history),
                temperature=0.0,
                top_k=50,
                top_p=0.9,
                use_chat_template=True,
                usage_stats=usage_stats,
                step_history=step_history,
                device=device_str,
                save_to_disk=True
            )
            st.session_state["lab_results"] = {
                "prompt": prompt,
                "tok_details": tok_details,
                "internals": internals,
                "generated_text": generated_text,
                "step_history": step_history,
                "usage_stats": usage_stats,
                "process_audit": process_audit,
                "phase1_capture": phase1_capture
            }

    res = st.session_state["lab_results"]
    tok_details = res["tok_details"]
    internals = res["internals"]
    generated_text = res["generated_text"]
    step_history = res["step_history"]
    process_audit = res["process_audit"]
    phase1_capture = res["phase1_capture"]

    tab_overview, tab_tokens, tab_embeddings, tab_layers, tab_attention, tab_hidden, tab_logits, tab_generation, tab_phase1, tab_phase2, tab_phase3 = st.tabs([
        "💬 Model Overview",
        "🔤 Tokenization",
        "🧬 Embeddings",
        "📐 Layers",
        "🔍 Attention",
        "🧠 Hidden States",
        "📊 Logits",
        "⏩ Generation Flow",
        "📁 Phase 1: Output Capture",
        "📑 Phase 2: Explanation",
        "🔍 Phase 3: Claim Verification"
    ])

    # 1. Overview
    with tab_overview:
        col_ov1, col_ov2 = st.columns([3, 2])
        with col_ov1:
            st.markdown("**Prompt:**")
            st.info(res["prompt"])
            st.markdown("**Generated Response:**")
            st.success(generated_text)
        with col_ov2:
            st.markdown("### 📋 Inference Summary")
            st.markdown(f"""
            - **Model:** `{actual_model_name}`
            - **Layers:** `{internals['num_layers']}`
            - **Attention Heads:** `{internals['num_heads']}`
            - **Embedding Dimension:** `{internals['hidden_dim']}`
            - **Tokens Generated:** `{len(step_history)}`
            """)

    # 2. Tokenization
    with tab_tokens:
        st.markdown(f"**Token Sequence:** `{tok_details['display_tokens_str']}`")
        st.dataframe(pd.DataFrame(tok_details.get("token_table", [])), use_container_width=True)

    # 3. Embeddings
    with tab_embeddings:
        fig_pca = plot_embeddings_pca(internals["input_embeddings"], tok_details["tokens"], tok_details["token_ids_list"])
        st.plotly_chart(fig_pca, use_container_width=True)

    # 4. Layers
    with tab_layers:
        st.metric("Total Layers", internals["num_layers"])
        st.metric("Hidden Size", internals["hidden_dim"])

    # 5. Attention
    with tab_attention:
        attns = internals.get("attentions", [])
        if attns:
            sel_l = st.slider("Layer", 0, len(attns) - 1, 0, key="lab_sel_l")
            sel_h = st.slider("Head", 0, attns[0].shape[1] - 1, 0, key="lab_sel_h")
            fig_attn = plot_attention_heatmap(attns[sel_l][0, sel_h].numpy(), tok_details["tokens"], sel_l, sel_h)
            st.plotly_chart(fig_attn, use_container_width=True)

    # 6. Hidden States
    with tab_hidden:
        hids = internals.get("hidden_states", [])
        if hids:
            fig_hid = plot_hidden_states_pca(hids, tok_details["tokens"], [0, len(hids)//2, len(hids)-1])
            st.plotly_chart(fig_hid, use_container_width=True)

    # 7. Logits
    with tab_logits:
        top_preds = internals.get("top_predictions", [])
        if top_preds:
            st.plotly_chart(plot_logits_bar_chart(top_preds), use_container_width=True)
            st.dataframe(pd.DataFrame(top_preds), use_container_width=True)

    # 8. Generation Flow
    with tab_generation:
        st.plotly_chart(plot_token_generation_timeline(step_history), use_container_width=True)

    # 9. Phase 1 Capture
    with tab_phase1:
        st.markdown("### 📁 Phase 1: Complete Output Capture")
        st.json(phase1_capture)

    # 10. Phase 2 Structured Explanation
    with tab_phase2:
        st.markdown("### 📑 Phase 2: Structured Explanation Interrogation")
        if st.button("Interrogate Model for Structured Explanation", key="p2_interrogate_btn"):
            with st.spinner("Extracting concise steps, assumptions, claims and evidence..."):
                p2_res = run_structured_explanation(loaded_model, tokenizer, res["prompt"], generated_text, device=device_str)
                st.session_state["p2_res"] = p2_res
        if "p2_res" in st.session_state:
            st.json(st.session_state["p2_res"])

    # 11. Phase 3 Claim Verification
    with tab_phase3:
        st.markdown("### 🔍 Phase 3: Claim Verification Pipeline")
        st.caption(
            "Extracts factual claims from the original answer and evaluates them against "
            "deterministic calculation engines, curated knowledge bases, and encyclopedic retrieval. "
            "Zero citations are fabricated; model agreement is not counted as independent evidence."
        )

        if st.button("🚀 Verify Output Claims", key="p3_run_verify_btn", type="primary"):
            with st.spinner("Extracting and verifying claims..."):
                p3_res = claim_verifier.execute_phase3_verification(
                    original_answer=generated_text,
                    original_run_id=phase1_capture["run_id"],
                    prompt=res["prompt"],
                    is_truncated=phase1_capture["response"]["is_truncated"]
                )
                st.session_state["p3_res"] = p3_res

        if "p3_res" in st.session_state:
            p3_data = st.session_state["p3_res"]
            summary = p3_data["status_summary"]

            # Summary Metrics
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Supported", summary.get("SUPPORTED", 0))
            c2.metric("Contradicted", summary.get("CONTRADICTED", 0))
            c3.metric("Partially Supported", summary.get("PARTIALLY_SUPPORTED", 0))
            c4.metric("Insufficient Evidence", summary.get("INSUFFICIENT_EVIDENCE", 0))
            c5.metric("Not Verifiable", summary.get("NOT_VERIFIABLE", 0))

            st.markdown("---")
            st.markdown("#### Extracted Claims Breakdown")
            for c in p3_data["claims"]:
                status = c["verification_status"]
                badge_color = {
                    "SUPPORTED": "#22c55e",
                    "CONTRADICTED": "#ef4444",
                    "PARTIALLY_SUPPORTED": "#f59e0b",
                    "INSUFFICIENT_EVIDENCE": "#64748b",
                    "NOT_VERIFIABLE": "#8b5cf6",
                    "PENDING": "#94a3b8"
                }.get(status, "#94a3b8")

                with st.expander(f"[{status}] {c['claim_id']}: {c['exact_claim_text']}", expanded=True):
                    st.markdown(f"""
                    <div style="background-color:#0f172a; border-left:5px solid {badge_color}; border:1px solid #334155; border-left-width:5px; border-radius:6px; padding:12px;">
                        <span style="font-weight:700; color:#38bdf8;">Type:</span> {c['claim_type']} &nbsp;|&nbsp;
                        <span style="font-weight:700; color:{badge_color};">Status: {status}</span><br>
                        <div style="margin-top:6px;"><b>Explanation:</b> {c['explanation']}</div>
                        <div style="margin-top:4px; font-size:0.85rem; color:#94a3b8;">
                            <b>Method:</b> {c['verification_method']}<br>
                            <b>Source:</b> {c['source_title']} {f"(<a href='{c['source_url']}' target='_blank' style='color:#38bdf8;'>link</a>)" if c.get('source_url') else ""}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            st.download_button(
                "📥 Download Phase 3 Verification Report (JSON)",
                data=json.dumps(p3_data, indent=2, ensure_ascii=False),
                file_name=f"phase3_{p3_data['run_id']}.json",
                mime="application/json"
            )


# Dispatch active interface mode
if nav_mode == "💬 SURESH.GPT Assistant":
    render_localgpt_interface()
else:
    render_xray_lab_interface()
