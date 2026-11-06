"""
Workspace Chat Screen for SURESH.GPT Platform / LocalGPT Pro.
Matches the ChatGPT Platform dark interface:
- Top navbar with brand, model pill badge (⚡ Qwen 2.5 0.5B — Local (No API Key)), + New Chat, settings, user & logout
- Welcome screen: "Hello! How can I assist you today?" with 4 suggested prompt cards
- Multi-turn ChatGPT-style conversation with right-aligned user messages and left-aligned assistant turns
- Inline action icons below messages: Copy, Regenerate, Thumbs Up/Down, Edit, and X-Ray Drawer
- Integrated Mechanistic Interpretability X-Ray drawer for every assistant turn
- Document RAG toggle ("RAG Active") with source provenance chips
- Floating pill chat input dock with disclaimer
"""
import streamlit as st
import json
import torch
from typing import Dict, Any, Optional

import database
from memory import ConversationMemory
import chat
import xray
from export import export_chat_to_markdown, export_chat_to_pdf
from ui.components import render_welcome_screen, render_source_chips
from ui.theme import apply_theme
from ui.sidebar import render_sidebar
from auth.guards import require_login
from tokenizer import load_tokenizer
from model import load_model

# Page Setup
st.set_page_config(
    page_title="SURESH.GPT Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

database.init_db()

# Enforce login guard
user = require_login()
uid = user["uid"]

# Apply ChatGPT Platform dark theme
active_theme = st.session_state.get("theme", user.get("theme", "dark"))
apply_theme(active_theme)

# Render Sidebar
render_sidebar(user, active_conv_id=st.session_state.get("active_conv_id"), current_page="workspace")

# Load Cached Model & Tokenizer
has_cuda = torch.cuda.is_available()
model_choice = "Qwen/Qwen2.5-0.5B-Instruct" if not has_cuda else "Qwen/Qwen2.5-1.5B-Instruct"
model_short_name = "Qwen 2.5 0.5B" if "0.5B" in model_choice else "Qwen 2.5 1.5B"

@st.cache_resource(show_spinner="Loading Hugging Face Tokenizer...")
def get_cached_tokenizer(name: str):
    return load_tokenizer(name)

@st.cache_resource(show_spinner="Loading Local LLM (First run downloads weights)...")
def get_cached_model(name: str):
    return load_model(name)

try:
    with st.spinner("Initializing LLM tokenizer..."):
        tokenizer = get_cached_tokenizer(model_choice)
    with st.spinner("Initializing LLM weights..."):
        loaded_model, device_str = get_cached_model(model_choice)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()


def render_workspace_page(loaded_model: Any, tokenizer: Any, device_str: str):
    user_settings = database.get_user_settings(uid)

    # Manage active conversation
    if "active_conv_id" not in st.session_state or not st.session_state["active_conv_id"]:
        user_chats = database.get_conversations(uid=uid)
        if user_chats:
            st.session_state["active_conv_id"] = user_chats[0]["id"]
        else:
            st.session_state["active_conv_id"] = database.create_conversation("New Chat", uid=uid)

    conv_id = st.session_state["active_conv_id"]
    conv_meta = database.get_conversation(conv_id, uid=uid)
    if not conv_meta:
        conv_id = database.create_conversation("New Chat", uid=uid)
        st.session_state["active_conv_id"] = conv_id
        conv_meta = database.get_conversation(conv_id, uid=uid)

    memory_mgr = ConversationMemory(conv_id, uid=uid)

    if "xray_cache" not in st.session_state:
        st.session_state["xray_cache"] = {}

    if "edit_mode_msg_id" not in st.session_state:
        st.session_state["edit_mode_msg_id"] = None

    # =========================================================================
    # TOP NAVBAR (Matches reference screenshots 2 & 3)
    # Left: Robot + SURESH.GPT Platform
    # Center: Model badge capsule: ⚡ Qwen 2.5 0.5B — Local (No API Key) ⌵
    # Right: + New Chat (green), ⚙️ Settings, 👤 User name, 🚪 Logout
    # =========================================================================
    col_nav1, col_nav2, col_nav3 = st.columns([2.8, 3.8, 4.2])

    with col_nav1:
        st.markdown("""
        <div style="display:flex; align-items:center; gap:8px; padding-top:2px;">
            <div style="background:#10a37f; width:28px; height:28px; border-radius:6px; display:flex; align-items:center; justify-content:center; font-size:16px;">
                🤖
            </div>
            <span style="font-weight:700; font-size:1.15rem; color:#f4f4f5; letter-spacing:-0.02em;">
                SURESH.GPT Platform
            </span>
        </div>
        """, unsafe_allow_html=True)

    with col_nav2:
        st.markdown(f"""
        <div style="display:inline-flex; align-items:center; gap:8px; background:#202123; border:1px solid #383a42; border-radius:18px; padding:5px 14px; font-size:0.82rem; color:#ececec;">
            <span style="color:#eab308; font-size:0.95rem;">⚡</span>
            <span style="font-weight:500;">{model_short_name} — Local (No API Key)</span>
            <span style="color:#71717a; font-size:0.7rem;">⌵</span>
        </div>
        """, unsafe_allow_html=True)

    with col_nav3:
        col_nc, col_rag, col_st, col_us, col_lo = st.columns([1.5, 1.4, 0.7, 1.3, 0.7])
        with col_nc:
            if st.button("＋ New Chat", key="top_new_chat_btn", type="primary", use_container_width=True):
                new_id = database.create_conversation("New Chat", uid=uid)
                st.session_state["active_conv_id"] = new_id
                st.session_state["edit_mode_msg_id"] = None
                st.rerun()

        with col_rag:
            use_rag = st.checkbox("RAG Active", value=True, key=f"top_rag_{conv_id}", help="Retrieve context from uploaded documents")

        with col_st:
            if st.button("⚙️", key="top_settings_btn", help="Settings & Profile"):
                try:
                    st.switch_page("pages/profile.py")
                except Exception:
                    pass

        with col_us:
            u_name = user.get("display_name", "User")
            display_name = u_name if len(u_name) <= 10 else u_name[:9] + "…"
            st.markdown(f"""
            <div style="background:#202123; border:1px solid #383a42; border-radius:8px; padding:6px 6px; text-align:center; font-size:0.78rem; color:#e4e4e7; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
                👤 {display_name}
            </div>
            """, unsafe_allow_html=True)

        with col_lo:
            if st.button("⎋", key="top_logout_btn", help="Sign Out"):
                from auth.auth import logout
                logout()
                try:
                    st.switch_page("pages/login.py")
                except Exception:
                    st.rerun()

    # Subtle divider below top navbar
    st.markdown("<div style='border-bottom: 1px solid #23252c; margin: 8px 0 16px 0;'></div>", unsafe_allow_html=True)

    # Optional Collapsible Parameters & Export Toolbar
    with st.expander("🎛 Chat Parameters & 📥 Export", expanded=False):
        col_p1, col_p2, col_p3 = st.columns([2.5, 1, 1])
        with col_p1:
            system_instruction = st.text_input(
                "System Instructions",
                value=user.get("default_system_prompt") or "You are a helpful AI assistant. Explain concepts clearly with examples.",
                key=f"sys_input_{conv_id}"
            )
        history_for_export = memory_mgr.get_history()
        with col_p2:
            if st.button("📥 Export .MD", key=f"exp_md_{conv_id}", use_container_width=True):
                md_path = export_chat_to_markdown(conv_meta, history_for_export, uid)
                with open(md_path, "r", encoding="utf-8") as f:
                    md_data = f.read()
                st.download_button(
                    "Download .md",
                    data=md_data,
                    file_name=f"{conv_meta.get('title', 'chat')}.md",
                    mime="text/markdown",
                    key=f"dl_md_{conv_id}"
                )
        with col_p3:
            if st.button("📥 Export .PDF", key=f"exp_pdf_{conv_id}", use_container_width=True):
                pdf_path = export_chat_to_pdf(conv_meta, history_for_export, uid)
                with open(pdf_path, "rb") as f:
                    pdf_data = f.read()
                st.download_button(
                    "Download .pdf",
                    data=pdf_data,
                    file_name=f"{conv_meta.get('title', 'chat')}.pdf",
                    mime="application/pdf",
                    key=f"dl_pdf_{conv_id}"
                )

    # Fetch History
    history = memory_mgr.get_history()

    # Empty State: Welcome Screen with Prompt Cards
    prompt_from_card = None
    if not history:
        prompt_from_card = render_welcome_screen(user.get("display_name", "Friend"))

    # Render Chat History
    for msg in history:
        msg_id = msg["id"]
        role = msg["role"]
        content = msg["content"]
        sources = msg.get("sources", [])

        if role == "user":
            # User Message Turn: Right-aligned bubble + avatar
            with st.chat_message("user", avatar=user.get("photo_url")):
                st.markdown(content)

                # Action buttons below user message (Copy, Edit)
                col_u1, col_u2, _ = st.columns([0.6, 0.6, 6.0])
                with col_u1:
                    if st.button("📋", key=f"cp_u_{msg_id}", help="Copy message"):
                        st.code(content, language="markdown")
                with col_u2:
                    if st.button("✏️", key=f"edit_btn_{msg_id}", help="Edit prompt"):
                        st.session_state["edit_mode_msg_id"] = msg_id
                        st.session_state["edit_text"] = content
                        st.rerun()

        elif role == "assistant":
            # Assistant Message Turn: Left-aligned green robot avatar + bubble
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(content)

                if sources:
                    render_source_chips(sources)

                # Inline action icons below assistant response (Copy, Regenerate, Thumbs, X-Ray)
                col_a1, col_a2, col_a3, col_a4, col_a5, _ = st.columns([0.6, 0.6, 0.6, 0.6, 1.8, 3.5])
                with col_a1:
                    if st.button("📋", key=f"cp_{msg_id}", help="Copy response"):
                        st.code(content, language="markdown")
                with col_a2:
                    if st.button("🔄", key=f"rg_{msg_id}", help="Regenerate response"):
                        memory_mgr.delete_from_message(msg_id, include_self=True)
                        remaining_hist = memory_mgr.get_history()
                        if remaining_hist and remaining_hist[-1]["role"] == "user":
                            st.session_state["pending_user_query"] = remaining_hist[-1]["content"]
                            memory_mgr.delete_from_message(remaining_hist[-1]["id"], include_self=True)
                        st.rerun()
                with col_a3:
                    if st.button("👍", key=f"th_up_{msg_id}", help="Good response"):
                        st.toast("Feedback recorded!")
                with col_a4:
                    if st.button("👎", key=f"th_dn_{msg_id}", help="Bad response"):
                        st.toast("Feedback recorded!")
                with col_a5:
                    is_xray_open = st.session_state.get(f"show_xray_{msg_id}", False)
                    btn_label = "🔬 Close Drawer" if is_xray_open else "🔬 X-Ray Drawer"
                    if st.button(btn_label, key=f"xray_btn_{msg_id}", help="Inspect neural activations"):
                        st.session_state[f"show_xray_{msg_id}"] = not is_xray_open
                        st.rerun()

                # Inline Mechanistic Interpretability X-Ray Drawer
                if st.session_state.get(f"show_xray_{msg_id}", False):
                    with st.expander("🔬 LLM X-Ray Mechanistic Diagnostics", expanded=True):
                        if msg_id not in st.session_state["xray_cache"]:
                            with st.spinner("Extracting internal layer activations..."):
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

    # Edit Message Form if user clicked edit
    if st.session_state.get("edit_mode_msg_id"):
        edit_id = st.session_state["edit_mode_msg_id"]
        st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
        with st.form(key=f"edit_prompt_form_{edit_id}"):
            st.info("✏️ **Editing message:** Submitting will regenerate the response from this point onward.")
            new_text = st.text_input("Edit prompt:", value=st.session_state.get("edit_text", ""))
            col_sb1, col_sb2 = st.columns([1.2, 4])
            with col_sb1:
                edit_submitted = st.form_submit_button("🚀 Submit & Regenerate", type="primary")
            with col_sb2:
                cancel_edit = st.form_submit_button("Cancel")

            if edit_submitted and new_text.strip():
                memory_mgr.delete_from_message(edit_id, include_self=True)
                st.session_state["pending_user_query"] = new_text.strip()
                st.session_state["edit_mode_msg_id"] = None
                st.rerun()
            elif cancel_edit:
                st.session_state["edit_mode_msg_id"] = None
                st.rerun()

    # Query Input Handling
    user_query = prompt_from_card
    if "pending_user_query" in st.session_state and st.session_state["pending_user_query"]:
        user_query = st.session_state.pop("pending_user_query")

    chat_input_val = st.chat_input("Message SURESH.GPT or ask questions about your documents...")
    if chat_input_val:
        user_query = chat_input_val

    # Bottom Disclaimer Subtitle (matching screenshots 2 & 3)
    st.markdown("""
    <div style="text-align: center; font-size: 0.74rem; color: #71717a; margin-top: 6px; margin-bottom: 2px;">
        SURESH.GPT Platform can make mistakes. Verify critical facts and inspect document citations.
    </div>
    """, unsafe_allow_html=True)

    # Stream Assistant Turn
    if user_query:
        with st.chat_message("user", avatar=user.get("photo_url")):
            st.markdown(user_query)

        with st.chat_message("assistant", avatar="🤖"):
            stream_gen, retrieved_sources = chat.stream_chat_response(
                model=loaded_model,
                tokenizer=tokenizer,
                memory=memory_mgr,
                user_query=user_query,
                system_prompt=system_instruction if "system_instruction" in locals() else "You are a helpful AI assistant.",
                temperature=float(user_settings.get("temperature", 0.7)),
                top_k=int(user_settings.get("top_k", 50)),
                top_p=float(user_settings.get("top_p", 0.9)),
                max_new_tokens=int(user_settings.get("max_tokens", 256)),
                use_rag=use_rag,
                device=device_str,
                uid=uid
            )

            full_response = st.write_stream(stream_gen)

            if retrieved_sources:
                render_source_chips(retrieved_sources)

            asst_msg_id = memory_mgr.add_assistant_message(
                content=full_response,
                sources=retrieved_sources
            )

        st.rerun()


# Render workspace page
render_workspace_page(loaded_model, tokenizer, device_str)
