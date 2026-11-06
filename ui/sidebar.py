"""
Sidebar component for SURESH.GPT Platform / LocalGPT Pro.
Provides seamless, reliable multi-page navigation across:
- Chat Workspace (pages/workspace.py)
- Document RAG Library (pages/documents.py)
- LLM X-Ray Lab & Claim Verifier (pages/xray_lab.py)
- User Profile & Settings (pages/profile.py)
"""
import streamlit as st
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
import database
from auth.auth import logout


def group_conversations_by_date(conversations: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Groups conversations into Pinned, Today, Yesterday, Previous 7 Days, and Older."""
    now = datetime.now(timezone.utc)
    today_start = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
    yesterday_start = today_start - timedelta(days=1)
    seven_days_ago = today_start - timedelta(days=7)

    groups = {
        "Pinned": [],
        "Today": [],
        "Yesterday": [],
        "Previous 7 Days": [],
        "Older": []
    }

    for c in conversations:
        if c.get("pinned"):
            groups["Pinned"].append(c)
            continue

        updated_str = c.get("updated_at") or c.get("created_at")
        try:
            dt = datetime.fromisoformat(str(updated_str).replace("Z", "+00:00"))
            if dt >= today_start:
                groups["Today"].append(c)
            elif dt >= yesterday_start:
                groups["Yesterday"].append(c)
            elif dt >= seven_days_ago:
                groups["Previous 7 Days"].append(c)
            else:
                groups["Older"].append(c)
        except Exception:
            groups["Older"].append(c)

    return groups


def render_sidebar(user: Dict[str, Any], active_conv_id: Optional[str] = None, current_page: str = "workspace") -> Dict[str, Any]:
    """
    Renders the ChatGPT Platform sidebar with persistent navigation routes.
    Clicking any chat or new chat smoothly switches to the Workspace.
    """
    uid = user["uid"]
    action_result = {"selected_conv_id": active_conv_id}

    # Brand Title Header
    st.sidebar.markdown("""
    <div style="display:flex; align-items:center; gap:10px; padding: 4px 2px 10px 2px;">
        <div style="background:#10a37f; width:32px; height:32px; border-radius:8px; display:flex; align-items:center; justify-content:center; font-size:18px;">
            🤖
        </div>
        <div>
            <div style="font-weight:700; font-size:1.12rem; color:#f4f4f5; line-height:1.2;">SURESH.GPT</div>
        </div>
        <div style="margin-left:auto; background:rgba(16,185,129,0.15); color:#10b981; border:1px solid #10b981; font-size:0.62rem; font-weight:700; padding:2px 7px; border-radius:12px; letter-spacing:0.04em;">
            LOCAL LLM
        </div>
    </div>
    """, unsafe_allow_html=True)

    # =========================================================================
    # NAVIGATION HUB: 3 direct page buttons
    # =========================================================================
    col_nav_w, col_nav_d, col_nav_x = st.sidebar.columns(3)
    with col_nav_w:
        w_type = "primary" if current_page == "workspace" else "secondary"
        if st.button("💬 Chat", key="sb_nav_chat_btn", use_container_width=True, type=w_type, help="Go to Chat Workspace"):
            try:
                st.switch_page("pages/workspace.py")
            except Exception:
                st.rerun()

    with col_nav_d:
        d_type = "primary" if current_page == "documents" else "secondary"
        if st.button("📁 RAG", key="sb_nav_docs_btn", use_container_width=True, type=d_type, help="Go to Document Manager"):
            try:
                st.switch_page("pages/documents.py")
            except Exception:
                st.rerun()

    with col_nav_x:
        x_type = "primary" if current_page == "xray_lab" else "secondary"
        if st.button("🔬 Lab", key="sb_nav_xray_btn", use_container_width=True, type=x_type, help="Go to LLM X-Ray Lab"):
            try:
                st.switch_page("pages/xray_lab.py")
            except Exception:
                st.rerun()

    st.sidebar.markdown("<div style='margin-top: 6px;'></div>", unsafe_allow_html=True)

    # + New Chat Button with Ctrl+K shortcut hint
    col_nc1, col_nc2 = st.sidebar.columns([4, 1.3])
    with col_nc1:
        new_chat_clicked = st.button("＋ New Chat", key="sb_new_chat_main", use_container_width=True)
    with col_nc2:
        st.markdown("""
        <div style="background:#1e1f26; border:1px solid #2e3038; border-radius:8px; text-align:center; padding:7px 0; font-size:0.75rem; color:#8e8ea0; font-family:'JetBrains Mono',monospace;">
            Ctrl+K
        </div>
        """, unsafe_allow_html=True)

    if new_chat_clicked:
        new_id = database.create_conversation("New Chat", uid=uid)
        st.session_state["active_conv_id"] = new_id
        st.session_state["edit_mode_msg_id"] = None
        try:
            st.switch_page("pages/workspace.py")
        except Exception:
            st.rerun()

    # Chat Search Input
    search_query = st.sidebar.text_input(
        "Search chats",
        placeholder="🔍 Search chats...",
        label_visibility="collapsed",
        key="sidebar_search_input"
    ).strip().lower()

    # Fetch User's Conversations
    conversations = database.get_conversations(uid=uid)
    if search_query:
        conversations = [c for c in conversations if search_query in c.get("title", "").lower()]

    # Section Label: RECENT CONVERSATIONS
    st.sidebar.markdown("""
    <div style="font-size:0.72rem; font-weight:700; color:#71717a; text-transform:uppercase; letter-spacing:0.06em; margin-top:12px; margin-bottom:6px; padding-left:4px;">
        RECENT CONVERSATIONS
    </div>
    """, unsafe_allow_html=True)

    if not conversations:
        st.sidebar.markdown("""
        <div style="font-size:0.8rem; color:#71717a; padding:8px 4px;">
            No conversations yet. Start a new chat above!
        </div>
        """, unsafe_allow_html=True)
    else:
        for c in conversations:
            cid = c["id"]
            is_active = (cid == active_conv_id and current_page == "workspace")
            title = c.get("title", "Untitled Chat")
            pinned = bool(c.get("pinned", 0))

            display_title = title if len(title) <= 20 else title[:18] + "..."
            icon_prefix = "📌 " if pinned else "💬 "

            col_btn, col_pin, col_del = st.sidebar.columns([5.5, 1.2, 1.2])
            with col_btn:
                btn_type = "primary" if is_active else "secondary"
                if st.button(f"{icon_prefix}{display_title}", key=f"c_btn_{cid}", use_container_width=True, type=btn_type):
                    st.session_state["active_conv_id"] = cid
                    st.session_state["edit_mode_msg_id"] = None
                    try:
                        st.switch_page("pages/workspace.py")
                    except Exception:
                        st.rerun()

            with col_pin:
                pin_icon = "📍" if pinned else "📌"
                if st.button(pin_icon, key=f"c_pin_{cid}", help="Pin / Unpin chat"):
                    database.toggle_pin_conversation(cid)
                    st.rerun()

            with col_del:
                if st.button("🗑️", key=f"c_del_{cid}", help="Delete chat"):
                    database.delete_conversation(cid, uid=uid)
                    if cid == active_conv_id:
                        remaining = database.get_conversations(uid=uid)
                        st.session_state["active_conv_id"] = remaining[0]["id"] if remaining else database.create_conversation("New Chat", uid=uid)
                    st.rerun()

    st.sidebar.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

    # Document Manager (RAG) Pill Button at bottom of list
    col_doc_btn, col_doc_badge = st.sidebar.columns([3.8, 2.2])
    with col_doc_btn:
        if st.button("📄 Document Manager", key="sb_doc_mgr_btn", use_container_width=True, help="Manage private RAG documents"):
            try:
                st.switch_page("pages/documents.py")
            except Exception:
                st.rerun()
    with col_doc_badge:
        st.markdown("""
        <div style="background:rgba(16,185,129,0.18); border:1px solid #10b981; border-radius:6px; text-align:center; padding:7px 0; font-size:0.68rem; font-weight:700; color:#10b981; letter-spacing:0.04em;">
            PDF/DOCX
        </div>
        """, unsafe_allow_html=True)

    # X-Ray Lab link
    if st.sidebar.button("🔬 LLM X-Ray Lab & Inspector", key="sb_xray_lab_btn", use_container_width=True):
        try:
            st.switch_page("pages/xray_lab.py")
        except Exception:
            st.rerun()

    st.sidebar.markdown("<div style='margin-top: 14px; border-top: 1px solid #22242a; padding-top: 10px;'></div>", unsafe_allow_html=True)

    # User Profile Card at bottom
    user_name = user.get("display_name", "Local User")
    initial = (user_name[:1] if user_name else "U").upper()

    st.sidebar.markdown(f"""
    <div style="display:flex; align-items:center; gap:10px; padding:6px 4px; margin-bottom:8px;">
        <div style="background:#0d9488; width:34px; height:34px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-weight:700; color:white; font-size:0.95rem;">
            {initial}
        </div>
        <div style="overflow:hidden;">
            <div style="font-weight:600; font-size:0.88rem; color:#f4f4f5; white-space:nowrap; text-overflow:ellipsis; overflow:hidden;">
                {user_name}
            </div>
            <div style="font-size:0.72rem; color:#10b981; font-weight:500;">
                ● Local LLM Active
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_u1, col_u2 = st.sidebar.columns(2)
    with col_u1:
        if st.button("⚙️ Profile", key="sb_btn_profile", use_container_width=True):
            try:
                st.switch_page("pages/profile.py")
            except Exception:
                st.rerun()
    with col_u2:
        if st.button("🚪 Logout", key="sb_btn_logout", use_container_width=True):
            logout()
            try:
                st.switch_page("pages/login.py")
            except Exception:
                st.rerun()

    return action_result
