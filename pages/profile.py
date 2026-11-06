"""
User Profile and Preferences Page for LocalGPT Pro.
Provides:
- User profile display (name, email, photo, sign-in provider, member since)
- Edit display name & theme toggle (dark/light)
- Per-user generation settings (system prompt, temperature, top-k, top-p, max tokens)
- Data removal tools: "Delete all my chats" & "Delete my account"
"""
import streamlit as st
import database
from auth.guards import require_login
from auth.auth import logout
from ui.theme import apply_theme
from ui.sidebar import render_sidebar
import rag

# Page Setup
st.set_page_config(
    page_title="Profile & Settings — SURESH.GPT Pro",
    page_icon="⚙️",
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
render_sidebar(user, active_conv_id=st.session_state.get("active_conv_id"), current_page="profile")


def render_profile_page():
    # Top navigation bar
    col_back, _ = st.columns([2.5, 7.5])
    with col_back:
        if st.button("← Back to Chat Workspace", key="prof_back_btn", type="primary", use_container_width=True):
            try:
                st.switch_page("pages/workspace.py")
            except Exception:
                st.rerun()

    profile = database.get_user_profile(uid) or user
    settings = database.get_user_settings(uid)

    st.markdown("""
    <div style="margin-top: 10px; margin-bottom: 24px;">
        <h1 style="font-size: 2.1rem; font-weight: 800; color: #ececec; margin: 0; letter-spacing: -0.02em;">
            👤 User Profile &amp; Settings
        </h1>
        <p style="font-size: 0.92rem; color: #8e8ea0; margin-top: 4px;">
            Personalize your identity, default assistant instructions, and model sampling hyperparameters.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("""
        <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 16px; padding: 22px; text-align: center; margin-bottom: 20px;">
        """, unsafe_allow_html=True)

        photo = profile.get("photo_url") or f"https://api.dicebear.com/7.x/bottts/svg?seed={uid}"
        st.markdown(f"""
        <img src="{photo}" style="width: 84px; height: 84px; border-radius: 50%; border: 3px solid #38bdf8; margin-bottom: 12px; background: #1e293b;" />
        <h3 style="margin: 0; color: #f8fafc;">{profile.get('display_name', 'User')}</h3>
        <p style="color: #94a3b8; font-size: 0.85rem; margin-top: 4px;">{profile.get('email')}</p>
        <span style="background: rgba(56, 189, 248, 0.15); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 9999px; padding: 2px 10px; font-size: 0.75rem; color: #38bdf8; text-transform: uppercase;">{profile.get('provider', 'email')}</span>
        <div style="margin-top: 14px; font-size: 0.78rem; color: #64748b;">
            Member since: {profile.get('created_at', '')[:10]}
        </div>
        </div>
        """, unsafe_allow_html=True)

        # Theme Selector
        st.markdown("#### 🎨 Theme Preference")
        current_theme = profile.get("theme", "dark")
        selected_theme = st.selectbox("Appearance", options=["dark", "light"], index=0 if current_theme == "dark" else 1)
        if selected_theme != current_theme:
            database.update_user_profile(uid, theme=selected_theme)
            st.session_state["theme"] = selected_theme
            st.rerun()

    with col2:
        # Edit Profile Name
        with st.form("edit_profile_form"):
            st.markdown("### ✏️ Edit Profile Information")
            new_name = st.text_input("Display Name", value=profile.get("display_name", ""))
            new_prompt = st.text_area(
                "Default System Instructions",
                value=profile.get("default_system_prompt", "You are a helpful AI assistant. Explain concepts clearly with examples."),
                help="This prompt guides the assistant in all new chats."
            )
            submit_profile = st.form_submit_button("Save Profile & System Prompt", type="primary")

            if submit_profile:
                database.update_user_profile(uid, display_name=new_name.strip(), default_system_prompt=new_prompt.strip())
                st.session_state["user"]["display_name"] = new_name.strip()
                st.session_state["user"]["default_system_prompt"] = new_prompt.strip()
                st.success("Profile preferences saved!")
                st.rerun()

        st.markdown("---")

        # Edit Generation Settings
        with st.form("edit_settings_form"):
            st.markdown("### ⚙️ Generation Hyperparameters")
            temperature = st.slider("Temperature", min_value=0.0, max_value=1.5, value=float(settings.get("temperature", 0.7)), step=0.05)
            top_k = st.slider("Top-K", min_value=1, max_value=100, value=int(settings.get("top_k", 50)), step=5)
            top_p = st.slider("Top-P", min_value=0.1, max_value=1.0, value=float(settings.get("top_p", 0.9)), step=0.05)
            max_tokens = st.slider("Max Generation Tokens", min_value=30, max_value=1024, value=int(settings.get("max_tokens", 256)), step=16)
            submit_settings = st.form_submit_button("Save Generation Settings", type="primary")

            if submit_settings:
                database.update_user_settings(uid, temperature=temperature, top_k=top_k, top_p=top_p, max_tokens=max_tokens)
                st.success("Model generation hyperparameters updated!")
                st.rerun()

    # Data removal area
    st.markdown("---")
    st.markdown("### 🛑 Danger Zone")
    col_d1, col_d2 = st.columns(2)

    with col_d1:
        if st.button("🗑️ Delete All My Chats", type="secondary", use_container_width=True):
            database.delete_all_user_chats(uid)
            st.success("All personal conversations deleted.")
            st.rerun()

    with col_d2:
        if st.button("⚠️ Delete My Account & All Data", type="secondary", use_container_width=True):
            rag.clear_user_documents(uid)
            database.delete_user_account_data(uid)
            logout()
            st.warning("Account and all associated records permanently purged.")
            try:
                st.switch_page("pages/login.py")
            except Exception:
                st.rerun()


# Execute
render_profile_page()
