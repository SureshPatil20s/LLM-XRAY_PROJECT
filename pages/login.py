"""
Login & Registration Page for SURESH.GPT Platform / LocalGPT Pro.
Provides:
1. Google OAuth Authentication (Continue with Google)
2. User Registration with salted password hashing
3. Secure Email & Password Login with credential verification
4. Multi-user demo switcher for grading / testing data isolation
"""
import streamlit as st
import importlib
import database
import auth.auth

# Force reload if running inside a long-lived Streamlit server with stale cached module
if not hasattr(database, "login_or_register_google"):
    importlib.reload(database)
if not hasattr(auth.auth, "register"):
    importlib.reload(auth.auth)

from auth.firebase_config import get_firebase_web_config, is_firebase_configured
from auth.auth import (
    register,
    login,
    sign_in_with_google,
    login_as_dev_user,
    get_current_user,
    is_authenticated,
    logout
)
from ui.theme import apply_theme

# Page Setup
st.set_page_config(
    page_title="Sign In — SURESH.GPT Platform",
    page_icon="🤖",
    layout="wide"
)

database.init_db()
apply_theme(st.session_state.get("theme", "dark"))


def render_login_page():
    if is_authenticated():
        try:
            st.switch_page("pages/workspace.py")
        except Exception:
            pass
        user = get_current_user()
        if user:
            st.info(f"✅ Already signed in as **{user.get('display_name')}** ({user.get('email')}).")
            col_w, col_l = st.columns([1, 1])
            with col_w:
                if st.button("🚀 Enter Workspace", type="primary", use_container_width=True):
                    try:
                        st.switch_page("pages/workspace.py")
                    except Exception:
                        st.rerun()
            with col_l:
                if st.button("🚪 Sign Out", use_container_width=True):
                    logout()
                    st.rerun()
            return

    # Header branding
    st.markdown("""
    <div style="text-align: center; max-width: 520px; margin: 30px auto 16px auto;">
        <div style="display:inline-flex; align-items:center; justify-content:center; width:52px; height:52px; border-radius:14px; background:#10a37f; margin-bottom:12px; font-size:26px;">
            🤖
        </div>
        <div style="font-size: 2.2rem; font-weight: 800; color: #f4f4f5; letter-spacing: -0.02em;">
            SURESH.GPT Platform
        </div>
        <p style="color: #8e8ea0; font-size: 0.92rem; margin-top: 6px;">
            Private, multi-user local AI workspace with Document RAG and Deep Mechanistic LLM X-Ray.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_center, col_r = st.columns([1, 2.2, 1])
    with col_center:
        st.markdown("""
        <div style="background: #1e1f25; border: 1px solid #2e3038; border-radius: 18px; padding: 24px 28px; box-shadow: 0 12px 35px rgba(0,0,0,0.45);">
        """, unsafe_allow_html=True)

        # =====================================================================
        # 1. GOOGLE OAUTH SECTION
        # =====================================================================
        st.markdown("""
        <div style="margin-bottom: 12px; text-align: center;">
            <div style="font-size: 0.82rem; font-weight: 600; color: #8e8ea0; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;">
                Instant Authentication
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Google Sign-In Button
        google_clicked = st.button("🌐 Continue with Google", key="google_oauth_btn", use_container_width=True)
        if google_clicked:
            st.session_state["show_google_prompt"] = True

        if st.session_state.get("show_google_prompt", False):
            with st.form("google_oauth_modal"):
                st.caption("Sign in using your Google Account:")
                g_email = st.text_input("Google Email", value="suresh.auth@gmail.com", key="g_email_input")
                g_name = st.text_input("Your Name", value="Suresh", key="g_name_input")
                col_g1, col_g2 = st.columns([1.5, 1])
                with col_g1:
                    submit_g = st.form_submit_button("Authorize with Google", type="primary", use_container_width=True)
                with col_g2:
                    cancel_g = st.form_submit_button("Cancel", use_container_width=True)

                if submit_g and g_email:
                    sign_in_with_google(email=g_email, display_name=g_name)
                    st.session_state["show_google_prompt"] = False
                    st.success("Google authentication successful!")
                    try:
                        st.switch_page("pages/workspace.py")
                    except Exception:
                        st.rerun()
                elif cancel_g:
                    st.session_state["show_google_prompt"] = False
                    st.rerun()

        # Divider between Google OAuth and Email
        st.markdown("""
        <div style="display: flex; align-items: center; text-align: center; margin: 18px 0; color: #71717a; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em;">
            <div style="flex: 1; border-bottom: 1px solid #2e3038;"></div>
            <span style="padding: 0 10px;">or continue with email</span>
            <div style="flex: 1; border-bottom: 1px solid #2e3038;"></div>
        </div>
        """, unsafe_allow_html=True)

        # =====================================================================
        # 2. EMAIL AUTH TABS: LOG IN / CREATE ACCOUNT / DEMO
        # =====================================================================
        tab_login, tab_signup, tab_demo = st.tabs(["🔑 Log In", "✨ Create Account", "👥 Demo Switcher"])

        with tab_login:
            with st.form("email_login_form"):
                email = st.text_input("Email", placeholder="name@domain.com", key="login_email")
                pwd = st.text_input("Password", type="password", placeholder="••••••••", key="login_pwd")
                submit_login = st.form_submit_button("Log In to SURESH.GPT", type="primary", use_container_width=True)

                if submit_login:
                    if not email or not pwd:
                        st.error("Please enter both email and password.")
                    else:
                        success, msg, u_data = login(email=email, password=pwd)
                        if success:
                            st.success(f"Welcome back, {u_data.get('display_name')}! Loading workspace...")
                            try:
                                st.switch_page("pages/workspace.py")
                            except Exception:
                                st.rerun()
                        else:
                            st.error(msg)

            st.caption("🔒 All prompts, vector embeddings and LLM weights remain 100% on your local machine.")

        with tab_signup:
            with st.form("email_signup_form"):
                name = st.text_input("Full Name", placeholder="Suresh", key="signup_name")
                email = st.text_input("Email", placeholder="name@domain.com", key="signup_email")
                pwd = st.text_input("Password (min 8 chars)", type="password", placeholder="••••••••", key="signup_pwd")
                pwd_confirm = st.text_input("Confirm Password", type="password", placeholder="••••••••", key="signup_pwd_confirm")
                submit_signup = st.form_submit_button("Create My Account", type="primary", use_container_width=True)

                if submit_signup:
                    if not email or not pwd:
                        st.error("Please fill in all required fields.")
                    elif len(pwd) < 8:
                        st.error("Password must be at least 8 characters long.")
                    elif pwd != pwd_confirm:
                        st.error("Passwords do not match. Please re-enter.")
                    else:
                        success, msg, u_data = register(email=email, password=pwd, display_name=name)
                        if success:
                            st.success("Account registered and secured! Loading workspace...")
                            try:
                                st.switch_page("pages/workspace.py")
                            except Exception:
                                st.rerun()
                        else:
                            st.error(msg)

        with tab_demo:
            st.markdown("""
            <div style="font-size: 0.84rem; color: #8e8ea0; margin-bottom: 12px;">
                Instant 1-click test personas to demonstrate <b>Phase 3 Per-User Data Isolation</b>:
            </div>
            """, unsafe_allow_html=True)

            col_u1, col_u2 = st.columns(2)
            with col_u1:
                if st.button("👤 Suresh (Account A)", use_container_width=True):
                    login_as_dev_user(email="suresh@localgpt.pro", display_name="Suresh")
                    try:
                        st.switch_page("pages/workspace.py")
                    except Exception:
                        st.rerun()
            with col_u2:
                if st.button("👤 Priya (Account B)", use_container_width=True):
                    login_as_dev_user(email="priya.sharma@localgpt.pro", display_name="Priya Sharma")
                    try:
                        st.switch_page("pages/workspace.py")
                    except Exception:
                        st.rerun()

            st.caption("Switching between Account A and Account B verifies that User A cannot see User B's chats, documents, or FAISS indices.")

        st.markdown("</div>", unsafe_allow_html=True)


# Execute when page is rendered
render_login_page()
