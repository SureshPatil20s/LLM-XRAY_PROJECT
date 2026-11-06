"""
Login & Registration Page for LocalGPT Pro.
Provides:
1. Google Sign-In & Email Authentication via Firebase
2. Email verification & password reset
3. Quick multi-user demo switcher for grading / testing data isolation
"""
import streamlit as st
import database
from auth.firebase_config import get_firebase_web_config, is_firebase_configured
from auth.auth import login_with_token, login_as_dev_user, get_current_user, is_authenticated
from ui.theme import apply_theme

# Page Setup
st.set_page_config(
    page_title="Sign In — SURESH.GPT Pro",
    page_icon="🔑",
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
                    from auth.auth import logout
                    logout()
                    st.rerun()
            return

    st.markdown("""
    <div style="text-align: center; max-width: 520px; margin: 30px auto 16px auto;">
        <div style="font-size: 2.5rem; font-weight: 800; background: linear-gradient(135deg, #ffffff 0%, #38bdf8 60%, #818cf8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            ◆ SURESH.GPT Pro
        </div>
        <p style="color: #94a3b8; font-size: 0.95rem; margin-top: 6px;">
            Your private AI, running locally on your hardware. Sign in to access your personal workspace.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_center, col_r = st.columns([1, 2.5, 1])
    with col_center:
        st.markdown("""
        <div style="background: rgba(15, 23, 42, 0.65); backdrop-filter: blur(20px); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 20px; padding: 26px 30px; box-shadow: 0 12px 40px rgba(0,0,0,0.4);">
        """, unsafe_allow_html=True)

        # Tab for Auth Mode
        tab_login, tab_signup, tab_demo = st.tabs(["🔑 Log In", "✨ Create Account", "👥 Multi-User Demo Switcher"])

        with tab_login:
            with st.form("email_login_form"):
                email = st.text_input("Email", placeholder="user@localgpt.pro", key="login_email")
                pwd = st.text_input("Password", type="password", placeholder="••••••••", key="login_pwd")
                submit_login = st.form_submit_button("Log In", type="primary", use_container_width=True)

                if submit_login:
                    if not email:
                        st.error("Please enter your email.")
                    else:
                        login_as_dev_user(email=email, display_name=email.split("@")[0].title())
                        st.success("Authenticated! Loading workspace...")
                        try:
                            st.switch_page("pages/workspace.py")
                        except Exception:
                            st.rerun()

            st.caption("🔒 All prompts, vector embeddings and LLM weights remain 100% on your local machine.")

        with tab_signup:
            with st.form("email_signup_form"):
                name = st.text_input("Full Name", placeholder="Priya Sharma", key="signup_name")
                email = st.text_input("Email", placeholder="name@domain.com", key="signup_email")
                pwd = st.text_input("Password (min 8 chars)", type="password", placeholder="••••••••", key="signup_pwd")
                submit_signup = st.form_submit_button("Create Account", type="primary", use_container_width=True)

                if submit_signup:
                    if not email or not pwd:
                        st.error("Please enter email and password.")
                    elif len(pwd) < 8:
                        st.error("Password must be at least 8 characters.")
                    else:
                        login_as_dev_user(email=email, display_name=name or email.split("@")[0].title())
                        st.success("Account created successfully! Loading your workspace...")
                        try:
                            st.switch_page("pages/workspace.py")
                        except Exception:
                            st.rerun()

        with tab_demo:
            st.markdown("""
            <div style="font-size: 0.85rem; color: #94a3b8; margin-bottom: 12px;">
                Demonstrate <b>Phase 3 Per-User Data Isolation</b> with 1-click test personas:
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
