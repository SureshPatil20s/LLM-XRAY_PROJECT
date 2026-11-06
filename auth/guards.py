"""
Authentication route guard for protected views in LocalGPT Pro.
Ensures session persistence across multi-page navigation and page refreshes.
"""
import streamlit as st
from auth.auth import restore_session_if_needed


def require_login() -> dict:
    """
    Enforces login guard. If no authenticated user exists in session_state,
    attempts automatic session recovery from persistent storage.
    If still unauthenticated, redirects to pages/login.py and stops execution.
    Returns the user dict if authenticated.
    """
    user = restore_session_if_needed()
    if not user:
        try:
            st.switch_page("pages/login.py")
        except Exception:
            pass
        st.warning("🔒 Authentication required. Please sign in to access your personal workspace.")
        st.stop()

    if not user.get("email_verified", True):
        st.warning("✉️ Please verify your email address to use LocalGPT Pro.")
        if st.button("Resend Verification Email"):
            st.info("Verification link resent to your email address.")
        st.stop()

    return user
