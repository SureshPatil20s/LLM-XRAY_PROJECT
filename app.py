"""
LocalGPT Pro: Entry Point & Main Router
========================================
Routes user to login page or personal workspace.
"""
import streamlit as st
import database
from auth.auth import is_authenticated

st.set_page_config(
    page_title="SURESH.GPT Pro — Multi-User Local AI",
    page_icon="🤖",
    layout="wide"
)

database.init_db()

if not is_authenticated():
    try:
        st.switch_page("pages/login.py")
    except Exception:
        from pages.login import render_login_page
        render_login_page()
else:
    try:
        st.switch_page("pages/workspace.py")
    except Exception:
        from pages.workspace import render_workspace_page
        # Fallback render
