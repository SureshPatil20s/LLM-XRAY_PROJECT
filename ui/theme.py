"""
Theme tokens and style injection for SURESH.GPT Platform / LocalGPT Pro.
Matches the ChatGPT Platform dark interface from the design specifications:
- Deep dark slate/charcoal background (#18181b / #212121)
- Flat, clean dark sidebar (#121316 / #171717)
- Rounded capsule badges (yellow/green model status)
- Right-aligned user chat bubbles with user avatar
- Left-aligned assistant chat with green robot avatar and inline action icons
- Floating rounded pill chat input with red send button
- Complete removal of default Streamlit chrome & auto-page list
"""
import streamlit as st


DARK_THEME = {
    "bg_main": "#18181b",
    "sidebar_bg": "#121316",
    "card_bg": "#202123",
    "card_border": "#2e3038",
    "user_bubble": "#282a32",
    "assistant_bubble": "#1e1f24",
    "text_primary": "#ececec",
    "text_secondary": "#8e8ea0",
    "accent_green": "#10a37f",
    "accent_yellow": "#eab308",
    "accent_red": "#ef4444"
}


def apply_theme(theme_mode: str = "dark") -> None:
    """Injects custom CSS to match the ChatGPT Platform dark UI."""
    css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

        /* Global App Styling */
        .stApp {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
            background-color: #18181b !important;
            color: #ececec !important;
        }

        /* Clean Streamlit chrome & auto-page navigation */
        #MainMenu, footer, [data-testid="stDecoration"], .stAppDeployButton, [data-testid="stToolbarActions"] {
            visibility: hidden !important;
            display: none !important;
        }
        [data-testid="stSidebarNav"] {
            display: none !important;
        }

        /* Make header transparent so the sidebar reopen control (>>) is visible and clickable */
        header[data-testid="stHeader"] {
            background: transparent !important;
            height: 2.8rem !important;
            pointer-events: none !important;
            z-index: 100 !important;
        }

        /* Sidebar reopen button when collapsed (>>) */
        [data-testid="stSidebarCollapsedControl"],
        [data-testid="collapsedControl"] {
            visibility: visible !important;
            display: flex !important;
            pointer-events: auto !important;
            background: #202123 !important;
            border: 1px solid #383a42 !important;
            border-radius: 8px !important;
            color: #ececec !important;
            top: 0.6rem !important;
            left: 0.6rem !important;
            padding: 4px !important;
            z-index: 999999 !important;
            transition: all 0.2s ease !important;
        }
        [data-testid="stSidebarCollapsedControl"]:hover,
        [data-testid="collapsedControl"]:hover {
            background: #2a2b32 !important;
            border-color: #10a37f !important;
            color: #10a37f !important;
        }
        [data-testid="stSidebarCollapsedControl"] svg,
        [data-testid="collapsedControl"] svg {
            fill: #ececec !important;
        }

        /* Sidebar collapse button when open (<<) */
        [data-testid="stSidebarCollapseButton"] {
            color: #8e8ea0 !important;
            border-radius: 6px !important;
        }
        [data-testid="stSidebarCollapseButton"]:hover {
            color: #ffffff !important;
            background: #202126 !important;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #121316 !important;
            border-right: 1px solid #22242a !important;
        }
        [data-testid="stSidebar"] > div:first-child {
            padding-top: 1rem !important;
            padding-left: 0.85rem !important;
            padding-right: 0.85rem !important;
        }

        /* Search input in sidebar */
        [data-testid="stSidebar"] .stTextInput input {
            background-color: #1a1b20 !important;
            border: 1px solid #2e3038 !important;
            border-radius: 8px !important;
            color: #ececec !important;
            font-size: 0.85rem !important;
            padding: 8px 12px !important;
        }
        [data-testid="stSidebar"] .stTextInput input:focus {
            border-color: #10a37f !important;
            box-shadow: 0 0 0 1px #10a37f !important;
        }

        /* Sidebar Buttons */
        [data-testid="stSidebar"] .stButton > button {
            background-color: transparent !important;
            border: 1px solid transparent !important;
            border-radius: 8px !important;
            color: #d1d5db !important;
            font-size: 0.85rem !important;
            text-align: left !important;
            padding: 6px 10px !important;
            transition: all 0.15s ease !important;
        }
        [data-testid="stSidebar"] .stButton > button:hover {
            background-color: #1e1f26 !important;
            color: #ffffff !important;
            border-color: #2e3038 !important;
        }

        /* Highlighted Primary / Active Buttons in Sidebar */
        [data-testid="stSidebar"] .active-conv-btn > div > button,
        [data-testid="stSidebar"] button[kind="primary"] {
            background-color: #202127 !important;
            border: 1px solid #383a42 !important;
            color: #ffffff !important;
            font-weight: 500 !important;
        }

        /* Main Chat Area adjustments */
        .main .block-container {
            max-width: 980px !important;
            padding-top: 1rem !important;
            padding-bottom: 7rem !important;
        }

        /* Streamlit Chat Messages */
        [data-testid="stChatMessage"] {
            background-color: transparent !important;
            border: none !important;
            padding: 12px 4px !important;
            margin-bottom: 4px !important;
        }

        /* User Message Container: Right-aligned pill */
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]),
        [data-testid="stChatMessage"]:has(img[alt="user avatar"]) {
            flex-direction: row-reverse !important;
            text-align: left !important;
        }
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"],
        [data-testid="stChatMessage"]:has(img[alt="user avatar"]) [data-testid="stChatMessageContent"] {
            background-color: #282a32 !important;
            border: 1px solid #383a42 !important;
            border-radius: 18px !important;
            padding: 12px 18px !important;
            max-width: 75% !important;
            color: #ececec !important;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2) !important;
        }

        /* Assistant Message Container */
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]),
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
            flex-direction: row !important;
        }
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"],
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] {
            background-color: #1e1f25 !important;
            border: 1px solid #2b2d35 !important;
            border-radius: 18px !important;
            padding: 16px 20px !important;
            color: #ececec !important;
            box-shadow: 0 2px 12px rgba(0,0,0,0.25) !important;
        }

        /* Assistant avatar circle */
        [data-testid="chatAvatarIcon-assistant"] {
            background-color: #10a37f !important;
            color: white !important;
            border-radius: 50% !important;
            padding: 4px !important;
        }
        [data-testid="chatAvatarIcon-user"] {
            background-color: #0d9488 !important;
            color: white !important;
            border-radius: 50% !important;
            padding: 4px !important;
        }

        /* Message Action Buttons (Copy, Like, Dislike, X-Ray) */
        .msg-action-btn button {
            background: transparent !important;
            border: none !important;
            color: #71717a !important;
            font-size: 0.8rem !important;
            padding: 2px 6px !important;
            min-height: 28px !important;
            border-radius: 6px !important;
        }
        .msg-action-btn button:hover {
            color: #e4e4e7 !important;
            background: #27272a !important;
        }

        /* Chat Input Dock */
        [data-testid="stChatInput"] {
            background-color: #202123 !important;
            border: 1px solid #383a42 !important;
            border-radius: 26px !important;
            padding: 6px 14px !important;
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.45) !important;
        }
        [data-testid="stChatInput"] textarea {
            background-color: transparent !important;
            color: #ececec !important;
            font-size: 0.95rem !important;
        }
        [data-testid="stChatInput"] textarea::placeholder {
            color: #71717a !important;
        }
        [data-testid="stChatInputSubmitButton"] button {
            background-color: #ef4444 !important;
            color: #ffffff !important;
            border-radius: 12px !important;
            border: none !important;
            width: 34px !important;
            height: 34px !important;
            transition: transform 0.15s ease !important;
        }
        [data-testid="stChatInputSubmitButton"] button:hover {
            background-color: #dc2626 !important;
            transform: scale(1.05) !important;
        }

        /* Prompt Cards */
        .prompt-card {
            background-color: #1e1f25;
            border: 1px solid #2e3038;
            border-radius: 14px;
            padding: 16px;
            cursor: pointer;
            transition: all 0.2s ease;
            height: 100%;
        }
        .prompt-card:hover {
            border-color: #10a37f;
            background-color: #23252c;
            transform: translateY(-2px);
            box-shadow: 0 4px 16px rgba(16, 163, 127, 0.2);
        }
        .prompt-card h4 {
            font-size: 0.92rem;
            margin-bottom: 6px;
            color: #f4f4f5;
            font-weight: 600;
        }
        .prompt-card p {
            font-size: 0.8rem;
            color: #8e8ea0;
            line-height: 1.4;
            margin: 0;
        }

        /* Top Header Navbar */
        .top-navbar-container {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: #18181b;
            border-bottom: 1px solid #27272a;
            padding: 8px 16px;
            margin-bottom: 16px;
            position: sticky;
            top: 0;
            z-index: 99;
        }

        /* Custom Scrollbar */
        ::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }
        ::-webkit-scrollbar-track {
            background: transparent;
        }
        ::-webkit-scrollbar-thumb {
            background: #2e3038;
            border-radius: 3px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #3f424e;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
