"""
Authentication state management and user session handling.
Provides:
- User Registration with salted password hashing
- User Login & credential verification
- Google OAuth authentication & profile sync
- Session persistence across multi-page navigation and page refreshes
"""
import streamlit as st
import importlib
from typing import Optional, Dict, Any, Tuple
from auth.firebase_admin_init import verify_token
import database

# Ensure live module sync if Streamlit kept an older database module in memory
if not hasattr(database, "login_or_register_google"):
    importlib.reload(database)


def restore_session_if_needed() -> Optional[Dict[str, Any]]:
    """Restores user session from persistent DB storage if session_state is empty."""
    if "user" in st.session_state and st.session_state["user"] is not None:
        return st.session_state["user"]

    last_uid = database.get_last_active_uid()
    if last_uid:
        user_record = database.get_user_profile(last_uid)
        if user_record:
            user_session = {
                "uid": user_record["uid"],
                "email": user_record.get("email", ""),
                "display_name": user_record.get("display_name", "User"),
                "photo_url": user_record.get("photo_url", ""),
                "provider": user_record.get("provider", "local"),
                "email_verified": True,
                "id_token": f"restored_token_::{user_record['uid']}",
                "theme": user_record.get("theme", "dark"),
                "default_system_prompt": user_record.get("default_system_prompt")
            }
            st.session_state["user"] = user_session
            st.session_state["theme"] = user_session["theme"]
            return user_session
    return None


def get_current_user() -> Optional[Dict[str, Any]]:
    """Returns the authenticated user dict from session state if logged in."""
    return restore_session_if_needed()


def is_authenticated() -> bool:
    """Checks whether a valid user session is active (or restorable)."""
    return restore_session_if_needed() is not None


def register(email: str, password: str, display_name: Optional[str] = None) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Registers a new user with email and password.
    Returns (success, message, user_dict).
    """
    success, msg, profile = database.register_user(email, password, display_name)
    if success and profile:
        user_session = {
            "uid": profile["uid"],
            "email": profile["email"],
            "display_name": profile["display_name"],
            "photo_url": profile["photo_url"],
            "provider": "email",
            "email_verified": True,
            "id_token": f"email_token_::{profile['uid']}",
            "theme": profile.get("theme", "dark"),
            "default_system_prompt": profile.get("default_system_prompt")
        }
        st.session_state["user"] = user_session
        st.session_state["theme"] = user_session["theme"]
        return True, msg, user_session
    return False, msg, None


def login(email: str, password: str) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Authenticates user with email and password.
    Returns (success, message, user_dict).
    """
    success, msg, profile = database.authenticate_user(email, password)
    if success and profile:
        user_session = {
            "uid": profile["uid"],
            "email": profile["email"],
            "display_name": profile["display_name"],
            "photo_url": profile["photo_url"],
            "provider": profile.get("provider", "email"),
            "email_verified": True,
            "id_token": f"email_token_::{profile['uid']}",
            "theme": profile.get("theme", "dark"),
            "default_system_prompt": profile.get("default_system_prompt")
        }
        st.session_state["user"] = user_session
        st.session_state["theme"] = user_session["theme"]
        return True, msg, user_session
    return False, msg, None


def sign_in_with_google(email: str, display_name: Optional[str] = None, photo_url: Optional[str] = None) -> Dict[str, Any]:
    """
    Authenticates or signs up a user via Google OAuth provider.
    """
    profile = database.login_or_register_google(email, display_name, photo_url)
    user_session = {
        "uid": profile["uid"],
        "email": profile["email"],
        "display_name": profile["display_name"],
        "photo_url": profile["photo_url"],
        "provider": "google",
        "email_verified": True,
        "id_token": f"google_token_::{profile['uid']}",
        "theme": profile.get("theme", "dark"),
        "default_system_prompt": profile.get("default_system_prompt")
    }
    st.session_state["user"] = user_session
    st.session_state["theme"] = user_session["theme"]
    return user_session


def login_with_token(id_token: str) -> Dict[str, Any]:
    """
    Verifies Firebase token, syncs user with database, and sets session state.
    """
    decoded = verify_token(id_token)
    uid = decoded["uid"]
    email = decoded.get("email", "")
    display_name = decoded.get("name") or (email.split("@")[0] if email else "Local User")
    photo_url = decoded.get("picture", "")
    provider = decoded.get("firebase", {}).get("sign_in_provider", "firebase")
    email_verified = decoded.get("email_verified", True)

    user_record = database.upsert_user(
        uid=uid,
        email=email,
        display_name=display_name,
        photo_url=photo_url,
        provider=provider
    )

    user_session = {
        "uid": uid,
        "email": email,
        "display_name": display_name,
        "photo_url": photo_url,
        "provider": provider,
        "email_verified": email_verified,
        "id_token": id_token,
        "theme": user_record.get("theme", "dark"),
        "default_system_prompt": user_record.get("default_system_prompt")
    }

    st.session_state["user"] = user_session
    st.session_state["theme"] = user_session["theme"]
    database.set_last_active_uid(uid)
    return user_session


def login_as_dev_user(email: str, display_name: Optional[str] = None, provider: str = "local") -> Dict[str, Any]:
    """
    Local developer / offline login for testing multi-user environments.
    Generates a deterministic local uid based on email.
    """
    safe_email = email.strip().lower()
    uid = "usr_" + "".join(c for c in safe_email if c.isalnum())
    name = display_name or safe_email.split("@")[0].title()
    photo_url = f"https://api.dicebear.com/7.x/bottts/svg?seed={uid}"

    user_record = database.upsert_user(
        uid=uid,
        email=safe_email,
        display_name=name,
        photo_url=photo_url,
        provider=provider
    )

    user_session = {
        "uid": uid,
        "email": safe_email,
        "display_name": name,
        "photo_url": photo_url,
        "provider": provider,
        "email_verified": True,
        "id_token": f"dev_token_::{uid}::{safe_email}::{name}::{photo_url}",
        "theme": user_record.get("theme", "dark"),
        "default_system_prompt": user_record.get("default_system_prompt")
    }

    st.session_state["user"] = user_session
    st.session_state["theme"] = user_session["theme"]
    database.set_last_active_uid(uid)
    return user_session


def logout() -> None:
    """Terminates active session and clears user session state."""
    database.set_last_active_uid(None)
    st.session_state.pop("user", None)
    st.session_state.pop("active_conv_id", None)
    st.session_state.pop("xray_cache", None)
    st.session_state.pop("edit_mode_msg_id", None)
