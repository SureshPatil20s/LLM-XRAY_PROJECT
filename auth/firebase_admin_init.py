"""
Initializes firebase-admin SDK and verifies ID tokens on the backend.
"""
import os
import json
from typing import Optional, Dict, Any
import firebase_admin
from firebase_admin import credentials, auth

from auth.firebase_config import get_service_account_path, is_firebase_configured

_FIREBASE_APP_INITIALIZED = False


def init_firebase_admin() -> bool:
    """Initializes the Firebase Admin SDK once."""
    global _FIREBASE_APP_INITIALIZED
    if _FIREBASE_APP_INITIALIZED or len(firebase_admin._apps) > 0:
        _FIREBASE_APP_INITIALIZED = True
        return True

    sa_path = get_service_account_path()
    if os.path.exists(sa_path):
        try:
            cred = credentials.Certificate(sa_path)
            firebase_admin.initialize_app(cred)
            _FIREBASE_APP_INITIALIZED = True
            return True
        except Exception as e:
            print(f"[Firebase Admin Warning] Failed to initialize: {e}")
            return False
    return False


def verify_token(id_token: str) -> Dict[str, Any]:
    """
    Verifies Firebase ID token with firebase-admin SDK.
    Raises exception if token is invalid, revoked, or expired.
    Also supports development tokens when offline/mocking.
    """
    if init_firebase_admin():
        # Production Firebase verification
        decoded_token = auth.verify_id_token(id_token, check_revoked=True)
        return decoded_token
    else:
        # Development / Offline Fallback verification
        if id_token.startswith("dev_token_"):
            parts = id_token.split("::")
            if len(parts) >= 3:
                return {
                    "uid": parts[1],
                    "email": parts[2],
                    "name": parts[3] if len(parts) > 3 else parts[2].split("@")[0],
                    "picture": parts[4] if len(parts) > 4 else "",
                    "email_verified": True,
                    "firebase": {"sign_in_provider": "password"}
                }
        raise ValueError("Firebase Admin SDK is not initialized and token is not valid.")

