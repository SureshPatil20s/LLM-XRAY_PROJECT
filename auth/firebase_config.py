"""
Loads Firebase Web and Admin configuration from environment variables (.env).
"""
import os
from dotenv import load_dotenv

# Load .env file from project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))


def get_firebase_web_config() -> dict:
    """Returns Firebase Web configuration dictionary for frontend SDK."""
    return {
        "apiKey": os.getenv("FIREBASE_API_KEY", ""),
        "authDomain": os.getenv("FIREBASE_AUTH_DOMAIN", "localgpt-pro.firebaseapp.com"),
        "projectId": os.getenv("FIREBASE_PROJECT_ID", "localgpt-pro"),
        "appId": os.getenv("FIREBASE_APP_ID", ""),
    }


def get_service_account_path() -> str:
    """Returns path to the service account JSON key file."""
    custom_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "serviceAccountKey.json")
    if os.path.isabs(custom_path):
        return custom_path
    return os.path.join(BASE_DIR, custom_path)


def is_firebase_configured() -> bool:
    """Checks whether Firebase credentials are fully configured."""
    sa_path = get_service_account_path()
    web_conf = get_firebase_web_config()
    return bool(os.path.exists(sa_path) and web_conf["apiKey"])

