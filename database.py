"""
Database module for LocalGPT Pro.
Manages per-user persistent storage of:
- Users (profile, preferences, credentials/provider, last login)
- Chats & Messages (pinned chats, date tracking, user isolation)
- Documents metadata (doc_id, uid, filename, pages, chunks, uploaded_at)
- User Settings (temperature, top_k, top_p, max_tokens, system prompt)
"""
import sqlite3
import os
import json
import uuid
import hashlib
import secrets
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple

DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "conversations")
DB_PATH = os.path.join(DB_DIR, "localgpt.db")


def _get_connection() -> sqlite3.Connection:
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Initialize database tables with per-user data isolation schemas."""
    conn = _get_connection()
    with conn:
        # 1. Users table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                uid TEXT PRIMARY KEY,
                email TEXT,
                display_name TEXT,
                photo_url TEXT,
                provider TEXT DEFAULT 'email',
                created_at TEXT NOT NULL,
                last_login TEXT NOT NULL,
                theme TEXT DEFAULT 'dark',
                default_system_prompt TEXT DEFAULT 'You are a helpful AI assistant. Explain concepts clearly with examples.'
            )
        """)

        # 2. Chats table (with uid and pinned support)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                uid TEXT NOT NULL DEFAULT 'local_default_user',
                title TEXT NOT NULL,
                pinned INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        # Check if uid and pinned columns exist in conversations (for migration from Phase 2)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(conversations)")
        columns = [row["name"] for row in cursor.fetchall()]
        if "uid" not in columns:
            cursor.execute("ALTER TABLE conversations ADD COLUMN uid TEXT NOT NULL DEFAULT 'local_default_user'")
        if "pinned" not in columns:
            cursor.execute("ALTER TABLE conversations ADD COLUMN pinned INTEGER DEFAULT 0")

        # Check if password_hash and salt exist in users
        cursor.execute("PRAGMA table_info(users)")
        user_columns = [row["name"] for row in cursor.fetchall()]
        if "password_hash" not in user_columns:
            cursor.execute("ALTER TABLE users ADD COLUMN password_hash TEXT")
        if "salt" not in user_columns:
            cursor.execute("ALTER TABLE users ADD COLUMN salt TEXT")

        # 3. Messages table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sources_json TEXT,
                xray_data_json TEXT,
                timestamp TEXT NOT NULL,
                created_at TEXT,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            )
        """)

        cursor.execute("PRAGMA table_info(messages)")
        msg_columns = [row["name"] for row in cursor.fetchall()]
        if "created_at" not in msg_columns:
            cursor.execute("ALTER TABLE messages ADD COLUMN created_at TEXT")

        # 4. User Documents table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                uid TEXT NOT NULL,
                filename TEXT NOT NULL,
                pages INTEGER DEFAULT 1,
                chunks INTEGER DEFAULT 1,
                uploaded_at TEXT NOT NULL
            )
        """)

        # 5. User Settings table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                uid TEXT PRIMARY KEY,
                temperature REAL DEFAULT 0.7,
                top_k INTEGER DEFAULT 50,
                top_p REAL DEFAULT 0.9,
                max_tokens INTEGER DEFAULT 256
            )
        """)

        # 6. Persistent App State (active session persistence across multi-page navigation)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS app_state (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)

    conn.close()


def set_last_active_uid(uid: Optional[str]) -> None:
    """Sets or clears the persistent last active user ID."""
    init_db()
    conn = _get_connection()
    with conn:
        if uid:
            conn.execute("INSERT OR REPLACE INTO app_state (key, value) VALUES ('last_active_uid', ?)", (uid,))
        else:
            conn.execute("DELETE FROM app_state WHERE key = 'last_active_uid'")
    conn.close()


def get_last_active_uid() -> Optional[str]:
    """Retrieves the persistent last active user ID if any."""
    init_db()
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM app_state WHERE key = 'last_active_uid'")
    row = cursor.fetchone()
    conn.close()
    return row["value"] if row else None


# ==============================================================================
# USER MANAGEMENT & PROFILES
# ==============================================================================

def upsert_user(
    uid: str,
    email: Optional[str] = None,
    display_name: Optional[str] = None,
    photo_url: Optional[str] = None,
    provider: str = "email"
) -> Dict[str, Any]:
    """Inserts a new user or updates last_login and profile info."""
    init_db()
    now = datetime.now(timezone.utc).isoformat()
    conn = _get_connection()
    with conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE uid = ?", (uid,))
        existing = cursor.fetchone()

        if existing:
            update_name = display_name if display_name else existing["display_name"]
            update_photo = photo_url if photo_url else existing["photo_url"]
            update_email = email if email else existing["email"]
            cursor.execute("""
                UPDATE users 
                SET email = ?, display_name = ?, photo_url = ?, last_login = ?, provider = ?
                WHERE uid = ?
            """, (update_email, update_name, update_photo, now, provider, uid))
        else:
            default_name = display_name or (email.split("@")[0] if email else "Local User")
            cursor.execute("""
                INSERT INTO users (uid, email, display_name, photo_url, provider, created_at, last_login)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (uid, email or "", default_name, photo_url or "", provider, now, now))
            
            # Create default settings entry
            cursor.execute("""
                INSERT OR IGNORE INTO settings (uid, temperature, top_k, top_p, max_tokens)
                VALUES (?, 0.7, 50, 0.9, 256)
            """, (uid,))
            
    conn.close()
    return get_user_profile(uid) or {}


def get_user_profile(uid: str) -> Optional[Dict[str, Any]]:
    """Fetches user profile by uid."""
    init_db()
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE uid = ?", (uid,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    """Fetches user profile by email."""
    init_db()
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE LOWER(email) = ?", (email.strip().lower(),))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def hash_password(password: str, salt: str) -> str:
    """Computes SHA-256 salted password hash."""
    return hashlib.sha256((salt + password).encode('utf-8')).hexdigest()


def register_user(email: str, password: str, display_name: Optional[str] = None) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Registers a new user with email, hashed password, and display name.
    Returns (success, message, user_profile).
    """
    init_db()
    norm_email = email.strip().lower()
    existing = get_user_by_email(norm_email)
    if existing:
        return False, "An account with this email already exists. Please log in.", None

    uid = "usr_" + secrets.token_hex(8)
    salt = secrets.token_hex(16)
    pwd_hash = hash_password(password, salt)
    name = display_name.strip() if (display_name and display_name.strip()) else norm_email.split("@")[0].title()
    photo_url = f"https://api.dicebear.com/7.x/bottts/svg?seed={uid}"
    now = datetime.now(timezone.utc).isoformat()

    conn = _get_connection()
    with conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO users (uid, email, display_name, photo_url, provider, created_at, last_login, password_hash, salt)
            VALUES (?, ?, ?, ?, 'email', ?, ?, ?, ?)
        """, (uid, norm_email, name, photo_url, now, now, pwd_hash, salt))

        cursor.execute("""
            INSERT OR IGNORE INTO settings (uid, temperature, top_k, top_p, max_tokens)
            VALUES (?, 0.7, 50, 0.9, 256)
        """, (uid,))
    conn.close()

    set_last_active_uid(uid)
    return True, "Account registered successfully!", get_user_profile(uid)


def authenticate_user(email: str, password: str) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Validates user credentials against stored salted hash.
    Returns (success, message, user_profile).
    """
    init_db()
    norm_email = email.strip().lower()
    user = get_user_by_email(norm_email)
    if not user:
        return False, "No account found with this email. Please register first.", None

    if user.get("provider") == "google" and not user.get("password_hash"):
        return False, "This account was created with Google. Please use 'Continue with Google'.", None

    stored_hash = user.get("password_hash")
    salt = user.get("salt")
    if not stored_hash or not salt:
        # Fallback for dev users created without password
        now = datetime.now(timezone.utc).isoformat()
        conn = _get_connection()
        with conn:
            conn.execute("UPDATE users SET last_login = ? WHERE uid = ?", (now, user["uid"]))
        conn.close()
        set_last_active_uid(user["uid"])
        return True, "Login successful!", user

    computed_hash = hash_password(password, salt)
    if computed_hash != stored_hash:
        return False, "Incorrect password. Please verify and try again.", None

    now = datetime.now(timezone.utc).isoformat()
    conn = _get_connection()
    with conn:
        conn.execute("UPDATE users SET last_login = ? WHERE uid = ?", (now, user["uid"]))
    conn.close()

    set_last_active_uid(user["uid"])
    return True, "Login successful!", user


def login_or_register_google(email: str, display_name: Optional[str] = None, photo_url: Optional[str] = None) -> Dict[str, Any]:
    """
    Authenticates or registers a user via Google OAuth.
    """
    init_db()
    norm_email = email.strip().lower()
    existing = get_user_by_email(norm_email)
    now = datetime.now(timezone.utc).isoformat()

    if existing:
        uid = existing["uid"]
        conn = _get_connection()
        with conn:
            conn.execute("""
                UPDATE users SET last_login = ?, provider = 'google'
                WHERE uid = ?
            """, (now, uid))
        conn.close()
    else:
        uid = "usr_g_" + secrets.token_hex(8)
        name = display_name or norm_email.split("@")[0].title()
        avatar = photo_url or f"https://api.dicebear.com/7.x/avataaars/svg?seed={norm_email}"
        conn = _get_connection()
        with conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO users (uid, email, display_name, photo_url, provider, created_at, last_login)
                VALUES (?, ?, ?, ?, 'google', ?, ?)
            """, (uid, norm_email, name, avatar, now, now))
            cursor.execute("""
                INSERT OR IGNORE INTO settings (uid, temperature, top_k, top_p, max_tokens)
                VALUES (?, 0.7, 50, 0.9, 256)
            """, (uid,))
        conn.close()

    set_last_active_uid(uid)
    return get_user_profile(uid) or {}


def update_user_profile(
    uid: str,
    display_name: Optional[str] = None,
    photo_url: Optional[str] = None,
    theme: Optional[str] = None,
    default_system_prompt: Optional[str] = None
) -> None:
    """Updates user profile preferences."""
    init_db()
    conn = _get_connection()
    with conn:
        cursor = conn.cursor()
        if display_name is not None:
            cursor.execute("UPDATE users SET display_name = ? WHERE uid = ?", (display_name, uid))
        if photo_url is not None:
            cursor.execute("UPDATE users SET photo_url = ? WHERE uid = ?", (photo_url, uid))
        if theme is not None:
            cursor.execute("UPDATE users SET theme = ? WHERE uid = ?", (theme, uid))
        if default_system_prompt is not None:
            cursor.execute("UPDATE users SET default_system_prompt = ? WHERE uid = ?", (default_system_prompt, uid))
    conn.close()


def get_user_settings(uid: str) -> Dict[str, Any]:
    """Retrieves user generation settings."""
    init_db()
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM settings WHERE uid = ?", (uid,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return {"uid": uid, "temperature": 0.7, "top_k": 50, "top_p": 0.9, "max_tokens": 256}


def update_user_settings(uid: str, temperature: float, top_k: int, top_p: float, max_tokens: int) -> None:
    """Saves user generation settings."""
    init_db()
    conn = _get_connection()
    with conn:
        conn.execute("""
            INSERT INTO settings (uid, temperature, top_k, top_p, max_tokens)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(uid) DO UPDATE SET
                temperature = excluded.temperature,
                top_k = excluded.top_k,
                top_p = excluded.top_p,
                max_tokens = excluded.max_tokens
        """, (uid, temperature, top_k, top_p, max_tokens))
    conn.close()


def delete_all_user_chats(uid: str) -> None:
    """Deletes all chats and associated messages for a specific user."""
    init_db()
    conn = _get_connection()
    with conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM conversations WHERE uid = ?", (uid,))
        conv_ids = [r[0] for r in cursor.fetchall()]
        for cid in conv_ids:
            cursor.execute("DELETE FROM messages WHERE conversation_id = ?", (cid,))
        cursor.execute("DELETE FROM conversations WHERE uid = ?", (uid,))
    conn.close()


def delete_user_account_data(uid: str) -> None:
    """Deletes all database records associated with a user."""
    init_db()
    conn = _get_connection()
    with conn:
        delete_all_user_chats(uid)
        conn.execute("DELETE FROM documents WHERE uid = ?", (uid,))
        conn.execute("DELETE FROM settings WHERE uid = ?", (uid,))
        conn.execute("DELETE FROM users WHERE uid = ?", (uid,))
    conn.close()


# ==============================================================================
# PER-USER CHAT MANAGEMENT
# ==============================================================================

def create_conversation(title: str = "New Chat", uid: str = "local_default_user") -> str:
    """Creates a new conversation associated with a specific user ID."""
    init_db()
    conv_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    conn = _get_connection()
    with conn:
        conn.execute(
            "INSERT INTO conversations (id, uid, title, pinned, created_at, updated_at) VALUES (?, ?, ?, 0, ?, ?)",
            (conv_id, uid, title, now, now)
        )
    conn.close()
    return conv_id


def get_conversations(uid: Optional[str] = None) -> List[Dict[str, Any]]:
    """Returns list of conversations for a user, sorted by pinned (desc) then updated_at (desc)."""
    init_db()
    conn = _get_connection()
    cursor = conn.cursor()
    if uid:
        cursor.execute(
            "SELECT id, uid, title, pinned, created_at, updated_at FROM conversations WHERE uid = ? ORDER BY pinned DESC, updated_at DESC",
            (uid,)
        )
    else:
        cursor.execute(
            "SELECT id, uid, title, pinned, created_at, updated_at FROM conversations ORDER BY pinned DESC, updated_at DESC"
        )
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_conversation(conv_id: str, uid: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieves conversation details by ID, validating user ownership if uid provided."""
    init_db()
    conn = _get_connection()
    cursor = conn.cursor()
    if uid:
        cursor.execute("SELECT id, uid, title, pinned, created_at, updated_at FROM conversations WHERE id = ? AND uid = ?", (conv_id, uid))
    else:
        cursor.execute("SELECT id, uid, title, pinned, created_at, updated_at FROM conversations WHERE id = ?", (conv_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def update_conversation_title(conv_id: str, new_title: str) -> None:
    """Updates the title and updated_at timestamp of a conversation."""
    init_db()
    now = datetime.now(timezone.utc).isoformat()
    conn = _get_connection()
    with conn:
        conn.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
            (new_title, now, conv_id)
        )
    conn.close()


def toggle_pin_conversation(conv_id: str) -> bool:
    """Toggles the pinned status of a conversation. Returns new pinned state."""
    init_db()
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT pinned FROM conversations WHERE id = ?", (conv_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return False
    current_pinned = row[0] or 0
    new_pinned = 0 if current_pinned else 1
    with conn:
        cursor.execute("UPDATE conversations SET pinned = ? WHERE id = ?", (new_pinned, conv_id))
    conn.close()
    return bool(new_pinned)


def touch_conversation(conv_id: str) -> None:
    """Updates the conversation's updated_at timestamp."""
    init_db()
    now = datetime.now(timezone.utc).isoformat()
    conn = _get_connection()
    with conn:
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (now, conv_id)
        )
    conn.close()


def delete_conversation(conv_id: str, uid: Optional[str] = None) -> None:
    """Deletes a conversation and all associated messages."""
    init_db()
    conn = _get_connection()
    with conn:
        if uid:
            conn.execute("DELETE FROM messages WHERE conversation_id IN (SELECT id FROM conversations WHERE id = ? AND uid = ?)", (conv_id, uid))
            conn.execute("DELETE FROM conversations WHERE id = ? AND uid = ?", (conv_id, uid))
        else:
            conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conv_id,))
            conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
    conn.close()


def add_message(
    conv_id: str,
    role: str,
    content: str,
    sources: Optional[List[Dict[str, Any]]] = None,
    xray_data: Optional[Dict[str, Any]] = None
) -> str:
    """Adds a message to a conversation."""
    init_db()
    msg_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    sources_str = json.dumps(sources) if sources else None
    
    xray_str = None
    if xray_data:
        try:
            xray_str = json.dumps(xray_data)
        except Exception:
            xray_str = json.dumps({"note": "X-Ray data present"})

    conn = _get_connection()
    with conn:
        conn.execute(
            """INSERT INTO messages 
               (id, conversation_id, role, content, sources_json, xray_data_json, timestamp, created_at) 
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (msg_id, conv_id, role, content, sources_str, xray_str, now, now)
        )
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (now, conv_id)
        )
    conn.close()
    return msg_id


def get_messages(conv_id: str) -> List[Dict[str, Any]]:
    """Retrieves all messages for a given conversation in chronological order."""
    init_db()
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """SELECT id, conversation_id, role, content, sources_json, xray_data_json, timestamp 
           FROM messages WHERE conversation_id = ? ORDER BY rowid ASC""",
        (conv_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    
    messages = []
    for r in rows:
        d = dict(r)
        d["sources"] = json.loads(d["sources_json"]) if d.get("sources_json") else []
        d["xray_data"] = json.loads(d["xray_data_json"]) if d.get("xray_data_json") else None
        messages.append(d)
    return messages


def delete_messages_from(conv_id: str, from_msg_id: str, include_self: bool = True) -> None:
    """Deletes messages starting from a specific message ID onwards (for edit / regenerate)."""
    init_db()
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT rowid FROM messages WHERE conversation_id = ? AND id = ?",
        (conv_id, from_msg_id)
    )
    row = cursor.fetchone()
    if row:
        target_rowid = row[0]
        with conn:
            op = ">=" if include_self else ">"
            conn.execute(
                f"DELETE FROM messages WHERE conversation_id = ? AND rowid {op} ?",
                (conv_id, target_rowid)
            )
    conn.close()


# ==============================================================================
# PER-USER DOCUMENT METADATA
# ==============================================================================

def add_user_document(uid: str, filename: str, pages: int, chunks: int) -> str:
    """Records an ingested document for a specific user."""
    init_db()
    doc_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    conn = _get_connection()
    with conn:
        conn.execute("""
            INSERT INTO documents (doc_id, uid, filename, pages, chunks, uploaded_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (doc_id, uid, filename, pages, chunks, now))
    conn.close()
    return doc_id


def get_user_documents(uid: str) -> List[Dict[str, Any]]:
    """Retrieves all indexed document records belonging to a user."""
    init_db()
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT doc_id, uid, filename, pages, chunks, uploaded_at 
        FROM documents 
        WHERE uid = ? 
        ORDER BY uploaded_at DESC
    """, (uid,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_user_document(uid: str, doc_id: str) -> Optional[str]:
    """Deletes a document record for a user. Returns the filename deleted."""
    init_db()
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT filename FROM documents WHERE doc_id = ? AND uid = ?", (doc_id, uid))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return None
    filename = row[0]
    with conn:
        cursor.execute("DELETE FROM documents WHERE doc_id = ? AND uid = ?", (doc_id, uid))
    conn.close()
    return filename
