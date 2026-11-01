"""
Database module for LocalGPT.
Manages persistent storage of conversations and message history using SQLite.
"""
import sqlite3
import os
import json
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "conversations")
DB_PATH = os.path.join(DB_DIR, "localgpt.db")


def _get_connection() -> sqlite3.Connection:
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Initialize database tables if they do not exist."""
    conn = _get_connection()
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sources_json TEXT,
                xray_data_json TEXT,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            )
        """)
    conn.close()


def create_conversation(title: str = "New Chat") -> str:
    """Creates a new conversation and returns its ID."""
    init_db()
    conv_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    conn = _get_connection()
    with conn:
        conn.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (conv_id, title, now, now)
        )
    conn.close()
    return conv_id


def get_conversations() -> List[Dict[str, Any]]:
    """Returns list of conversations sorted by latest update."""
    init_db()
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_conversation(conv_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves conversation details by ID."""
    init_db()
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?", (conv_id,))
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


def delete_conversation(conv_id: str) -> None:
    """Deletes a conversation and all associated messages."""
    init_db()
    conn = _get_connection()
    with conn:
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
    
    # Store lightweight X-Ray summary or keys if large
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
               (id, conversation_id, role, content, sources_json, xray_data_json, timestamp) 
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (msg_id, conv_id, role, content, sources_str, xray_str, now)
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

