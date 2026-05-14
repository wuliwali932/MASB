import json
import os
import sqlite3
from typing import Dict, List, Optional

import bcrypt


# Config: fixed DB path inside the package directory (auth.db). No env var override.
_DB_PATH = os.path.join(os.path.dirname(__file__), "auth.db")
_conn: Optional[sqlite3.Connection] = None


def init_db(db_path: Optional[str] = None):
    """Create (or re-open) the sqlite database and users table."""
    global _DB_PATH, _conn
    if db_path:
        _DB_PATH = db_path
    if _conn:
        try:
            _conn.close()
        except Exception:
            pass
        _conn = None

    if _DB_PATH != ":memory:" and os.path.dirname(_DB_PATH):
        os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)

    _conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
    _conn.row_factory = sqlite3.Row
    cur = _conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            hashed_password TEXT NOT NULL,
            full_name TEXT,
            disabled INTEGER DEFAULT 0,
            role TEXT CHECK(role IN ('patient', 'physician', 'administrator')) NOT NULL
        )
        """
    )
    _conn.commit()

    # Migration: if an older DB existed without the `role` column, add it.
    # SQLite's CREATE TABLE IF NOT EXISTS won't modify existing tables, so
    # we detect the absence of `role` and add the column safely.
    cur.execute("PRAGMA table_info('users')")
    cols = [r[1] for r in cur.fetchall()]
    if 'role' not in cols:
        # Add the column without strict CHECK constraint (can't alter to add CHECK in SQLite).
        cur.execute("ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'patient'")
        # Backfill existing rows with a sensible default (patient).
        cur.execute("UPDATE users SET role = 'patient' WHERE role IS NULL OR role = ''")
        _conn.commit()


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        init_db(None)
    # After init_db, _conn must be set. Assert for type-checkers and safety.
    assert _conn is not None
    return _conn


def get_password_hash(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
    except Exception:
        return False


def create_user(username: str, password: str, role: str = "patient", full_name: Optional[str] = None, disabled: bool = False):
    conn = _get_conn()
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO users (username, hashed_password, full_name, disabled, role) VALUES (?, ?, ?, ?, ?)",
            (username, get_password_hash(password), full_name or username, 1 if disabled else 0, role),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        # already exists
        pass


def get_user(username: str) -> Optional[Dict]:
    cur = _get_conn().cursor()
    cur.execute("SELECT username, hashed_password, full_name, disabled, role FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    if not row:
        return None
    return {
        "username": row["username"],
        "hashed_password": row["hashed_password"],
        "full_name": row["full_name"],
        "disabled": bool(row["disabled"]),
        "role": row["role"],
    }


def get_all_users() -> List[Dict]:
    cur = _get_conn().cursor()
    cur.execute("SELECT username, full_name, disabled, role FROM users ORDER BY username")
    rows = cur.fetchall()
    return [{"username": r["username"], "full_name": r["full_name"], "disabled": bool(r["disabled"]), "role": r["role"]} for r in rows]


def load_predefined_users(file_path: Optional[str] = None, clear_db: bool = True):
    """Load users from JSON file and optionally clear DB first.

    JSON format: [{"username":"alice","password":"secret","full_name":"Alice","disabled":false}, ...]
    """
    if file_path is None:
        file_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "bench_data",
            "agentic_data_tool",
            "predefined_users.json",
        )
    if not os.path.exists(file_path):
        return
    conn = _get_conn()
    cur = conn.cursor()
    if clear_db:
        cur.execute("DELETE FROM users")
        conn.commit()
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for u in data:
        username = u.get("username")
        password = u.get("password")
        if username and password:
            create_user(username, password, role=u.get("role", "patient"), full_name=u.get("full_name"), disabled=bool(u.get("disabled", False)))
