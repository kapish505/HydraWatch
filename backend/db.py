"""
SQLite alert storage for HydraWatch.

WHY SQLITE?
  Every alert the system fires needs to be persisted so operators can:
  - Review past alerts
  - Compare detection times to actual leak reports
  - Audit false alarms

  SQLite is perfect here: zero-config, file-based, fast for this volume
  (we expect at most a few dozen alerts per day). No need for PostgreSQL
  or any external database server.

Table schema:
  alerts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp     TEXT NOT NULL,        -- when the alert was generated
    severity      TEXT NOT NULL,        -- CRITICAL / MEDIUM / LOW
    suspect_nodes TEXT NOT NULL,        -- JSON array of node names
    confidence    REAL NOT NULL,        -- 0-1 ensemble confidence
    anomaly_score REAL,                 -- LSTM anomaly score
    xgb_prob      REAL,                 -- XGBoost probability
    shap_features TEXT,                 -- JSON array of SHAP explanations
    detected_at   TEXT,                 -- timestamp of detection
    location      TEXT,                 -- human-readable estimated location
    resolved      INTEGER DEFAULT 0    -- 0=active, 1=resolved
  )
"""

import sqlite3
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime


DB_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DB_DIR / "hydrawatch.db"


def get_connection() -> sqlite3.Connection:
    """Get or create database connection."""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row  # Allows dict-like access to rows
    conn.execute("PRAGMA journal_mode=WAL")  # Better concurrent read performance
    return conn


def init_db():
    """Create the alerts table if it doesn't exist."""
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp     TEXT NOT NULL,
            severity      TEXT NOT NULL,
            suspect_nodes TEXT NOT NULL,
            confidence    REAL NOT NULL,
            anomaly_score REAL,
            xgb_prob      REAL,
            shap_features TEXT,
            detected_at   TEXT,
            location      TEXT,
            resolved      INTEGER DEFAULT 0,
            created_at    TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT DEFAULT 'user',
            status TEXT DEFAULT 'active',
            profile_picture_url TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS activity_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            action TEXT NOT NULL,
            timestamp TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)
    conn.commit()
    conn.close()


def insert_alert(alert_data: Dict[str, Any]) -> int:
    """
    Insert a new alert into the database.

    Args:
        alert_data: Dict from EnsembleDetector.predict() with keys:
            severity, suspect_nodes, confidence, anomaly_score,
            shap_features, detected_at, estimated_location

    Returns:
        The ID of the inserted alert
    """
    conn = get_connection()
    cursor = conn.execute(
        """
        INSERT INTO alerts (
            timestamp, severity, suspect_nodes, confidence,
            anomaly_score, xgb_prob, shap_features, detected_at, location
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now().isoformat(),
            alert_data.get("severity", "LOW"),
            json.dumps(alert_data.get("suspect_nodes", [])),
            alert_data.get("confidence", 0.0),
            alert_data.get("anomaly_score", 0.0),
            alert_data.get("xgb_probability", 0.0),
            json.dumps(alert_data.get("shap_features", [])),
            alert_data.get("detected_at", datetime.now().isoformat()),
            alert_data.get("estimated_location", ""),
        ),
    )
    alert_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return alert_id


def get_recent_alerts(limit: int = 50) -> List[Dict[str, Any]]:
    """
    Get the most recent alerts, newest first.

    Args:
        limit: maximum number of alerts to return

    Returns:
        List of alert dicts
    """
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()

    alerts = []
    for row in rows:
        alert = dict(row)
        # Parse JSON fields
        alert["suspect_nodes"] = json.loads(alert.get("suspect_nodes", "[]"))
        alert["shap_features"] = json.loads(alert.get("shap_features", "[]"))
        alerts.append(alert)

    return alerts


def get_alert_by_id(alert_id: int) -> Optional[Dict[str, Any]]:
    """Get a single alert by ID."""
    conn = get_connection()
    row = conn.execute("SELECT * FROM alerts WHERE id = ?", (alert_id,)).fetchone()
    conn.close()

    if row is None:
        return None

    alert = dict(row)
    alert["suspect_nodes"] = json.loads(alert.get("suspect_nodes", "[]"))
    alert["shap_features"] = json.loads(alert.get("shap_features", "[]"))
    return alert


def resolve_alert(alert_id: int) -> bool:
    """Mark an alert as resolved."""
    conn = get_connection()
    cursor = conn.execute(
        "UPDATE alerts SET resolved = 1 WHERE id = ?", (alert_id,)
    )
    conn.commit()
    updated = cursor.rowcount > 0
    conn.close()
    return updated


def get_alert_count() -> Dict[str, int]:
    """Get alert counts by severity."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT severity, COUNT(*) as count FROM alerts WHERE resolved = 0 GROUP BY severity"
    ).fetchall()
    conn.close()

    counts = {"CRITICAL": 0, "MEDIUM": 0, "LOW": 0, "total": 0}
    for row in rows:
        counts[row["severity"]] = row["count"]
        counts["total"] += row["count"]
    return counts


def clear_alerts():
    """Delete all alerts (useful for testing/demo resets)."""
    conn = get_connection()
    conn.execute("DELETE FROM alerts")
    conn.commit()
    conn.close()


# ═══════════════════════════════════════════════════════════════════════════
#  User Management & Activity Logs
# ═══════════════════════════════════════════════════════════════════════════

def create_user(email: str, password_hash: str, role: str = "user") -> int:
    conn = get_connection()
    try:
        cursor = conn.execute(
            "INSERT INTO users (email, password_hash, role) VALUES (?, ?, ?)",
            (email.lower(), password_hash, role)
        )
        user_id = cursor.lastrowid
        conn.commit()
        return user_id
    except sqlite3.IntegrityError:
        conn.close()
        raise ValueError(f"Email {email} already exists")
    finally:
        conn.close()

def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    row = conn.execute("SELECT * FROM users WHERE email = ?", (email.lower(),)).fetchone()
    conn.close()
    return dict(row) if row else None

def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return dict(row) if row else None

def get_all_users(limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, email, role, status, profile_picture_url, created_at FROM users ORDER BY created_at DESC LIMIT ? OFFSET ?", 
        (limit, offset)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def update_user_status(user_id: int, status: str) -> bool:
    conn = get_connection()
    cursor = conn.execute("UPDATE users SET status = ? WHERE id = ?", (status, user_id))
    conn.commit()
    updated = cursor.rowcount > 0
    conn.close()
    return updated

def update_profile_picture(user_id: int, url: str) -> bool:
    conn = get_connection()
    cursor = conn.execute("UPDATE users SET profile_picture_url = ? WHERE id = ?", (url, user_id))
    conn.commit()
    updated = cursor.rowcount > 0
    conn.close()
    return updated

def log_activity(user_id: int, action: str):
    conn = get_connection()
    conn.execute(
        "INSERT INTO activity_logs (user_id, action) VALUES (?, ?)",
        (user_id, action)
    )
    conn.commit()
    conn.close()

def get_activity_logs(limit: int = 50) -> List[Dict[str, Any]]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT l.id, l.user_id, l.action, l.timestamp, u.email "
        "FROM activity_logs l JOIN users u ON l.user_id = u.id "
        "ORDER BY l.timestamp DESC LIMIT ?", 
        (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# Initialize the database on import
init_db()
