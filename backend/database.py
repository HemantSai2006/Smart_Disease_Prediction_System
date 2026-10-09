"""
AI-Based Disease Prediction System - Database Module

Handles SQLite storage for user prediction history with model versioning support.
"""

import os
import json
import sqlite3
from datetime import datetime
from typing import List, Dict, Any

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "predictions.db")


def get_db_connection(db_path: str = DEFAULT_DB_PATH):
    """Establishes a connection to the SQLite database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DEFAULT_DB_PATH):
    """
    Initializes the SQLite database with the PredictionHistory table.
    Safely migrates table to add `model_version` without deleting existing records.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS PredictionHistory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symptoms TEXT NOT NULL,
            predicted_disease TEXT NOT NULL,
            model_name TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    
    # Safe migration: add model_version column if missing
    cursor.execute("PRAGMA table_info(PredictionHistory)")
    columns = [col[1] for col in cursor.fetchall()]
    if "model_version" not in columns:
        try:
            cursor.execute("ALTER TABLE PredictionHistory ADD COLUMN model_version TEXT DEFAULT 'v1.0'")
        except Exception:
            pass

    conn.commit()
    conn.close()


def save_prediction(
    symptoms: List[str],
    predicted_disease: str,
    model_name: str,
    model_version: str = "v2.0",
    db_path: str = DEFAULT_DB_PATH
) -> int:
    """
    Saves a new prediction record into SQLite.
    Symptoms are serialized as JSON text with the active model_version.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    now_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    symptoms_json = json.dumps(symptoms)

    cursor.execute("""
        INSERT INTO PredictionHistory (symptoms, predicted_disease, model_name, created_at, model_version)
        VALUES (?, ?, ?, ?, ?)
    """, (symptoms_json, predicted_disease, model_name, now_iso, model_version))

    record_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return record_id


def get_history(limit: int = 50, db_path: str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """
    Retrieves recent predictions from the database, newest first.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, symptoms, predicted_disease, model_name, created_at, 
               COALESCE(model_version, 'v1.0') AS model_version
        FROM PredictionHistory
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))

    rows = cursor.fetchall()
    history = []
    for row in rows:
        try:
            symptoms_list = json.loads(row["symptoms"])
        except Exception:
            symptoms_list = [row["symptoms"]]

        history.append({
            "id": row["id"],
            "symptoms": symptoms_list,
            "predicted_disease": row["predicted_disease"],
            "model_name": row["model_name"],
            "model_version": row["model_version"],
            "created_at": row["created_at"]
        })

    conn.close()
    return history


if __name__ == "__main__":
    init_db()
    print("Database initialized and verified successfully at:", DEFAULT_DB_PATH)
