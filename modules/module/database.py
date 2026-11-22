import sqlite3
from pathlib import Path


# database.py is located at K-Fitness/modules/module/database.py.
# parents[2] resolves to the K-Fitness project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE_PATH = PROJECT_ROOT / "kfitness_sessions.db"


def get_connection():
    """Open a connection to the local SQLite database."""
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    """Create the workout sessions table if it does not already exist."""
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT NOT NULL,
                ended_at TEXT NOT NULL,
                duration_seconds REAL NOT NULL,
                exercise TEXT NOT NULL,
                reps INTEGER NOT NULL DEFAULT 0,
                avg_form_score REAL,
                best_form_score INTEGER,
                lowest_form_score INTEGER,
                form_corrections INTEGER NOT NULL DEFAULT 0
            )
        """)
        connection.commit()


def save_session(
    started_at,
    ended_at,
    duration_seconds,
    exercise="Squats",
    reps=0,
    avg_form_score=None,
    best_form_score=None,
    lowest_form_score=None,
    form_corrections=0
):
    """Save one completed workout session and return its database ID."""
    with get_connection() as connection:
        cursor = connection.execute("""
            INSERT INTO sessions (
                started_at,
                ended_at,
                duration_seconds,
                exercise,
                reps,
                avg_form_score,
                best_form_score,
                lowest_form_score,
                form_corrections
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            started_at,
            ended_at,
            duration_seconds,
            exercise,
            reps,
            avg_form_score,
            best_form_score,
            lowest_form_score,
            form_corrections
        ))
        connection.commit()
        return cursor.lastrowid


def get_all_sessions():
    """Return all sessions, newest first."""
    with get_connection() as connection:
        rows = connection.execute("""
            SELECT *
            FROM sessions
            ORDER BY started_at DESC
        """).fetchall()
        return [dict(row) for row in rows]


def get_session_by_id(session_id):
    """Return one session by ID, or None if it does not exist."""
    with get_connection() as connection:
        row = connection.execute("""
            SELECT *
            FROM sessions
            WHERE id = ?
        """, (session_id,)).fetchone()
        return dict(row) if row else None


def get_session_statistics():
    """Return aggregate workout statistics."""
    with get_connection() as connection:
        row = connection.execute("""
            SELECT
                COUNT(*) AS total_sessions,
                COALESCE(SUM(reps), 0) AS total_reps,
                COALESCE(SUM(duration_seconds), 0) AS total_duration,
                AVG(avg_form_score) AS overall_avg_score
            FROM sessions
        """).fetchone()
        return dict(row)


def delete_session(session_id):
    """Delete a session by ID."""
    with get_connection() as connection:
        connection.execute(
            "DELETE FROM sessions WHERE id = ?",
            (session_id,)
        )
        connection.commit()


if __name__ == "__main__":
    initialize_database()
    print("K-Fitness database initialized successfully.")
    print(f"Database location: {DATABASE_PATH}")
