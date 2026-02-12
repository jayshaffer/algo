"""Shared database connection factory."""

import os
from contextlib import contextmanager

import psycopg2
from psycopg2.extras import RealDictCursor


def get_connection():
    """Create database connection from DATABASE_URL."""
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL must be set")
    return psycopg2.connect(database_url)


@contextmanager
def get_cursor():
    """Context manager for database cursor with auto-commit/rollback."""
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            yield cur
            conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
