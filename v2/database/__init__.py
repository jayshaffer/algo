"""Database access layer — shared connection, separate query namespaces."""

from .connection import get_connection, get_cursor

__all__ = ["get_connection", "get_cursor"]
