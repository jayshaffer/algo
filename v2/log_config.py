"""Centralized logging configuration for the trading platform."""

import logging
import os
from logging.handlers import RotatingFileHandler

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

# Loggers that get their own file handler
FILE_LOGGERS = ["trader", "pipeline", "session", "v2.ideation_claude", "v2.tools", "v2.claude_client"]

# Sentinel attribute on handlers we install. Lets us detect re-entry without
# being confused by handlers a third-party library installed.
_ALGO_HANDLER_TAG = "_algo_owned"


def _has_algo_handler(logger: logging.Logger) -> bool:
    return any(getattr(h, _ALGO_HANDLER_TAG, False) for h in logger.handlers)


def setup_logging(level=logging.INFO, log_dir="logs"):
    """Configure logging for the trading platform.

    Idempotent and tolerant of third-party libraries that may have installed
    handlers before us — we tag our own handlers and only short-circuit when
    we see *our* tag, not any handler. Otherwise an unrelated library
    handler would silently disable file logging and skip log-dir creation.
    """
    root = logging.getLogger()
    root.setLevel(level)

    # Console handler — install once; tolerate other libs' handlers nearby.
    if not _has_algo_handler(root):
        console = logging.StreamHandler()
        console.setLevel(level)
        console.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT))
        setattr(console, _ALGO_HANDLER_TAG, True)
        root.addHandler(console)

    # File handlers per named logger — also idempotent.
    os.makedirs(log_dir, exist_ok=True)
    for name in FILE_LOGGERS:
        named = logging.getLogger(name)
        if _has_algo_handler(named):
            continue
        log_file = os.path.join(log_dir, f"{name.replace('.', '_')}.log")
        handler = RotatingFileHandler(
            log_file, maxBytes=5 * 1024 * 1024, backupCount=3
        )
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT))
        setattr(handler, _ALGO_HANDLER_TAG, True)
        named.addHandler(handler)
