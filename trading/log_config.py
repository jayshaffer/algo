"""Centralized logging configuration for the trading platform."""

import logging
import os
from logging.handlers import RotatingFileHandler

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

# Loggers that get their own file handler
FILE_LOGGERS = ["trader", "pipeline", "trading.ideation_claude", "trading.tools"]


def setup_logging(level=logging.INFO, log_dir="logs"):
    """Configure logging for the trading platform.

    - Root logger: console handler at `level` with compact format
    - Named file handlers: one RotatingFileHandler per module in FILE_LOGGERS
      (5 MB max, 3 backups) written to `log_dir/`

    Safe to call multiple times — skips if handlers are already attached.
    """
    root = logging.getLogger()
    if root.handlers:
        return

    root.setLevel(level)

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT))
    root.addHandler(console)

    # File handlers per named logger
    os.makedirs(log_dir, exist_ok=True)
    for name in FILE_LOGGERS:
        log_file = os.path.join(log_dir, f"{name.replace('.', '_')}.log")
        handler = RotatingFileHandler(
            log_file, maxBytes=5 * 1024 * 1024, backupCount=3
        )
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT))
        logging.getLogger(name).addHandler(handler)
