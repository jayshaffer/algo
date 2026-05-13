"""Tests for v2/log_config.py — handler installation idempotency and
tolerance of third-party root handlers."""

import logging
import os
import tempfile

import pytest

from v2.log_config import _ALGO_HANDLER_TAG, FILE_LOGGERS, setup_logging


@pytest.fixture
def isolated_logging():
    """Snapshot + restore root and FILE_LOGGERS handlers around each test
    so `setup_logging` runs against a known-clean baseline."""
    root = logging.getLogger()
    saved_root = list(root.handlers)
    saved_named = {n: list(logging.getLogger(n).handlers) for n in FILE_LOGGERS}
    try:
        root.handlers.clear()
        for n in FILE_LOGGERS:
            logging.getLogger(n).handlers.clear()
        yield
    finally:
        root.handlers[:] = saved_root
        for n, hs in saved_named.items():
            logging.getLogger(n).handlers[:] = hs


def test_installs_console_handler_on_first_call(isolated_logging):
    with tempfile.TemporaryDirectory() as tmp:
        setup_logging(log_dir=tmp)

    root = logging.getLogger()
    algo_handlers = [h for h in root.handlers if getattr(h, _ALGO_HANDLER_TAG, False)]
    assert len(algo_handlers) == 1, "Expected exactly one algo-owned root handler"


def test_creates_log_dir_even_when_third_party_added_root_handler(isolated_logging):
    """P3.35: a pre-existing root handler from another library used to make
    setup_logging() return early — log dir was never created and per-file
    handlers never installed. Now we tag our handlers and only skip if we
    see our own tag."""
    third_party = logging.StreamHandler()
    logging.getLogger().addHandler(third_party)

    with tempfile.TemporaryDirectory() as tmp_parent:
        log_dir = os.path.join(tmp_parent, "fresh_log_dir_does_not_yet_exist")
        assert not os.path.exists(log_dir)

        setup_logging(log_dir=log_dir)

        assert os.path.isdir(log_dir), (
            "setup_logging should create log_dir even when a third-party "
            "library has already installed a root handler"
        )

    # Per-file handlers should be on each named logger.
    for n in FILE_LOGGERS:
        named = logging.getLogger(n)
        algo_handlers = [
            h for h in named.handlers
            if getattr(h, _ALGO_HANDLER_TAG, False)
        ]
        assert len(algo_handlers) >= 1, (
            f"Logger {n!r} should have an algo file handler"
        )


def test_idempotent_does_not_double_add_handlers(isolated_logging):
    """Calling setup_logging twice must not duplicate handlers — that
    would emit each log line twice."""
    with tempfile.TemporaryDirectory() as tmp:
        setup_logging(log_dir=tmp)
        setup_logging(log_dir=tmp)

    root = logging.getLogger()
    algo_root = [h for h in root.handlers if getattr(h, _ALGO_HANDLER_TAG, False)]
    assert len(algo_root) == 1

    for n in FILE_LOGGERS:
        named = logging.getLogger(n)
        algo = [h for h in named.handlers if getattr(h, _ALGO_HANDLER_TAG, False)]
        assert len(algo) == 1, (
            f"Logger {n!r} should have exactly one algo file handler after "
            f"two setup_logging calls, got {len(algo)}"
        )
