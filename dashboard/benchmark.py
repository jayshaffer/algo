"""SPY benchmark data fetching and alpha computation for the local dashboard."""

import logging
import os
import time
from datetime import date, datetime

logger = logging.getLogger(__name__)


def compute_alpha(snapshots, benchmark):
    """Compute portfolio alpha vs SPY benchmark.

    Returns {"portfolio_return", "spy_return", "alpha"} or None.
    """
    if not snapshots or len(snapshots) < 2 or not benchmark:
        return None

    spy_map = {b["date"]: b["close"] for b in benchmark}

    spy_start = None
    port_start = None
    for snap in snapshots:
        date_str = str(snap["date"])
        if date_str in spy_map:
            spy_start = spy_map[date_str]
            port_start = float(snap["portfolio_value"])
            break

    spy_end = None
    port_end = None
    for snap in reversed(snapshots):
        date_str = str(snap["date"])
        if date_str in spy_map:
            spy_end = spy_map[date_str]
            port_end = float(snap["portfolio_value"])
            break

    if spy_start is None or spy_end is None or spy_start == spy_end:
        return None
    if port_start is None or port_start == 0:
        return None

    portfolio_return = ((port_end - port_start) / port_start) * 100
    spy_return = ((spy_end - spy_start) / spy_start) * 100
    alpha = portfolio_return - spy_return

    return {
        "portfolio_return": portfolio_return,
        "spy_return": spy_return,
        "alpha": alpha,
    }
