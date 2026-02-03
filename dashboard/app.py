"""Alpaca Learning Platform - Dashboard"""

from flask import Flask, render_template, jsonify, request

from queries import (
    get_positions,
    get_latest_snapshot,
    get_current_strategy,
    get_recent_ticker_signals,
    get_recent_macro_signals,
    get_signal_summary,
    get_recent_decisions,
    get_decision_stats,
    get_equity_curve,
    get_performance_metrics,
    get_thesis_stats,
    get_theses,
)

app = Flask(__name__)


@app.route("/")
def portfolio():
    """Portfolio overview page."""
    positions = get_positions()
    snapshot = get_latest_snapshot()
    strategy = get_current_strategy()

    return render_template(
        "portfolio.html",
        positions=positions,
        snapshot=snapshot,
        strategy=strategy,
    )


@app.route("/signals")
def signals():
    """Market signals page."""
    ticker_signals = get_recent_ticker_signals(days=7, limit=50)
    macro_signals = get_recent_macro_signals(days=7, limit=20)
    signal_summary = get_signal_summary(days=7)

    return render_template(
        "signals.html",
        ticker_signals=ticker_signals,
        macro_signals=macro_signals,
        signal_summary=signal_summary,
    )


@app.route("/theses")
def theses():
    """Theses page."""
    status_filter = request.args.get('status', 'active')
    sort_by = request.args.get('sort', 'newest')

    stats = get_thesis_stats()
    thesis_list = get_theses(status_filter, sort_by)

    return render_template(
        "theses.html",
        stats=stats,
        theses=thesis_list,
        current_filter=status_filter,
        current_sort=sort_by,
    )


@app.route("/decisions")
def decisions():
    """Decision history page."""
    recent_decisions = get_recent_decisions(days=30, limit=50)
    stats = get_decision_stats(days=30)

    return render_template(
        "decisions.html",
        decisions=recent_decisions,
        stats=stats,
    )


@app.route("/performance")
def performance():
    """Performance charts page."""
    equity_curve = get_equity_curve(days=90)
    metrics = get_performance_metrics(days=30)

    # Format equity data for Chart.js
    equity_data = [
        {
            "date": str(row["date"]),
            "portfolio_value": float(row["portfolio_value"]),
            "cash": float(row["cash"]),
            "buying_power": float(row["buying_power"]),
        }
        for row in equity_curve
    ] if equity_curve else []

    return render_template(
        "performance.html",
        equity_data=equity_data,
        metrics=metrics,
    )


@app.route("/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy"})


@app.route("/api/portfolio")
def api_portfolio():
    """API endpoint for portfolio data."""
    positions = get_positions()
    snapshot = get_latest_snapshot()

    return jsonify({
        "positions": [dict(p) for p in positions] if positions else [],
        "snapshot": dict(snapshot) if snapshot else None,
    })


@app.route("/api/signals")
def api_signals():
    """API endpoint for signals data."""
    ticker_signals = get_recent_ticker_signals(days=7, limit=50)
    macro_signals = get_recent_macro_signals(days=7, limit=20)

    return jsonify({
        "ticker_signals": [dict(s) for s in ticker_signals] if ticker_signals else [],
        "macro_signals": [dict(s) for s in macro_signals] if macro_signals else [],
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True)
