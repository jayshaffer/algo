"""Alpaca Learning Platform - Dashboard"""

from benchmark import (
    compute_alpha,
    enrich_snapshots_with_deposits,
    get_deposit_history,
    get_spy_benchmark,
)
from flask import Flask, jsonify, redirect, render_template, request, url_for
from queries import (
    close_thesis,
    get_audit_finding,
    get_current_strategy,
    get_decision_signal_refs_batch,
    get_decision_stats,
    get_equity_curve,
    get_latest_snapshot,
    get_open_audit_findings,
    get_open_orders,
    get_performance_metrics,
    get_playbook_actions,
    get_positions,
    get_recent_audit_runs,
    get_recent_decisions,
    get_recent_macro_signals,
    get_recent_ticker_signals,
    get_recent_tweets,
    get_signal_attribution,
    get_signal_summary,
    get_strategy_memos,
    get_strategy_rules,
    get_theses,
    get_thesis_stats,
    get_today_playbook,
    update_audit_finding_status,
)

app = Flask(__name__)


# P1.10: dashboard mutating endpoints (close-thesis et al.) rely on docker
# port mapping for network isolation (compose binds to 127.0.0.1:3000). This
# header check is defense-in-depth: blocks cross-origin browser POSTs that
# send form-urlencoded bodies (which don't trigger a CORS preflight) by
# requiring a custom header that cross-origin scripts can't add without one.
def _require_dashboard_origin():
    if request.headers.get("X-Requested-With") != "dashboard":
        return jsonify({"error": "Forbidden"}), 403
    return None


@app.route("/")
def portfolio():
    """Portfolio overview page."""
    positions = get_positions()
    snapshot = get_latest_snapshot()
    playbook = get_today_playbook()
    open_orders = get_open_orders()

    equity_curve = get_equity_curve(days=90)
    benchmark_data = []
    alpha_stats = None
    if equity_curve:
        dates = [row["date"] for row in equity_curve]
        benchmark_data = get_spy_benchmark(dates[0], dates[-1])
        deposits = get_deposit_history()
        enriched = enrich_snapshots_with_deposits(equity_curve, deposits)
        alpha_stats = compute_alpha(enriched, benchmark_data)

    return render_template(
        "portfolio.html",
        positions=positions,
        snapshot=snapshot,
        playbook=playbook,
        open_orders=open_orders,
        alpha_stats=alpha_stats,
    )


@app.route("/playbook")
def playbook():
    """Today's playbook view."""
    today = get_today_playbook()
    actions = get_playbook_actions(today['id']) if today else []
    return render_template("playbook.html", playbook=today, actions=actions)


@app.route("/attribution")
def attribution():
    """Signal attribution dashboard."""
    scores = get_signal_attribution()
    return render_template("attribution.html", scores=scores)


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
    decision_ids = [d['id'] for d in recent_decisions] if recent_decisions else []
    signal_refs = get_decision_signal_refs_batch(decision_ids)

    return render_template(
        "decisions.html",
        decisions=recent_decisions,
        stats=stats,
        signal_refs=signal_refs,
    )


@app.route("/performance")
def performance():
    """Performance charts page."""
    equity_curve = get_equity_curve(days=90)
    metrics = get_performance_metrics(days=30)

    benchmark_data = []
    alpha_stats = None
    enriched = []
    if equity_curve:
        dates = [row["date"] for row in equity_curve]
        benchmark_data = get_spy_benchmark(dates[0], dates[-1])
        deposits = get_deposit_history()
        enriched = enrich_snapshots_with_deposits(equity_curve, deposits)
        alpha_stats = compute_alpha(enriched, benchmark_data)

    equity_data = [
        {
            "date": str(row["date"]),
            "portfolio_value": float(row["portfolio_value"]),
            "cash": float(row["cash"]),
            "buying_power": float(row["buying_power"]),
            "cumulative_deposits": float(row.get("cumulative_deposits", row["portfolio_value"])),
        }
        for row in enriched
    ]

    return render_template(
        "performance.html",
        equity_data=equity_data,
        metrics=metrics,
        benchmark_data=benchmark_data,
        alpha_stats=alpha_stats,
    )


@app.route("/strategy")
def strategy():
    """Strategy state, rules, and memos page."""
    state = get_current_strategy()
    rules = get_strategy_rules(status='active')
    memos = get_strategy_memos(days=30)
    return render_template("strategy.html", state=state, rules=rules, memos=memos)


@app.route("/tweets")
def tweets():
    """Tweets page."""
    tweet_list = get_recent_tweets(days=30, limit=50)
    return render_template("tweets.html", tweets=tweet_list)


@app.route("/api/theses/<int:thesis_id>/close", methods=["POST"])
def api_close_thesis(thesis_id):
    """Close a thesis with status and optional reason."""
    forbidden = _require_dashboard_origin()
    if forbidden:
        return forbidden

    data = request.get_json() or {}
    status = data.get("status")
    reason = data.get("reason", "").strip() or None

    if status not in ("invalidated", "expired"):
        return jsonify({"error": "Invalid status. Must be 'invalidated' or 'expired'."}), 400

    try:
        success = close_thesis(thesis_id, status, reason)
        if success:
            return jsonify({"success": True})
        else:
            return jsonify({"error": "Thesis not found or already closed."}), 404
    except Exception:
        return jsonify({"error": "Internal server error"}), 500


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


@app.route("/audit")
def audit_page():
    findings = get_open_audit_findings()
    runs = get_recent_audit_runs()
    return render_template("audit.html", findings=findings, runs=runs)


@app.route("/audit/findings/<int:finding_id>")
def audit_finding_page(finding_id):
    f = get_audit_finding(finding_id)
    if not f:
        return "Not found", 404
    return render_template("audit_finding.html", finding=f)


@app.route("/audit/findings/<int:finding_id>/status", methods=["POST"])
def audit_finding_status(finding_id):
    status = request.form.get("status")
    note = request.form.get("note")
    update_audit_finding_status(finding_id, status, note)
    return redirect(url_for("audit_page"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=False)
