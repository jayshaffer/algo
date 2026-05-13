"""Alpaca Learning Platform - Dashboard"""

from benchmark import (
    compute_alpha,
    enrich_snapshots_with_deposits,
    get_deposit_history,
    get_spy_benchmark,
)
from flask import Flask, abort, jsonify, render_template, request
from queries import (
    close_thesis,
    get_agent_event_types,
    get_current_strategy,
    get_decision,
    get_decision_signal_refs_batch,
    get_decision_signals_full,
    get_decision_stats,
    get_decision_tweets,
    get_equity_curve,
    get_latest_snapshot,
    get_llm_call,
    get_open_orders,
    get_performance_metrics,
    get_playbook_action,
    get_playbook_actions,
    get_positions,
    get_recent_agent_events,
    get_recent_decisions,
    get_recent_macro_signals,
    get_recent_session_costs,
    get_recent_ticker_signals,
    get_recent_tweets,
    get_session,
    get_session_decisions,
    get_session_events,
    get_session_llm_calls,
    get_session_memo,
    get_session_stage_costs,
    get_session_theses_created,
    get_session_tweets,
    get_signal_attribution,
    get_signal_summary,
    get_strategy_memos,
    get_strategy_rules,
    get_theses,
    get_thesis,
    get_thesis_decisions,
    get_thesis_playbook_actions,
    get_thesis_stats,
    get_ticker_attribution,
    get_ticker_decisions,
    get_ticker_open_orders,
    get_ticker_position,
    get_ticker_signals,
    get_ticker_theses,
    get_today_playbook,
    lookup_session_id_by_date,
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
    """Signal attribution dashboard, optionally filtered to one category."""
    category = request.args.get("category") or None
    scores = get_signal_attribution(category=category)
    return render_template(
        "attribution.html",
        scores=scores,
        current_category=category,
    )


@app.route("/signals")
def signals():
    """Market signals page (optionally filtered by category)."""
    category = request.args.get("category") or None
    ticker_signals = get_recent_ticker_signals(days=7, limit=50, category=category)
    macro_signals = get_recent_macro_signals(days=7, limit=20, category=category)
    signal_summary = get_signal_summary(days=7)
    return render_template(
        "signals.html",
        ticker_signals=ticker_signals,
        macro_signals=macro_signals,
        signal_summary=signal_summary,
        current_category=category,
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


@app.route("/decision/<int:decision_id>")
def decision_detail(decision_id):
    """Single decision deep-dive with linked signals, thesis, and session."""
    decision = get_decision(decision_id)
    if not decision:
        abort(404)
    signals = get_decision_signals_full(decision_id)
    tweets = get_decision_tweets(decision_id)
    parent_action = (
        get_playbook_action(decision["playbook_action_id"])
        if decision.get("playbook_action_id")
        else None
    )
    session_id = lookup_session_id_by_date(decision["date"])
    return render_template(
        "decision_detail.html",
        decision=decision,
        signals=signals,
        tweets=tweets,
        parent_action=parent_action,
        session_id=session_id,
    )


@app.route("/thesis/<int:thesis_id>")
def thesis_detail(thesis_id):
    """Single thesis with linked decisions and playbook actions."""
    thesis = get_thesis(thesis_id)
    if not thesis:
        abort(404)
    decisions = get_thesis_decisions(thesis_id)
    actions = get_thesis_playbook_actions(thesis_id)
    origin_session_id = lookup_session_id_by_date(thesis["created_at"].date()) if thesis.get("created_at") else None
    return render_template(
        "thesis_detail.html",
        thesis=thesis,
        decisions=decisions,
        actions=actions,
        origin_session_id=origin_session_id,
    )


@app.route("/ticker/<sym>")
def ticker_overview(sym):
    """Aggregate view of position, theses, decisions, signals for one ticker."""
    sym = sym.upper()
    position = get_ticker_position(sym)
    theses = get_ticker_theses(sym)
    decisions = get_ticker_decisions(sym)
    signals = get_ticker_signals(sym)
    open_orders = get_ticker_open_orders(sym)
    attribution = get_ticker_attribution(sym)
    return render_template(
        "ticker.html",
        sym=sym,
        position=position,
        theses=theses,
        decisions=decisions,
        signals=signals,
        open_orders=open_orders,
        attribution=attribution,
    )


@app.route("/session/<int:session_id>")
def session_detail(session_id):
    """Unified per-session view: stages, events, decisions, theses, tweets, memo, LLM calls."""
    session = get_session(session_id)
    if not session:
        abort(404)
    stages = get_session_stage_costs(session_id)
    decisions = get_session_decisions(session_id)
    theses_created = get_session_theses_created(session_id)
    memo = get_session_memo(session_id)
    tweets = get_session_tweets(session_id)
    events = get_session_events(session_id, limit=200)
    llm_calls = get_session_llm_calls(session_id)
    return render_template(
        "session_detail.html",
        session=session,
        stages=stages,
        decisions=decisions,
        theses_created=theses_created,
        memo=memo,
        tweets=tweets,
        events=events,
        llm_calls=llm_calls,
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
    memos_raw = get_strategy_memos(days=30)
    memos = []
    for m in memos_raw:
        m = dict(m)
        m["session_id"] = lookup_session_id_by_date(m["session_date"])
        memos.append(m)
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


@app.route("/costs")
def costs_page():
    """Per-session token usage + USD cost overview."""
    sessions = get_recent_session_costs(limit=30)
    return render_template("costs.html", sessions=sessions)


@app.route("/costs/<int:session_id>")
def costs_session_page(session_id):
    """Per-stage cost breakdown for a single session."""
    stages = get_session_stage_costs(session_id)
    if not stages:
        return "Session not found or no stage cost data", 404
    total_cost = sum(float(s["cost_usd"] or 0) for s in stages)
    return render_template(
        "costs_session.html",
        session_id=session_id,
        stages=stages,
        total_cost=total_cost,
    )


@app.route("/events")
def events_page():
    """Recent agent_events (tool invocations, risk blocks, loop recoveries, ...)."""
    event_type = request.args.get("type") or None
    session_id = request.args.get("session", type=int) or None
    events = get_recent_agent_events(
        limit=200, event_type=event_type, session_id=session_id
    )
    type_counts = get_agent_event_types(days=14)
    return render_template(
        "events.html",
        events=events,
        type_counts=type_counts,
        current_type=event_type,
        current_session=session_id,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=False)
