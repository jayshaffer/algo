"""Alpaca Learning Platform - Dashboard (placeholder)"""

from flask import Flask, jsonify

app = Flask(__name__)


@app.route("/")
def index():
    return jsonify({
        "status": "ok",
        "message": "Dashboard placeholder - Phase 4 implementation pending"
    })


@app.route("/health")
def health():
    return jsonify({"status": "healthy"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)
