#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ $# -eq 0 ]; then
    echo "Usage: $0 <command> [args...]"
    echo "Example: $0 python trading/main.py"
    exit 1
fi

cleanup() {
    echo "Stopping containers..."
    docker compose down
}

trap cleanup EXIT

echo "Starting containers..."
docker compose up -d

echo "Waiting for services..."
sleep 5

echo "Running: $@"
docker compose exec -T "$@"
