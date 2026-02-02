#!/bin/bash
# Setup Ollama models for the trading platform
# Run after: docker compose up -d

set -e

echo "Waiting for Ollama to be ready..."
until curl -s http://localhost:11434/api/tags > /dev/null 2>&1; do
    sleep 2
done
echo "Ollama is ready!"

echo ""
echo "Pulling phi3:mini (news classification, ~2GB)..."
docker compose exec ollama ollama pull phi3:mini

echo ""
echo "Pulling nomic-embed-text (embeddings, ~270MB)..."
docker compose exec ollama ollama pull nomic-embed-text

echo ""
echo "Verifying models..."
docker compose exec ollama ollama list

echo ""
echo "Ollama setup complete!"
