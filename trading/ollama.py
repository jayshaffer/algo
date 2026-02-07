"""Ollama client for embeddings and LLM inference."""

import os
import json
from typing import Optional

import httpx
import numpy as np


def get_ollama_url() -> str:
    """Get Ollama API URL from environment."""
    return os.environ.get("OLLAMA_URL", "http://localhost:11434")


def embed(text: str, model: str = "nomic-embed-text") -> list[float]:
    """
    Generate embeddings for text using Ollama.

    Args:
        text: Text to embed
        model: Embedding model (default: nomic-embed-text)

    Returns:
        List of floats representing the embedding vector
    """
    url = f"{get_ollama_url()}/api/embeddings"

    response = httpx.post(
        url,
        json={"model": model, "prompt": text},
        timeout=10000.0
    )
    response.raise_for_status()

    return response.json()["embedding"]


def embed_batch(texts: list[str], model: str = "nomic-embed-text") -> list[list[float]]:
    """
    Generate embeddings for multiple texts in a single API call.

    Uses the /api/embed endpoint which accepts batch input,
    avoiding per-item HTTP round trips.

    Args:
        texts: List of texts to embed
        model: Embedding model

    Returns:
        List of embedding vectors
    """
    if not texts:
        return []

    url = f"{get_ollama_url()}/api/embed"

    response = httpx.post(
        url,
        json={"model": model, "input": texts},
        timeout=10000.0
    )
    response.raise_for_status()

    return response.json()["embeddings"]


def chat(
    prompt: str,
    model: str = "qwen2.5:14b",
    system: Optional[str] = None,
    temperature: float = 0.1
) -> str:
    """
    Generate a chat completion using Ollama.

    Args:
        prompt: User prompt
        model: Chat model (default: qwen2.5:14b)
        system: Optional system prompt
        temperature: Sampling temperature (lower = more deterministic)

    Returns:
        Model response text
    """
    url = f"{get_ollama_url()}/api/chat"

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = httpx.post(
        url,
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature}
        },
        timeout=10000.0
    )
    response.raise_for_status()

    return response.json()["message"]["content"]


def chat_json(
    prompt: str,
    model: str = "qwen2.5:14b",
    system: Optional[str] = None
) -> dict:
    """
    Generate a chat completion and parse as JSON.

    Args:
        prompt: User prompt (should request JSON output)
        model: Chat model
        system: Optional system prompt

    Returns:
        Parsed JSON response

    Raises:
        ValueError: If response is not valid JSON
    """
    response = chat(prompt, model, system, temperature=0.0)

    # Try to extract JSON from response (model may include extra text)
    text = response.strip()

    # Handle markdown code blocks
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]

    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON response: {response}") from e


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        a: First embedding vector
        b: Second embedding vector

    Returns:
        Cosine similarity score (0 to 1)
    """
    a_vec = np.asarray(a, dtype=np.float32)
    b_vec = np.asarray(b, dtype=np.float32)
    norm_a = np.linalg.norm(a_vec)
    norm_b = np.linalg.norm(b_vec)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a_vec, b_vec) / (norm_a * norm_b))


def cosine_similarity_batch(query: list[float], embeddings: list[list[float]]) -> list[float]:
    """
    Compute cosine similarity of a query vector against a batch of embeddings.

    Vectorized with numpy - much faster than looping cosine_similarity().

    Args:
        query: Query embedding vector
        embeddings: List of embedding vectors to compare against

    Returns:
        List of similarity scores
    """
    if not embeddings:
        return []

    query_vec = np.asarray(query, dtype=np.float32)
    matrix = np.asarray(embeddings, dtype=np.float32)

    query_norm = np.linalg.norm(query_vec)
    if query_norm == 0:
        return [0.0] * len(embeddings)

    row_norms = np.linalg.norm(matrix, axis=1)
    row_norms = np.where(row_norms == 0, 1.0, row_norms)

    similarities = (matrix @ query_vec) / (row_norms * query_norm)
    return similarities.tolist()


def check_ollama_health() -> bool:
    """Check if Ollama is running and responsive."""
    try:
        url = f"{get_ollama_url()}/api/tags"
        response = httpx.get(url, timeout=5.0)
        return response.status_code == 200
    except Exception:
        return False


def list_models() -> list[str]:
    """List available models in Ollama."""
    url = f"{get_ollama_url()}/api/tags"
    response = httpx.get(url, timeout=10.0)
    response.raise_for_status()

    return [model["name"] for model in response.json().get("models", [])]
