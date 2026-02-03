"""Ollama client for embeddings and LLM inference."""

import os
import json
from typing import Optional

import httpx


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
        timeout=30.0
    )
    response.raise_for_status()

    return response.json()["embedding"]


def embed_batch(texts: list[str], model: str = "nomic-embed-text") -> list[list[float]]:
    """
    Generate embeddings for multiple texts.

    Args:
        texts: List of texts to embed
        model: Embedding model

    Returns:
        List of embedding vectors
    """
    return [embed(text, model) for text in texts]


def chat(
    prompt: str,
    model: str = "phi3:mini",
    system: Optional[str] = None,
    temperature: float = 0.1
) -> str:
    """
    Generate a chat completion using Ollama.

    Args:
        prompt: User prompt
        model: Chat model (default: phi3:mini)
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
        timeout=200.0
    )
    response.raise_for_status()

    return response.json()["message"]["content"]


def chat_json(
    prompt: str,
    model: str = "phi3:mini",
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
    dot_product = sum(x * y for x, y in zip(a, b))
    magnitude_a = sum(x * x for x in a) ** 0.5
    magnitude_b = sum(x * x for x in b) ** 0.5

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


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
