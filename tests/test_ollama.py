"""Tests for trading/ollama.py - Ollama client for embeddings and LLM inference."""

import json
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from trading.ollama import (
    get_ollama_url,
    embed,
    embed_batch,
    chat,
    chat_json,
    cosine_similarity,
    cosine_similarity_batch,
    check_ollama_health,
    list_models,
)


# ---------------------------------------------------------------------------
# get_ollama_url
# ---------------------------------------------------------------------------

class TestGetOllamaUrl:
    """Tests for get_ollama_url()."""

    def test_returns_env_var_when_set(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_URL", "http://custom:9999")
        assert get_ollama_url() == "http://custom:9999"

    def test_returns_default_when_env_not_set(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_URL", raising=False)
        assert get_ollama_url() == "http://localhost:11434"

    def test_uses_ollama_env_fixture(self, ollama_env):
        assert get_ollama_url() == "http://localhost:11434"


# ---------------------------------------------------------------------------
# embed
# ---------------------------------------------------------------------------

class TestEmbed:
    """Tests for embed() - single text embedding."""

    @patch("trading.ollama.httpx.post")
    def test_returns_embedding_vector(self, mock_post, ollama_env):
        expected = [0.1, 0.2, 0.3]
        mock_response = MagicMock()
        mock_response.json.return_value = {"embedding": expected}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        result = embed("hello world")
        assert result == expected

    @patch("trading.ollama.httpx.post")
    def test_posts_to_correct_endpoint(self, mock_post, ollama_env):
        mock_response = MagicMock()
        mock_response.json.return_value = {"embedding": [0.0]}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        embed("test text", model="custom-model")

        call_args = mock_post.call_args
        assert "/api/embeddings" in call_args[0][0]
        payload = call_args[1]["json"]
        assert payload["model"] == "custom-model"
        assert payload["prompt"] == "test text"

    @patch("trading.ollama.httpx.post")
    def test_uses_default_model(self, mock_post, ollama_env):
        mock_response = MagicMock()
        mock_response.json.return_value = {"embedding": [0.0]}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        embed("test")

        payload = mock_post.call_args[1]["json"]
        assert payload["model"] == "nomic-embed-text"

    @patch("trading.ollama.httpx.post")
    def test_raises_on_http_error(self, mock_post, ollama_env):
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("HTTP 500")
        mock_post.return_value = mock_response

        with pytest.raises(Exception, match="HTTP 500"):
            embed("fail")


# ---------------------------------------------------------------------------
# embed_batch
# ---------------------------------------------------------------------------

class TestEmbedBatch:
    """Tests for embed_batch() - batch text embedding."""

    def test_empty_input_returns_empty_list(self, ollama_env):
        result = embed_batch([])
        assert result == []

    @patch("trading.ollama.httpx.post")
    def test_returns_list_of_embeddings(self, mock_post, ollama_env):
        expected = [[0.1, 0.2], [0.3, 0.4]]
        mock_response = MagicMock()
        mock_response.json.return_value = {"embeddings": expected}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        result = embed_batch(["hello", "world"])
        assert result == expected

    @patch("trading.ollama.httpx.post")
    def test_posts_to_embed_endpoint(self, mock_post, ollama_env):
        mock_response = MagicMock()
        mock_response.json.return_value = {"embeddings": [[0.0]]}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        embed_batch(["text1", "text2"], model="custom-model")

        call_args = mock_post.call_args
        assert "/api/embed" in call_args[0][0]
        payload = call_args[1]["json"]
        assert payload["model"] == "custom-model"
        assert payload["input"] == ["text1", "text2"]

    @patch("trading.ollama.httpx.post")
    def test_single_text_batch(self, mock_post, ollama_env):
        mock_response = MagicMock()
        mock_response.json.return_value = {"embeddings": [[0.5, 0.6]]}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        result = embed_batch(["only one"])
        assert len(result) == 1


# ---------------------------------------------------------------------------
# chat
# ---------------------------------------------------------------------------

class TestChat:
    """Tests for chat() - LLM chat completion."""

    @patch("trading.ollama.httpx.post")
    def test_returns_content_string(self, mock_post, ollama_env):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {"content": "Hello there!"}
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        result = chat("Hi")
        assert result == "Hello there!"

    @patch("trading.ollama.httpx.post")
    def test_sends_user_message_only_without_system(self, mock_post, ollama_env):
        mock_response = MagicMock()
        mock_response.json.return_value = {"message": {"content": "ok"}}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        chat("Hello")

        payload = mock_post.call_args[1]["json"]
        assert len(payload["messages"]) == 1
        assert payload["messages"][0]["role"] == "user"
        assert payload["messages"][0]["content"] == "Hello"

    @patch("trading.ollama.httpx.post")
    def test_includes_system_message_when_provided(self, mock_post, ollama_env):
        mock_response = MagicMock()
        mock_response.json.return_value = {"message": {"content": "ok"}}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        chat("Hello", system="Be helpful")

        payload = mock_post.call_args[1]["json"]
        assert len(payload["messages"]) == 2
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][0]["content"] == "Be helpful"
        assert payload["messages"][1]["role"] == "user"

    @patch("trading.ollama.httpx.post")
    def test_passes_model_and_temperature(self, mock_post, ollama_env):
        mock_response = MagicMock()
        mock_response.json.return_value = {"message": {"content": "ok"}}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        chat("Hello", model="llama3", temperature=0.7)

        payload = mock_post.call_args[1]["json"]
        assert payload["model"] == "llama3"
        assert payload["options"]["temperature"] == 0.7
        assert payload["stream"] is False

    @patch("trading.ollama.httpx.post")
    def test_uses_default_model(self, mock_post, ollama_env):
        mock_response = MagicMock()
        mock_response.json.return_value = {"message": {"content": "ok"}}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        chat("Hello")

        payload = mock_post.call_args[1]["json"]
        assert payload["model"] == "qwen3:14b"


# ---------------------------------------------------------------------------
# chat_json
# ---------------------------------------------------------------------------

class TestChatJson:
    """Tests for chat_json() - JSON-parsed chat completion."""

    @patch("trading.ollama.chat")
    def test_parses_clean_json(self, mock_chat):
        mock_chat.return_value = '{"action": "buy", "ticker": "AAPL"}'
        result = chat_json("Give me JSON")
        assert result == {"action": "buy", "ticker": "AAPL"}

    @patch("trading.ollama.chat")
    def test_handles_json_code_block(self, mock_chat):
        mock_chat.return_value = '```json\n{"key": "value"}\n```'
        result = chat_json("Give me JSON")
        assert result == {"key": "value"}

    @patch("trading.ollama.chat")
    def test_handles_generic_code_block(self, mock_chat):
        mock_chat.return_value = '```\n{"key": "value"}\n```'
        result = chat_json("Give me JSON")
        assert result == {"key": "value"}

    @patch("trading.ollama.chat")
    def test_raises_valueerror_on_invalid_json(self, mock_chat):
        mock_chat.return_value = "This is not JSON at all"
        with pytest.raises(ValueError, match="Failed to parse JSON"):
            chat_json("Give me JSON")

    @patch("trading.ollama.chat")
    def test_handles_whitespace_around_json(self, mock_chat):
        mock_chat.return_value = '  \n  {"x": 1}  \n  '
        result = chat_json("Give me JSON")
        assert result == {"x": 1}

    @patch("trading.ollama.chat")
    def test_passes_temperature_zero(self, mock_chat):
        mock_chat.return_value = '{"a": 1}'
        chat_json("prompt", model="test-model", system="system prompt")
        mock_chat.assert_called_once_with(
            "prompt", "test-model", "system prompt", temperature=0.0
        )

    @patch("trading.ollama.chat")
    def test_handles_json_array_response(self, mock_chat):
        mock_chat.return_value = '[1, 2, 3]'
        result = chat_json("Give me JSON")
        assert result == [1, 2, 3]

    @patch("trading.ollama.chat")
    def test_handles_nested_json(self, mock_chat):
        data = {"outer": {"inner": [1, 2], "key": "val"}}
        mock_chat.return_value = json.dumps(data)
        result = chat_json("Give me JSON")
        assert result == data


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    """Tests for cosine_similarity() - pairwise cosine similarity."""

    def test_identical_vectors(self):
        vec = [1.0, 2.0, 3.0]
        assert cosine_similarity(vec, vec) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector_a_returns_zero(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0

    def test_zero_vector_b_returns_zero(self):
        assert cosine_similarity([1.0, 2.0], [0.0, 0.0]) == 0.0

    def test_both_zero_vectors(self):
        assert cosine_similarity([0.0, 0.0], [0.0, 0.0]) == 0.0

    def test_returns_float(self):
        result = cosine_similarity([1.0, 0.0], [1.0, 0.0])
        assert isinstance(result, float)

    @pytest.mark.parametrize("a,b,expected", [
        ([1, 0], [1, 0], 1.0),
        ([1, 1], [1, 1], 1.0),
        ([3, 4], [4, 3], 0.96),
    ])
    def test_known_similarities(self, a, b, expected):
        assert cosine_similarity(a, b) == pytest.approx(expected, abs=0.01)


# ---------------------------------------------------------------------------
# cosine_similarity_batch
# ---------------------------------------------------------------------------

class TestCosineSimilarityBatch:
    """Tests for cosine_similarity_batch() - vectorized batch similarity."""

    def test_empty_embeddings_returns_empty(self):
        assert cosine_similarity_batch([1.0, 0.0], []) == []

    def test_zero_query_returns_zeros(self):
        embeddings = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        result = cosine_similarity_batch([0.0, 0.0], embeddings)
        assert result == [0.0, 0.0, 0.0]

    def test_single_embedding(self):
        query = [1.0, 0.0]
        embeddings = [[1.0, 0.0]]
        result = cosine_similarity_batch(query, embeddings)
        assert len(result) == 1
        assert result[0] == pytest.approx(1.0)

    def test_multiple_embeddings(self):
        query = [1.0, 0.0]
        embeddings = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
        result = cosine_similarity_batch(query, embeddings)
        assert len(result) == 3
        assert result[0] == pytest.approx(1.0)
        assert result[1] == pytest.approx(0.0)
        assert result[2] == pytest.approx(-1.0)

    def test_returns_list_of_floats(self):
        query = [1.0, 2.0]
        embeddings = [[3.0, 4.0], [5.0, 6.0]]
        result = cosine_similarity_batch(query, embeddings)
        assert isinstance(result, list)
        for v in result:
            assert isinstance(v, float)

    def test_handles_zero_embedding_in_batch(self):
        query = [1.0, 0.0]
        embeddings = [[0.0, 0.0], [1.0, 0.0]]
        result = cosine_similarity_batch(query, embeddings)
        assert len(result) == 2
        # Zero embedding uses norm=1 to avoid division by zero
        assert result[1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# check_ollama_health
# ---------------------------------------------------------------------------

class TestCheckOllamaHealth:
    """Tests for check_ollama_health()."""

    @patch("trading.ollama.httpx.get")
    def test_returns_true_on_200(self, mock_get, ollama_env):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        assert check_ollama_health() is True

    @patch("trading.ollama.httpx.get")
    def test_returns_false_on_non_200(self, mock_get, ollama_env):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response

        assert check_ollama_health() is False

    @patch("trading.ollama.httpx.get")
    def test_returns_false_on_connection_error(self, mock_get, ollama_env):
        mock_get.side_effect = ConnectionError("Connection refused")
        assert check_ollama_health() is False

    @patch("trading.ollama.httpx.get")
    def test_returns_false_on_timeout(self, mock_get, ollama_env):
        mock_get.side_effect = TimeoutError("Timed out")
        assert check_ollama_health() is False

    @patch("trading.ollama.httpx.get")
    def test_calls_correct_endpoint(self, mock_get, ollama_env):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        check_ollama_health()

        call_args = mock_get.call_args
        assert "/api/tags" in call_args[0][0]


# ---------------------------------------------------------------------------
# list_models
# ---------------------------------------------------------------------------

class TestListModels:
    """Tests for list_models()."""

    @patch("trading.ollama.httpx.get")
    def test_returns_model_names(self, mock_get, ollama_env):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "models": [
                {"name": "qwen3:14b"},
                {"name": "nomic-embed-text"},
                {"name": "llama3:8b"},
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = list_models()
        assert result == ["qwen3:14b", "nomic-embed-text", "llama3:8b"]

    @patch("trading.ollama.httpx.get")
    def test_returns_empty_list_when_no_models(self, mock_get, ollama_env):
        mock_response = MagicMock()
        mock_response.json.return_value = {"models": []}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = list_models()
        assert result == []

    @patch("trading.ollama.httpx.get")
    def test_returns_empty_when_models_key_missing(self, mock_get, ollama_env):
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = list_models()
        assert result == []

    @patch("trading.ollama.httpx.get")
    def test_raises_on_http_error(self, mock_get, ollama_env):
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("HTTP 500")
        mock_get.return_value = mock_response

        with pytest.raises(Exception, match="HTTP 500"):
            list_models()
