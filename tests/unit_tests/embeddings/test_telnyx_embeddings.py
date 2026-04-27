"""Unit tests for TelnyxEmbeddings."""

import os
from unittest.mock import patch

import pytest
from pydantic import SecretStr

from langchain_telnyx.embeddings import (
    DEFAULT_API_BASE,
    DEFAULT_MODEL,
    TelnyxEmbeddings,
)


def test_telnyx_embeddings_default_model() -> None:
    """Test default embedding model name."""
    with patch.dict(os.environ, {"TELNYX_API_KEY": "test-key"}):
        embeddings = TelnyxEmbeddings()
        assert embeddings.model == DEFAULT_MODEL


def test_telnyx_embeddings_custom_model() -> None:
    """Test custom embedding model name."""
    with patch.dict(os.environ, {"TELNYX_API_KEY": "test-key"}):
        embeddings = TelnyxEmbeddings(model="custom-embedding-model")
        assert embeddings.model == "custom-embedding-model"


def test_telnyx_embeddings_default_api_base() -> None:
    """Test default API base URL."""
    with patch.dict(os.environ, {"TELNYX_API_KEY": "test-key"}):
        embeddings = TelnyxEmbeddings()
        assert embeddings.telnyx_api_base == DEFAULT_API_BASE


def test_telnyx_embeddings_custom_api_base() -> None:
    """Test custom API base URL."""
    with patch.dict(os.environ, {"TELNYX_API_KEY": "test-key"}):
        embeddings = TelnyxEmbeddings(
            telnyx_api_base="https://custom.api.com/v1"
        )
        assert embeddings.telnyx_api_base == "https://custom.api.com/v1"


def test_telnyx_embeddings_api_key_from_env() -> None:
    """Test API key from environment variable."""
    with patch.dict(os.environ, {"TELNYX_API_KEY": "env-test-key"}):
        embeddings = TelnyxEmbeddings()
        assert (
            embeddings.telnyx_api_key.get_secret_value()
            == "env-test-key"
        )


def test_telnyx_embeddings_api_key_as_secret_str() -> None:
    """Test API key is stored as SecretStr."""
    with patch.dict(os.environ, {"TELNYX_API_KEY": "test-key"}):
        embeddings = TelnyxEmbeddings()
        assert isinstance(embeddings.telnyx_api_key, SecretStr)


def test_telnyx_embeddings_missing_api_key() -> None:
    """Test that missing API key raises error."""
    with patch.dict(os.environ, {}, clear=True):
        os.environ.pop("TELNYX_API_KEY", None)
        with pytest.raises(ValueError, match="TELNYX_API_KEY"):
            TelnyxEmbeddings()


def test_telnyx_embeddings_tiktoken_disabled() -> None:
    """Test tiktoken is disabled by default."""
    with patch.dict(os.environ, {"TELNYX_API_KEY": "test-key"}):
        embeddings = TelnyxEmbeddings()
        assert embeddings.tiktoken_enabled is False


def test_telnyx_embeddings_llm_type() -> None:
    """Test _llm_type property."""
    with patch.dict(os.environ, {"TELNYX_API_KEY": "test-key"}):
        embeddings = TelnyxEmbeddings()
        assert embeddings._llm_type == "telnyx-embedding"


def test_telnyx_embeddings_api_base_from_env() -> None:
    """Test API base from environment variable."""
    with patch.dict(
        os.environ,
        {
            "TELNYX_API_KEY": "test-key",
            "TELNYX_API_BASE": "https://env.api.com/v1",
        },
    ):
        embeddings = TelnyxEmbeddings()
        assert embeddings.telnyx_api_base == "https://env.api.com/v1"
