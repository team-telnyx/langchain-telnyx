"""Unit tests for ChatTelnyx."""

import os
from unittest.mock import patch

import pytest
from langchain_core.utils import convert_to_secret_str
from pydantic import SecretStr

from langchain_telnyx.chat_models import ChatTelnyx, DEFAULT_API_BASE, DEFAULT_MODEL


def test_chat_telnyx_default_model() -> None:
    """Test default model name."""
    with patch.dict(os.environ, {"TELNYX_API_KEY": "test-key"}):
        chat = ChatTelnyx()
        assert chat.model_name == DEFAULT_MODEL


def test_chat_telnyx_custom_model() -> None:
    """Test custom model name."""
    with patch.dict(os.environ, {"TELNYX_API_KEY": "test-key"}):
        chat = ChatTelnyx(model="custom-model")
        assert chat.model_name == "custom-model"


def test_chat_telnyx_default_api_base() -> None:
    """Test default API base URL."""
    with patch.dict(os.environ, {"TELNYX_API_KEY": "test-key"}):
        chat = ChatTelnyx()
        assert chat.telnyx_api_base == DEFAULT_API_BASE


def test_chat_telnyx_custom_api_base() -> None:
    """Test custom API base URL."""
    with patch.dict(os.environ, {"TELNYX_API_KEY": "test-key"}):
        chat = ChatTelnyx(
            telnyx_api_base="https://custom.api.com/v1"
        )
        assert chat.telnyx_api_base == "https://custom.api.com/v1"


def test_chat_telnyx_api_key_from_env() -> None:
    """Test API key from environment variable."""
    with patch.dict(os.environ, {"TELNYX_API_KEY": "env-test-key"}):
        chat = ChatTelnyx()
        assert (
            chat.telnyx_api_key.get_secret_value()
            == "env-test-key"
        )


def test_chat_telnyx_api_key_as_secret_str() -> None:
    """Test API key is stored as SecretStr."""
    with patch.dict(os.environ, {"TELNYX_API_KEY": "test-key"}):
        chat = ChatTelnyx()
        assert isinstance(chat.telnyx_api_key, SecretStr)


def test_chat_telnyx_missing_api_key() -> None:
    """Test that missing API key raises error."""
    with patch.dict(os.environ, {}, clear=True):
        # Remove TELNYX_API_KEY if it exists
        os.environ.pop("TELNYX_API_KEY", None)
        with pytest.raises(ValueError, match="TELNYX_API_KEY"):
            ChatTelnyx()


def test_chat_telnyx_llm_type() -> None:
    """Test _llm_type property."""
    with patch.dict(os.environ, {"TELNYX_API_KEY": "test-key"}):
        chat = ChatTelnyx()
        assert chat._llm_type == "telnyx-chat"


def test_chat_telnyx_tiktoken_disabled() -> None:
    """Test tiktoken is disabled by default."""
    with patch.dict(os.environ, {"TELNYX_API_KEY": "test-key"}):
        chat = ChatTelnyx()
        assert chat.tiktoken_enabled is False


def test_chat_telnyx_api_base_from_env() -> None:
    """Test API base from environment variable."""
    with patch.dict(
        os.environ,
        {
            "TELNYX_API_KEY": "test-key",
            "TELNYX_API_BASE": "https://env.api.com/v1",
        },
    ):
        chat = ChatTelnyx()
        assert chat.telnyx_api_base == "https://env.api.com/v1"
