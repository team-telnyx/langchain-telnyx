"""Telnyx Inference embeddings wrapper."""

from __future__ import annotations

from typing import Any

from langchain_core.utils import from_env, secret_from_env
from langchain_openai import OpenAIEmbeddings
from pydantic import Field, SecretStr

DEFAULT_API_BASE = "https://api.telnyx.com/v2/ai/openai"
DEFAULT_MODEL = "thenlper/gte-large"


class TelnyxEmbeddings(OpenAIEmbeddings):
    """`Telnyx` Embeddings API.

    See https://telnyx.com for information about Telnyx.

    To use, you should have the ``openai`` python package installed, and the
    environment variable ``TELNYX_API_KEY`` set with your API key.

    Example:
        .. code-block:: python

            from langchain_telnyx import TelnyxEmbeddings
            embeddings = TelnyxEmbeddings(model="thenlper/gte-large")
    """

    model: str = Field(default=DEFAULT_MODEL)
    """Model name to use."""

    openai_api_base: str | None = Field(
        alias="base_url",
        default_factory=from_env(
            "TELNYX_API_BASE", default=DEFAULT_API_BASE
        ),
    )
    """Base URL path for API requests."""

    openai_api_key: SecretStr | None = Field(
        alias="api_key",
        default_factory=secret_from_env(
            "TELNYX_API_KEY", default=None
        ),
    )
    """Automatically inferred from env var ``TELNYX_API_KEY``."""

    tiktoken_enabled: bool = False
    """Set this to False for non-OpenAI implementations."""

    embedding_ctx_length: int = 500
    """The maximum number of tokens to embed at once."""

    @property
    def _llm_type(self) -> str:
        return "telnyx-embedding"
