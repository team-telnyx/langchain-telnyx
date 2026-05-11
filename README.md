# langchain-telnyx

An integration package connecting [Telnyx](https://telnyx.com) AI Inference and LangChain.

Telnyx provides OpenAI-compatible LLMs hosted on its own GPU infrastructure. See the [Available Models](https://developers.telnyx.com/docs/inference/models) page for the full list.

## Installation

```bash
pip install langchain-telnyx
```

## Chat Models

```python
from langchain_telnyx import ChatTelnyx

chat = ChatTelnyx(model="moonshotai/Kimi-K2.6")
response = chat.invoke("Hello!")
```

## Embeddings

```python
from langchain_telnyx import TelnyxEmbeddings

embeddings = TelnyxEmbeddings(model="thenlper/gte-large")
vectors = embeddings.embed_query("Hello world")
```

## Configuration

Set the `TELNYX_API_KEY` environment variable with your Telnyx API key, or pass it via the `api_key` parameter.

Get an API key at [telnyx.com](https://telnyx.com).

By default, `langchain-telnyx` uses the following OpenAI-compatible base URL:

```text
https://api.telnyx.com/v2/ai/openai
```

You can override it either with the `base_url` parameter or with the `TELNYX_API_BASE` environment variable.

Example:

```python
from langchain_telnyx import ChatTelnyx

chat = ChatTelnyx(
    api_key="YOUR_API_KEY",
    base_url="https://api.telnyx.com/v2/ai/openai",
    model="moonshotai/Kimi-K2.6",
)
```

## Supported Models

See [Available Models — Telnyx Docs](https://developers.telnyx.com/docs/inference/models) for the full, up-to-date list.

### Chat Models

| Model ID | Parameters | Context | Best For |
| --- | ---:| ---:| --- |
| `moonshotai/Kimi-K2.6` | 1.0T | 256K | Highest intelligence, voice AI **(Recommended)** |
| `zai-org/GLM-5.1-FP8` | 753.9B | 202K | Reasoning, function calling |
| `MiniMaxAI/MiniMax-M2.7` | — | 2M | Cheapest, high intelligence |

### Embedding Models

| Model ID | Dimensions | Best For |
| --- | ---:| --- |
| `thenlper/gte-large` | 1024 | Text embeddings |

## Examples

You can find runnable smoke examples in [`examples/`](./examples):

- `examples/chat_basic.py`
- `examples/embeddings_basic.py`
- `examples/chat_with_env.py`

To run them from a local clone:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
export TELNYX_API_KEY="your-api-key"
python examples/chat_basic.py
python examples/chat_with_env.py
python examples/embeddings_basic.py
```

## Development

Run tests from a local clone:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install pytest pytest-asyncio langchain-core langchain-openai openai
python -m pytest -q
```

If you already have the virtual environment ready:

```bash
source .venv/bin/activate
python -m pytest -q
```

## License

MIT
