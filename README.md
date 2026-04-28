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

## License

MIT
