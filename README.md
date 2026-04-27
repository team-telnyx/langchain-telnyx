# langchain-telnyx

An integration package connecting Telnyx and LangChain.

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

Set the `TELNYX_API_KEY` environment variable with your Telnyx API key, or pass it via the `telnyx_api_key` parameter.

## License

MIT
