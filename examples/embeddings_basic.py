"""Basic embeddings example for langchain-telnyx.

Usage:
    export TELNYX_API_KEY="your-api-key"
    python examples/embeddings_basic.py
"""

from langchain_telnyx import TelnyxEmbeddings


def main() -> None:
    embeddings = TelnyxEmbeddings(model="thenlper/gte-large")
    vector = embeddings.embed_query("Hello world")
    print(f"Vector length: {len(vector)}")
    print(f"First 5 values: {vector[:5]}")


if __name__ == "__main__":
    main()
