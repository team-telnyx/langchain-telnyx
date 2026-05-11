"""Chat example showing explicit environment configuration.

Usage:
    export TELNYX_API_KEY="your-api-key"
    python examples/chat_with_env.py
"""

import os

from langchain_telnyx import ChatTelnyx


def main() -> None:
    api_key = os.environ.get("TELNYX_API_KEY")
    if not api_key:
        raise RuntimeError("Please set TELNYX_API_KEY before running this example.")

    chat = ChatTelnyx(api_key=api_key, model="moonshotai/Kimi-K2.6")
    response = chat.invoke("Return a short uptime-friendly greeting.")
    print(response.content)


if __name__ == "__main__":
    main()
