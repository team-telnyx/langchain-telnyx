"""Basic chat example for langchain-telnyx.

Usage:
    export TELNYX_API_KEY="your-api-key"
    python examples/chat_basic.py
"""

from langchain_telnyx import ChatTelnyx


def main() -> None:
    chat = ChatTelnyx(model="moonshotai/Kimi-K2.6")
    response = chat.invoke("Say hello in one short sentence.")
    print(response.content)


if __name__ == "__main__":
    main()
