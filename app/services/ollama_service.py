import os

import requests


OLLAMA_URL = os.getenv(
    "OLLAMA_HOST",
    os.getenv(
        "OLLAMA_URL",
        "http://localhost:11434",
    ),
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "llama3.2:3b",
)


def generate_response(
    prompt: str,
) -> str:
    """
    Send a prompt to the locally running
    Ollama model and return the response.
    """

    if not prompt.strip():
        raise ValueError(
            "Prompt cannot be empty."
        )

    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": OLLAMA_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "stream": False,
        },
        timeout=120,
    )

    response.raise_for_status()

    data = response.json()

    content = (
        data
        .get("message", {})
        .get("content")
    )

    if not content:
        raise ValueError(
            "Ollama returned an empty response."
        )

    return content.strip()
