import os
from typing import Any

import ollama


OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    os.getenv(
        "OLLAMA_URL",
        "http://host.docker.internal:11434",
    ),
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "nomic-embed-text",
)


def get_ollama_client() -> ollama.Client:
    """
    Create an Ollama client using the configured Ollama host.
    """

    return ollama.Client(
        host=OLLAMA_HOST
    )


def generate_embedding(
    text: str,
) -> list[float]:
    """
    Generate an embedding for a single piece of text.
    """

    if not text or not text.strip():
        raise ValueError(
            "Text cannot be empty."
        )

    client = get_ollama_client()

    response = client.embeddings(
        model=EMBEDDING_MODEL,
        prompt=text,
    )

    return response["embedding"]


def generate_embeddings(
    texts: list[str],
) -> list[list[float]]:
    """
    Generate embeddings for multiple pieces of text.
    """

    if not texts:
        return []

    return [
        generate_embedding(text)
        for text in texts
    ]


def embed_chunks(
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Generate embeddings for RAG chunks and attach
    each embedding to its corresponding chunk.
    """

    if not chunks:
        return []

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = generate_embeddings(
        texts
    )

    embedded_chunks = []

    for chunk, embedding in zip(
        chunks,
        embeddings,
    ):
        embedded_chunk = {
            **chunk,
            "embedding": embedding,
        }

        embedded_chunks.append(
            embedded_chunk
        )

    return embedded_chunks