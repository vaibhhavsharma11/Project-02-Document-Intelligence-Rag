import os

import ollama

from app.services.embedding_service import (
    OLLAMA_HOST,
)


OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "llama3.2:3b",
)


def get_ollama_client() -> ollama.Client:
    """
    Create an Ollama client using the configured Ollama host.
    """

    return ollama.Client(host=OLLAMA_HOST)


def generate_rag_answer(
    query: str,
    retrieved_chunks: list[dict],
) -> str:
    """
    Generate an answer using the user's query and retrieved document chunks.
    """

    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    if not retrieved_chunks:
        return (
            "I could not find relevant information in the uploaded "
            "documents to answer this question."
        )

    context_parts = []

    for index, chunk in enumerate(retrieved_chunks, start=1):
        text = chunk.get("text", "").strip()

        if not text:
            continue

        context_parts.append(
            f"[Source {index}]\n{text}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a document intelligence assistant.

Answer the user's question using ONLY the information contained
in the provided document context.

If the answer cannot be found in the context, clearly say that
the information is not available in the provided document.

Do not invent facts.
Do not use outside knowledge.
Keep the answer concise and useful.

Document context:
{context}

User question:
{query}

Answer:
""".strip()

    client = get_ollama_client()

    response = client.chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    return response["message"]["content"].strip()