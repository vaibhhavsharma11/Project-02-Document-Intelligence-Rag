from typing import Any

from app.services.ollama_service import generate_response


NO_ANSWER = (
    "I could not find relevant information in the "
    "uploaded documents to answer this question."
)


def _build_source_label(
    index: int,
    chunk: dict[str, Any],
) -> str:
    """
    Build a human-readable source label for a
    retrieved document chunk.
    """

    metadata = chunk.get(
        "metadata",
        {},
    )

    document_id = metadata.get(
        "document_id",
        "unknown",
    )

    page_number = metadata.get(
        "page_number",
        "unknown",
    )

    chunk_index = metadata.get(
        "chunk_index",
        "unknown",
    )

    return (
        f"[Source {index} — "
        f"Document ID: {document_id}, "
        f"Page: {page_number}, "
        f"Chunk: {chunk_index}]"
    )


def _build_context(
    chunks: list[dict[str, Any]],
) -> str:
    """
    Build explicitly numbered source context
    for grounded answer generation.
    """

    context_parts: list[str] = []

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):
        source_label = _build_source_label(
            index,
            chunk,
        )

        text = chunk.get(
            "text",
            "",
        )

        context_parts.append(
            f"{source_label}\n"
            f"{text}"
        )

    return "\n\n".join(
        context_parts
    )


def _build_prompt(
    query: str,
    context: str,
) -> str:
    """
    Build a strict grounded-generation prompt.
    """

    return f"""
You are an enterprise document intelligence assistant.

Answer the user's question using ONLY the information
contained in the retrieved document sources below.

Do not use outside knowledge.

If the retrieved sources do not contain enough
information to answer the question, respond exactly:

{NO_ANSWER}

Citation rules:

1. Every factual claim must be supported by one or
   more retrieved sources.

2. Cite the source immediately after the factual
   statement it supports.

3. Use the exact source labels provided below.

4. Do not invent source numbers.

5. Do not cite a source unless the information
   actually comes from that source.

6. If multiple sources support a statement,
   cite all relevant sources.

7. Do not create a separate bibliography.

8. Keep the answer concise and professional.

9. Do not provide uncited factual claims.

10. When answering with a list, cite each item or
    cite the sentence introducing the list if the
    same source supports every item.

Retrieved sources:

{context}

User question:

{query}

Answer:
""".strip()


def _contains_valid_source_citation(
    answer: str,
    source_count: int,
) -> bool:
    """
    Check whether the generated answer contains
    at least one valid source citation.
    """

    for index in range(
        1,
        source_count + 1,
    ):
        if f"[Source {index}" in answer:
            return True

    return False


def _append_source_citation(
    answer: str,
    chunks: list[dict[str, Any]],
) -> str:
    """
    Provide a deterministic citation fallback when
    the language model answers correctly but omits
    source citations.

    The first retrieved source is appended because
    retrieved chunks are already ranked by relevance.
    """

    if not chunks:
        return answer

    source_label = _build_source_label(
        1,
        chunks[0],
    )

    return (
        f"{answer.rstrip()} "
        f"{source_label}"
    )


def generate_rag_answer(
    query: str,
    chunks: list[dict[str, Any]],
) -> str:
    """
    Generate a grounded RAG answer using retrieved
    document chunks with explicit source citations.
    """

    if not query or not query.strip():
        raise ValueError(
            "Query cannot be empty."
        )

    if not chunks:
        return NO_ANSWER

    context = _build_context(
        chunks
    )

    prompt = _build_prompt(
        query,
        context,
    )

    answer = generate_response(
        prompt
    )

    if not answer or not answer.strip():
        return (
            "I could not generate an answer from "
            "the retrieved document context."
        )

    answer = answer.strip()

    if answer == NO_ANSWER:
        return NO_ANSWER

    if not _contains_valid_source_citation(
        answer,
        len(chunks),
    ):
        answer = _append_source_citation(
            answer,
            chunks,
        )

    return answer
