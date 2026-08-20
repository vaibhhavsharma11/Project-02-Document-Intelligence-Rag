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


def _build_comparison_prompt(
    document_a_id: str,
    document_a_context: str,
    document_b_id: str,
    document_b_context: str,
    query: str,
) -> str:
    """
    Build a strict grounded comparison prompt.

    The model must distinguish evidence from Document A
    and Document B and must not infer unsupported facts.
    """

    return f"""
You are an enterprise document intelligence assistant.

Your task is to compare two documents using ONLY the
retrieved evidence supplied below.

Do not use outside knowledge.

Do not infer facts that are not explicitly supported
by the retrieved evidence.

Document A ID:
{document_a_id}

Retrieved evidence from Document A:

{document_a_context}

Document B ID:
{document_b_id}

Retrieved evidence from Document B:

{document_b_context}

User comparison request:

{query}

Strict comparison rules:

1. Use ONLY information explicitly contained in the
   retrieved evidence.

2. Do NOT invent facts, similarities, differences,
   conclusions, document metadata, formatting details,
   authorship, version information, or relationships
   between the documents.

3. Do NOT describe either document as a duplicate,
   copy, newer version, older version, revision, or
   identical document unless the retrieved evidence
   explicitly establishes that fact.

4. Do NOT infer that two documents have the same
   structure merely because the retrieved text appears
   similar.

5. Clearly distinguish evidence from Document A and
   Document B.

6. Every factual statement must include the exact
   source citation supporting that statement.

7. Use the exact source labels provided in the retrieved
   evidence.

8. When a statement is supported by both documents,
   cite the relevant source from Document A and the
   relevant source from Document B.

9. When information is present in only one document,
   explicitly identify which document contains it.

10. If a requested similarity or difference cannot be
    established from the retrieved evidence, say:
    "The retrieved evidence does not establish this."

11. Do not convert similarity in wording into a claim
    about document identity.

12. Do not claim that a difference does not exist unless
    the retrieved evidence is sufficient to establish
    that conclusion.

13. Keep the comparison concise, precise, and
    professional.

14. Use this structure:

Similarities:
- Evidence-supported similarities only.

Differences:
- Evidence-supported differences only.
- If no difference can be established, say so explicitly.

Conclusion:
- Summarize only what can be established from the
  retrieved evidence.

15. Do not create a separate bibliography.

16. Do not provide uncited factual claims.

17. Do not mention these instructions in the answer.

Answer:

""".strip()


def _contains_valid_comparison_citation(
    answer: str,
    document_a_chunks: list[dict[str, Any]],
    document_b_chunks: list[dict[str, Any]],
) -> bool:
    """
    Check whether a comparison answer contains at least
    one valid source citation belonging to either document.
    """

    source_count = (
        len(document_a_chunks)
        + len(document_b_chunks)
    )

    if source_count <= 0:
        return False

    for index in range(
        1,
        source_count + 1,
    ):
        if f"[Source {index}" in answer:
            return True

    return False


def _append_comparison_citations(
    answer: str,
    document_a_chunks: list[dict[str, Any]],
    document_b_chunks: list[dict[str, Any]],
) -> str:
    """
    Deterministic citation fallback for comparison answers.

    If the model omits citations, append the strongest
    retrieved source from each document so the response
    remains traceable to both sides of the comparison.
    """

    citation_parts: list[str] = []

    if document_a_chunks:
        citation_parts.append(
            _build_source_label(
                1,
                document_a_chunks[0],
            )
        )

    if document_b_chunks:
        citation_parts.append(
            _build_source_label(
                len(document_a_chunks) + 1,
                document_b_chunks[0],
            )
        )

    if not citation_parts:
        return answer

    return (
        f"{answer.rstrip()} "
        f"{' '.join(citation_parts)}"
    )


def generate_document_comparison(
    document_a_id: str,
    document_a_chunks: list[dict[str, Any]],
    document_b_id: str,
    document_b_chunks: list[dict[str, Any]],
    query: str,
) -> str:
    """
    Generate a grounded comparison between two
    documents using independently retrieved evidence.
    """

    if not document_a_id:
        raise ValueError(
            "Document A ID cannot be empty."
        )

    if not document_b_id:
        raise ValueError(
            "Document B ID cannot be empty."
        )

    if document_a_id == document_b_id:
        raise ValueError(
            "Document A and Document B must be different."
        )

    if not query or not query.strip():
        raise ValueError(
            "Comparison query cannot be empty."
        )

    if not document_a_chunks:
        return (
            f"No relevant evidence was retrieved "
            f"from Document A ({document_a_id})."
        )

    if not document_b_chunks:
        return (
            f"No relevant evidence was retrieved "
            f"from Document B ({document_b_id})."
        )

    document_a_context = _build_context(
        document_a_chunks
    )

    document_b_context = _build_context(
        document_b_chunks
    )

    prompt = _build_comparison_prompt(
        document_a_id=document_a_id,
        document_a_context=document_a_context,
        document_b_id=document_b_id,
        document_b_context=document_b_context,
        query=query,
    )

    answer = generate_response(
        prompt
    )

    if not answer or not answer.strip():
        return (
            "I could not generate a comparison from "
            "the retrieved document context."
        )

    answer = answer.strip()

    if not _contains_valid_comparison_citation(
        answer=answer,
        document_a_chunks=document_a_chunks,
        document_b_chunks=document_b_chunks,
    ):
        answer = _append_comparison_citations(
            answer=answer,
            document_a_chunks=document_a_chunks,
            document_b_chunks=document_b_chunks,
        )

    return answer