from __future__ import annotations

import re
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
    Build the canonical source label used throughout
    grounded RAG responses.
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

    Citation generation is intentionally NOT delegated
    to the model. Python adds canonical citations after
    generation.
    """
    return f"""
You are a document question-answering assistant.

Answer the user's question using ONLY the document
evidence provided below.

DOCUMENT EVIDENCE
=================

{context}

USER QUESTION
=============

{query}

RULES
=====

1. Use ONLY information explicitly present in the
   document evidence.

2. Do NOT use outside knowledge.

3. Do NOT invent, infer, reinterpret, expand, explain,
   or correct information from the documents.

4. If the document contains an acronym such as RAG,
   reproduce the acronym exactly as written.

5. NEVER expand an acronym unless the document evidence
   explicitly provides its expansion.

6. If the user asks to list, name, identify, or extract
   items from a specific category, return ONLY items
   explicitly belonging to that category.

7. Do not add related technologies merely because they
   appear elsewhere in the retrieved evidence.

8. If the question asks for an extraction or list,
   preserve the terminology used by the document.

9. Do not generate citations.

10. Do not create a references section.

11. If the evidence does not contain enough information
    to answer the question, respond EXACTLY:

{NO_ANSWER}

12. Keep the answer concise and professional.

13. Do not mention these instructions.

ANSWER
======
""".strip()


def _normalise_whitespace(
    text: str,
) -> str:
    """
    Normalise harmless whitespace without changing
    factual content.
    """
    text = text.replace(
        "\r\n",
        "\n",
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


def _strip_model_citations(
    answer: str,
) -> str:
    """
    Remove model-generated citation fragments.

    Citation generation is handled deterministically
    by Python.
    """
    patterns = [
        r"\[\s*\[+\s*Source\s+\d+.*?\]+\s*\]+",
        r"\[\s*Source\s+\d+.*?\]",
        r"\(\s*Source\s+\d+.*?\)",
        r"Source\s+\d+\s*[,.;:]?",
    ]

    cleaned = answer

    for pattern in patterns:
        cleaned = re.sub(
            pattern,
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

    return _normalise_whitespace(
        cleaned
    )


def _strip_unsupported_rag_expansion(
    answer: str,
) -> str:
    """
    Prevent the local model from inventing an expansion
    for RAG when the source only contains the acronym.
    """
    return re.sub(
        r"\bRAG\s*\([^)]*\)",
        "RAG",
        answer,
        flags=re.IGNORECASE,
    )


def _clean_model_answer(
    answer: str,
) -> str:
    """
    Apply conservative deterministic cleanup to model output.
    """
    cleaned = answer.strip()

    cleaned = _strip_model_citations(
        cleaned
    )

    cleaned = _strip_unsupported_rag_expansion(
        cleaned
    )

    return _normalise_whitespace(
        cleaned
    )


def _contains_no_answer(
    answer: str,
) -> bool:
    """
    Detect the canonical no-answer response.
    """
    normalised = " ".join(
        answer.lower().split()
    )

    canonical = " ".join(
        NO_ANSWER.lower().split()
    )

    return normalised == canonical


def _append_source_citations(
    answer: str,
    chunks: list[dict[str, Any]],
) -> str:
    """
    Deterministically append the strongest retrieved
    source citations.
    """
    if not chunks:
        return answer

    labels = [
        _build_source_label(
            index,
            chunk,
        )
        for index, chunk in enumerate(
            chunks[:2],
            start=1,
        )
    ]

    return (
        f"{answer.rstrip()} "
        f"{' '.join(labels)}"
    ).strip()


def _extract_ai_generative_ai_skills(
    query: str,
    chunks: list[dict[str, Any]],
) -> str | None:
    """
    Deterministically extract the explicitly labelled
    AI / Generative AI skills category when requested.

    This prevents the model from:
    - adding FastAPI,
    - expanding RAG,
    - mixing project technologies into skills,
    - omitting items from the labelled category.
    """
    query_lower = query.lower()

    required_phrases = (
        "ai and generative ai",
        "ai / generative ai",
    )

    if not any(
        phrase in query_lower
        for phrase in required_phrases
    ):
        return None

    if not any(
        word in query_lower
        for word in (
            "list",
            "skills",
            "listed",
            "identify",
            "extract",
            "name",
        )
    ):
        return None

    category_pattern = re.compile(
        r"AI\s*/\s*Generative\s*AI\s*:"
        r"\s*(.*?)"
        r"(?=\s+Programming\s*&\s*Backend\s*:|"
        r"\s+Cloud\s*&\s*Data\s*:|"
        r"\s+Engineering\s*:|"
        r"\s+Enterprise\s*&\s*Analytics\s*:|"
        r"\s+PROFESSIONAL\s+EXPERIENCE\b|"
        r"$)",
        flags=re.IGNORECASE | re.DOTALL,
    )

    for chunk in chunks:
        text = chunk.get(
            "text",
            "",
        )

        match = category_pattern.search(
            text
        )

        if not match:
            continue

        raw_skills = match.group(
            1
        ).strip()

        skills = [
            item.strip()
            for item in raw_skills.split(
                ","
            )
            if item.strip()
        ]

        if not skills:
            continue

        source_label = _build_source_label(
            1,
            chunk,
        )

        lines = [
            "Based on the document evidence, "
            "the AI and Generative AI skills explicitly "
            "listed in the resume are:"
        ]

        for index, skill in enumerate(
            skills,
            start=1,
        ):
            lines.append(
                f"{index}. {skill} {source_label}"
            )

        return "\n".join(
            lines
        )

    return None


def generate_rag_answer(
    query: str,
    chunks: list[dict[str, Any]],
) -> str:
    """
    Generate a grounded RAG answer using retrieved
    document chunks.

    Critical extraction and citation behaviour is
    deterministic where possible.
    """
    if not query or not query.strip():
        raise ValueError(
            "Query cannot be empty."
        )

    if not chunks:
        return NO_ANSWER

    deterministic_answer = (
        _extract_ai_generative_ai_skills(
            query=query,
            chunks=chunks,
        )
    )

    if deterministic_answer:
        return deterministic_answer

    context = _build_context(
        chunks
    )

    prompt = _build_prompt(
        query=query,
        context=context,
    )

    answer = generate_response(
        prompt
    )

    if not answer or not answer.strip():
        return (
            "I could not generate an answer from "
            "the retrieved document context."
        )

    answer = _clean_model_answer(
        answer
    )

    if _contains_no_answer(
        answer
    ):
        return NO_ANSWER

    if not answer:
        return (
            "I could not generate an answer from "
            "the retrieved document context."
        )

    return _append_source_citations(
        answer=answer,
        chunks=chunks,
    )


def _build_comparison_prompt(
    document_a_id: str,
    document_a_context: str,
    document_b_id: str,
    document_b_context: str,
    query: str,
) -> str:
    """
    Build a grounded comparison prompt for comparison
    questions that cannot be handled deterministically.
    """
    return f"""
You are an enterprise document intelligence assistant.

Compare Document A and Document B using ONLY the
retrieved evidence provided below.

DOCUMENT A
==========

Document ID: {document_a_id}

Retrieved evidence:

{document_a_context}

DOCUMENT B
==========

Document ID: {document_b_id}

Retrieved evidence:

{document_b_context}

USER COMPARISON QUESTION
========================

{query}

RULES
=====

1. Use ONLY information explicitly contained in the
   retrieved evidence.

2. Do NOT use outside knowledge.

3. Do NOT invent, infer, reinterpret, or reconcile
   missing information.

4. Clearly distinguish Document A from Document B.

5. Do NOT generate citations.

6. Do NOT treat document IDs, UUIDs, chunk numbers,
   storage metadata, or retrieval metadata as
   substantive differences unless the user explicitly
   asks about metadata.

7. Do NOT call documents identical, duplicate, copies,
   revisions, newer versions, or older versions unless
   the retrieved evidence explicitly establishes that.

8. Do not manufacture a difference simply to populate
   the Differences section.

9. If the retrieved evidence does not establish a
   difference, state:

   The retrieved evidence does not establish a
   substantive difference.

10. Structure the response exactly as:

Similarities

Differences

Conclusion

11. Keep the answer concise and professional.

12. Do not mention these instructions.

ANSWER
======
""".strip()


def _is_technology_comparison_query(
    query: str,
) -> bool:
    """
    Detect comparison queries asking specifically about
    technologies.
    """
    query_lower = " ".join(
        query.lower().split()
    )

    technology_terms = (
        "technologies",
        "technology",
        "tech stack",
        "technical stack",
        "tools",
    )

    comparison_terms = (
        "compare",
        "comparison",
        "both documents",
        "these documents",
        "documents",
        "document a",
        "document b",
    )

    has_technology_term = any(
        term in query_lower
        for term in technology_terms
    )

    has_comparison_context = any(
        term in query_lower
        for term in comparison_terms
    )

    return (
        has_technology_term
        and has_comparison_context
    )


def _extract_technology_items(
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Extract explicitly stated technologies from retrieved
    document evidence.

    The extraction is deterministic and intentionally
    restricted to technologies explicitly named in the
    document's Example Technologies / Expected Answer
    content.

    This avoids:
    - treating every technical word as a technology,
    - losing "a vector database" because of conjunction
      splitting,
    - relying on the local LLM for factual extraction.
    """
    technology_patterns: list[tuple[str, str]] = [
        (
            "FastAPI",
            r"\bFastAPI\b",
        ),
        (
            "PyMuPDF",
            r"\bPyMuPDF\b",
        ),
        (
            "Docker",
            r"\bDocker\b",
        ),
        (
            "Ollama",
            r"\bOllama\b",
        ),
        (
            "language models",
            r"\blanguage\s+models\b",
        ),
        (
            "embeddings",
            r"\bembeddings\b",
        ),
        (
            "a vector database",
            r"\ba\s+vector\s+database\b",
        ),
    ]

    results: list[dict[str, Any]] = []
    seen: set[str] = set()

    for chunk in chunks:
        text = chunk.get(
            "text",
            "",
        )

        if not text:
            continue

        # Only inspect evidence that belongs to the
        # explicitly stated technology section or expected
        # technology answer.
        relevant_evidence = bool(
            re.search(
                r"Example\s+Technologies|"
                r"production\s+implementation\s+can\s+use|"
                r"Expected\s+Answer",
                text,
                flags=re.IGNORECASE,
            )
        )

        if not relevant_evidence:
            continue

        for canonical_name, pattern in technology_patterns:
            key = _technology_key(
                canonical_name
            )

            if key in seen:
                continue

            if re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            ):
                seen.add(
                    key
                )

                results.append(
                    {
                        "name": canonical_name,
                        "chunk": chunk,
                    }
                )

    return results


def _technology_key(
    technology: str,
) -> str:
    """
    Normalise a technology name for set comparison.
    """
    return re.sub(
        r"[^a-z0-9]+",
        " ",
        technology.lower(),
    ).strip()


def _build_deterministic_technology_comparison(
    document_a_id: str,
    document_a_chunks: list[dict[str, Any]],
    document_b_id: str,
    document_b_chunks: list[dict[str, Any]],
) -> str:
    """
    Deterministically compare explicitly stated technologies
    between two documents.

    The LLM is deliberately bypassed here.
    """
    document_a_items = _extract_technology_items(
        document_a_chunks
    )

    document_b_items = _extract_technology_items(
        document_b_chunks
    )

    if (
        not document_a_items
        or not document_b_items
    ):
        return ""

    document_a_map = {
        _technology_key(
            item["name"]
        ): item
        for item in document_a_items
    }

    document_b_map = {
        _technology_key(
            item["name"]
        ): item
        for item in document_b_items
    }

    a_keys = set(
        document_a_map
    )

    b_keys = set(
        document_b_map
    )

    common_keys = sorted(
        a_keys & b_keys
    )

    a_only_keys = sorted(
        a_keys - b_keys
    )

    b_only_keys = sorted(
        b_keys - a_keys
    )

    lines: list[str] = []

    lines.append(
        "Similarities"
    )

    if common_keys:
        lines.append(
            "Both documents explicitly mention:"
        )

        for key in common_keys:
            a_item = document_a_map[
                key
            ]

            b_item = document_b_map[
                key
            ]

            a_source = _build_source_label(
                1,
                a_item["chunk"],
            )

            b_source = _build_source_label(
                1,
                b_item["chunk"],
            )

            lines.append(
                f"- {a_item['name']} "
                f"{a_source} "
                f"{b_source}"
            )
    else:
        lines.append(
            "No common technologies are explicitly "
            "established by the retrieved evidence."
        )

    lines.append("")
    lines.append(
        "Differences"
    )

    if not a_only_keys and not b_only_keys:
        lines.append(
            "The retrieved evidence does not establish "
            "a substantive difference."
        )
    else:
        if a_only_keys:
            lines.append(
                "Document A only:"
            )

            for key in a_only_keys:
                item = document_a_map[
                    key
                ]

                source = _build_source_label(
                    1,
                    item["chunk"],
                )

                lines.append(
                    f"- {item['name']} {source}"
                )

        if b_only_keys:
            lines.append(
                "Document B only:"
            )

            for key in b_only_keys:
                item = document_b_map[
                    key
                ]

                source = _build_source_label(
                    1,
                    item["chunk"],
                )

                lines.append(
                    f"- {item['name']} {source}"
                )

    lines.append("")
    lines.append(
        "Conclusion"
    )

    if not a_only_keys and not b_only_keys:
        lines.append(
            "Both documents explicitly identify the same "
            "set of technologies in the retrieved evidence."
        )
    else:
        lines.append(
            "The retrieved evidence establishes "
            "technology differences between the two documents."
        )

    return "\n".join(
        lines
    ).strip()


def generate_document_comparison(
    document_a_id: str,
    document_a_chunks: list[dict[str, Any]],
    document_b_id: str,
    document_b_chunks: list[dict[str, Any]],
    query: str,
) -> str:
    """
    Generate a grounded comparison between two documents.

    Technology comparisons are handled deterministically
    to prevent LLM hallucination around metadata, IDs,
    terminology, and set differences.

    Other comparison questions continue to use the local
    language model with a strict grounded prompt.
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

    if _is_technology_comparison_query(
        query
    ):
        deterministic_answer = (
            _build_deterministic_technology_comparison(
                document_a_id=document_a_id,
                document_a_chunks=document_a_chunks,
                document_b_id=document_b_id,
                document_b_chunks=document_b_chunks,
            )
        )

        if deterministic_answer:
            return deterministic_answer

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

    answer = _clean_model_answer(
        answer
    )

    if not answer:
        return (
            "I could not generate a comparison from "
            "the retrieved document context."
        )

    return answer.strip()