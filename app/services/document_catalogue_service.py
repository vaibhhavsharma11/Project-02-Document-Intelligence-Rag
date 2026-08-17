from typing import Any

from app.services.vector_store import (
    VectorStore,
)


def build_document_summary(
    document_id: str,
    chunks: dict[str, Any],
) -> dict[str, Any] | None:
    """
    Build a document-level summary from indexed
    chunk metadata.
    """

    ids = chunks.get(
        "ids",
        [],
    )

    documents = chunks.get(
        "documents",
        [],
    )

    metadatas = chunks.get(
        "metadatas",
        [],
    )

    if not ids:
        return None

    if not metadatas:
        return None

    page_numbers = [
        metadata["page_number"]
        for metadata in metadatas
        if "page_number" in metadata
    ]

    chunk_indexes = [
        metadata["chunk_index"]
        for metadata in metadatas
        if "chunk_index" in metadata
    ]

    return {
        "document_id": document_id,
        "chunk_count": len(ids),
        "page_count": (
            len(set(page_numbers))
            if page_numbers
            else 0
        ),
        "character_count": sum(
            len(text)
            for text in documents
            if text
        ),
        "min_page_number": (
            min(page_numbers)
            if page_numbers
            else None
        ),
        "max_page_number": (
            max(page_numbers)
            if page_numbers
            else None
        ),
        "chunk_indexes": sorted(
            chunk_indexes
        ),
    }


def list_documents() -> list[dict[str, Any]]:
    """
    Return a catalogue of all indexed documents.
    """

    vector_store = VectorStore()

    chunks = vector_store.get_all_chunks()

    metadatas = chunks.get(
        "metadatas",
        [],
    )

    document_ids = sorted(
        {
            metadata["document_id"]
            for metadata in metadatas
            if metadata.get("document_id")
        }
    )

    documents = []

    for document_id in document_ids:
        document_chunks = (
            vector_store.get_document_chunks(
                document_id
            )
        )

        summary = build_document_summary(
            document_id,
            document_chunks,
        )

        if summary:
            documents.append(summary)

    return documents


def get_document(
    document_id: str,
) -> dict[str, Any] | None:
    """
    Return a single indexed document summary.
    """

    vector_store = VectorStore()

    chunks = vector_store.get_document_chunks(
        document_id
    )

    return build_document_summary(
        document_id,
        chunks,
    )