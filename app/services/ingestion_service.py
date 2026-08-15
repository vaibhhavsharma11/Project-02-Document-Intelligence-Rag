from app.services.chunking_service import (
    create_chunks,
)

from app.services.document_service import (
    extract_text_from_pdf,
)

from app.services.embedding_service import (
    embed_chunks,
)

from app.services.vector_store import (
    VectorStore,
)


def ingest_pdf(
    file_path: str,
    document_id: str,
) -> dict:
    """
    Extract, chunk, embed, and index a PDF
    for downstream RAG retrieval.
    """

    document = extract_text_from_pdf(
        file_path
    )

    document["document_id"] = (
        document_id
    )

    full_text = "\n\n".join(
        page["text"]
        for page in document["pages"]
        if page["text"]
    )

    document["text"] = full_text

    chunks = create_chunks(
        document
    )

    embedded_chunks = embed_chunks(
        chunks
    )

    vector_store = VectorStore()

    stored_count = (
        vector_store.add_chunks(
            embedded_chunks
        )
    )

    document["chunks"] = (
        embedded_chunks
    )

    document["vector_store"] = {
        "collection": (
            "document_chunks"
        ),
        "stored_chunks": stored_count,
    }

    return document