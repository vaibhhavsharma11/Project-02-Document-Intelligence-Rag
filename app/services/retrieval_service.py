from typing import Any

from app.services.embedding_service import (
    generate_embedding,
)

from app.services.vector_store import (
    DEFAULT_DISTANCE_THRESHOLD,
    VectorStore,
)


def retrieve_relevant_chunks(
    query: str,
    top_k: int = 3,
    document_id: str | None = None,
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
) -> list[dict[str, Any]]:
    """
    Generate an embedding for the query, perform semantic
    retrieval, apply a relevance threshold, and return
    normalized document chunks.
    """

    if not query or not query.strip():
        raise ValueError(
            "Query cannot be empty."
        )

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than zero."
        )

    if distance_threshold <= 0:
        raise ValueError(
            "distance_threshold must be greater than zero."
        )

    query_embedding = generate_embedding(
        query
    )

    vector_store = VectorStore()

    results = vector_store.search_with_threshold(
        query_embedding=query_embedding,
        top_k=top_k,
        distance_threshold=distance_threshold,
        document_id=document_id,
    )

    documents = results.get(
        "documents",
        [[]],
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]],
    )[0]

    distances = results.get(
        "distances",
        [[]],
    )[0]

    retrieved_chunks: list[dict[str, Any]] = []

    for index, text in enumerate(documents):
        metadata = (
            metadatas[index]
            if index < len(metadatas)
            else {}
        )

        distance = (
            distances[index]
            if index < len(distances)
            else None
        )

        retrieved_chunks.append(
            {
                "text": text,
                "metadata": metadata,
                "distance": distance,
            }
        )

    return retrieved_chunks