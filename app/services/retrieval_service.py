from typing import Any

from app.services.embedding_service import (
    generate_embedding,
)

from app.services.vector_store import (
    VectorStore,
)


def retrieve_relevant_chunks(
    query: str,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """
    Convert a user query into an embedding
    and retrieve the most relevant document
    chunks from ChromaDB.
    """

    if not query.strip():
        raise ValueError(
            "Query cannot be empty."
        )

    query_embedding = generate_embedding(
        query
    )

    vector_store = VectorStore()

    results = vector_store.search(
        query_embedding=query_embedding,
        top_k=top_k,
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

    retrieved_chunks = []

    for index, document in enumerate(
        documents
    ):
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
                "text": document,
                "metadata": metadata,
                "distance": distance,
            }
        )

    return retrieved_chunks