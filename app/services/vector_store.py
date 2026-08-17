from pathlib import Path
from typing import Any

import chromadb


CHROMA_DIRECTORY = Path("data/chroma")

COLLECTION_NAME = "document_chunks"

DEFAULT_DISTANCE_THRESHOLD = 500.0


class VectorStore:
    """
    Local ChromaDB vector store for document chunks.
    """

    def __init__(
        self,
        persist_directory: str = str(CHROMA_DIRECTORY),
    ) -> None:
        Path(persist_directory).mkdir(
            parents=True,
            exist_ok=True,
        )

        self.client = chromadb.PersistentClient(
            path=persist_directory
        )

        self.collection = (
            self.client.get_or_create_collection(
                name=COLLECTION_NAME
            )
        )

    def add_chunks(
        self,
        chunks: list[dict[str, Any]],
    ) -> int:
        """
        Store embedded chunks in ChromaDB.
        """

        if not chunks:
            return 0

        ids = [
            chunk["chunk_id"]
            for chunk in chunks
        ]

        embeddings = [
            chunk["embedding"]
            for chunk in chunks
        ]

        documents = [
            chunk["text"]
            for chunk in chunks
        ]

        metadatas = [
            {
                "document_id": chunk["document_id"],
                "page_number": chunk["page_number"],
                "chunk_index": chunk["chunk_index"],
            }
            for chunk in chunks
        ]

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        return len(chunks)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 3,
    ) -> dict[str, Any]:
        """
        Perform semantic similarity search.
        """

        if not query_embedding:
            raise ValueError(
                "Query embedding cannot be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        return self.collection.query(
            query_embeddings=[
                query_embedding
            ],
            n_results=top_k,
        )

    def search_with_threshold(
        self,
        query_embedding: list[float],
        top_k: int = 3,
        distance_threshold: float = (
            DEFAULT_DISTANCE_THRESHOLD
        ),
        document_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Perform semantic search and return only results
        whose distance is within the configured relevance
        threshold.

        Optionally restrict retrieval to a specific document.
        """

        if not query_embedding:
            raise ValueError(
                "Query embedding cannot be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        if distance_threshold <= 0:
            raise ValueError(
                "distance_threshold must be greater than zero."
            )

        where: dict[str, Any] | None = None

        if document_id:
            where = {
                "document_id": document_id,
            }

        results = self.collection.query(
            query_embeddings=[
                query_embedding
            ],
            n_results=top_k,
            where=where,
        )

        distances = results.get(
            "distances",
            [[]],
        )[0]

        if not distances:
            return {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            }

        relevant_indices = [
            index
            for index, distance in enumerate(
                distances
            )
            if distance <= distance_threshold
        ]

        return {
            "ids": [
                [
                    results["ids"][0][index]
                    for index in relevant_indices
                ]
            ],
            "documents": [
                [
                    results["documents"][0][index]
                    for index in relevant_indices
                ]
            ],
            "metadatas": [
                [
                    results["metadatas"][0][index]
                    for index in relevant_indices
                ]
            ],
            "distances": [
                [
                    results["distances"][0][index]
                    for index in relevant_indices
                ]
            ],
        }

    def get_all_chunks(
        self,
    ) -> dict[str, Any]:
        """
        Return all indexed chunks and their metadata.

        Used for document-level catalogue and
        inspection operations.
        """

        return self.collection.get(
            include=[
                "documents",
                "metadatas",
            ]
        )

    def get_document_chunks(
        self,
        document_id: str,
    ) -> dict[str, Any]:
        """
        Return all indexed chunks belonging to
        a specific document.
        """

        return self.collection.get(
            where={
                "document_id": document_id,
            },
            include=[
                "documents",
                "metadatas",
            ],
        )

    def count(self) -> int:
        """
        Return the number of stored chunks.
        """

        return self.collection.count()