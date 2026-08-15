from pathlib import Path
from typing import Any

import chromadb


CHROMA_DIRECTORY = Path(
    "data/chroma"
)

COLLECTION_NAME = (
    "document_chunks"
)


class VectorStore:
    """
    Local ChromaDB vector store for
    document chunks.
    """

    def __init__(
        self,
        persist_directory: str = str(
            CHROMA_DIRECTORY
        ),
    ) -> None:

        Path(
            persist_directory
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

        self.client = (
            chromadb.PersistentClient(
                path=persist_directory
            )
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
                "document_id": chunk[
                    "document_id"
                ],
                "page_number": chunk[
                    "page_number"
                ],
                "chunk_index": chunk[
                    "chunk_index"
                ],
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

    def count(self) -> int:
        """
        Return the number of stored chunks.
        """

        return self.collection.count()