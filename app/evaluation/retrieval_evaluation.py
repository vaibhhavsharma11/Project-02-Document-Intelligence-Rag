from dataclasses import dataclass

from app.services.retrieval_service import (
    retrieve_relevant_chunks,
)


EVALUATION_DISTANCE_THRESHOLD = 450.0


@dataclass
class RetrievalEvaluationResult:
    query: str
    expected_document_id: str
    expected_chunk_index: int | None
    retrieved_count: int
    top_distance: float | None
    top_relevance_score: float | None
    expected_document_retrieved: bool
    expected_chunk_retrieved: bool
    passed: bool


def evaluate_retrieval(
    query: str,
    expected_document_id: str,
    expected_chunk_index: int | None = None,
    top_k: int = 3,
) -> RetrievalEvaluationResult:
    """
    Evaluate whether semantic retrieval returns the
    expected document and, optionally, the expected
    chunk within the evaluation relevance threshold.
    """

    if not query or not query.strip():
        raise ValueError(
            "Query cannot be empty."
        )

    if not expected_document_id:
        raise ValueError(
            "Expected document ID cannot be empty."
        )

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than zero."
        )

    chunks = retrieve_relevant_chunks(
        query=query,
        top_k=top_k,
        document_id=expected_document_id,
        distance_threshold=EVALUATION_DISTANCE_THRESHOLD,
    )

    retrieved_count = len(chunks)

    top_distance = None
    top_relevance_score = None

    if chunks:
        top_distance = chunks[0].get(
            "distance"
        )

        top_relevance_score = chunks[0].get(
            "relevance_score"
        )

    expected_document_retrieved = any(
        chunk.get(
            "metadata",
            {},
        ).get(
            "document_id"
        )
        == expected_document_id
        for chunk in chunks
    )

    expected_chunk_retrieved = False

    if expected_chunk_index is not None:
        expected_chunk_retrieved = any(
            chunk.get(
                "metadata",
                {},
            ).get(
                "document_id"
            )
            == expected_document_id
            and chunk.get(
                "metadata",
                {},
            ).get(
                "chunk_index"
            )
            == expected_chunk_index
            for chunk in chunks
        )
    else:
        expected_chunk_retrieved = (
            expected_document_retrieved
        )

    passed = (
        expected_document_retrieved
        and expected_chunk_retrieved
    )

    return RetrievalEvaluationResult(
        query=query,
        expected_document_id=(
            expected_document_id
        ),
        expected_chunk_index=(
            expected_chunk_index
        ),
        retrieved_count=retrieved_count,
        top_distance=top_distance,
        top_relevance_score=(
            top_relevance_score
        ),
        expected_document_retrieved=(
            expected_document_retrieved
        ),
        expected_chunk_retrieved=(
            expected_chunk_retrieved
        ),
        passed=passed,
    )
