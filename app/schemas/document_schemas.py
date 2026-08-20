from typing import Any

from pydantic import BaseModel, Field


class DocumentQueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        description=(
            "Natural-language question or semantic search query."
        ),
    )

    top_k: int = Field(
        default=3,
        gt=0,
        description=(
            "Maximum number of relevant document chunks to retrieve."
        ),
    )

    document_id: str | None = Field(
        default=None,
        description=(
            "Optional document ID used to restrict retrieval "
            "to a specific document."
        ),
    )

    distance_threshold: float = Field(
        default=450.0,
        gt=0,
        description=(
            "Maximum vector distance allowed for a result "
            "to be considered relevant."
        ),
    )


class DocumentChunkResponse(BaseModel):
    text: str
    metadata: dict[str, Any]
    distance: float | None = None

    relevance_score: float | None = Field(
        default=None,
        description=(
            "Normalized retrieval relevance score between "
            "0 and 1. This is a relative retrieval-quality "
            "indicator, not a probability."
        ),
    )


class SearchResponse(BaseModel):
    query: str
    result_count: int
    results: list[DocumentChunkResponse]


class AskResponse(BaseModel):
    query: str
    answer: str
    result_count: int
    results: list[DocumentChunkResponse]


class DocumentSummaryResponse(BaseModel):
    document_id: str
    chunk_count: int
    page_count: int
    character_count: int
    min_page_number: int | None = None
    max_page_number: int | None = None
    chunk_indexes: list[int]


class DocumentListResponse(BaseModel):
    document_count: int
    documents: list[DocumentSummaryResponse]


class DocumentComparisonRequest(BaseModel):
    document_a_id: str = Field(
        ...,
        min_length=1,
        description=(
            "Document ID for the first document."
        ),
    )

    document_b_id: str = Field(
        ...,
        min_length=1,
        description=(
            "Document ID for the second document."
        ),
    )

    query: str = Field(
        ...,
        min_length=1,
        description=(
            "Natural-language comparison request."
        ),
    )

    top_k: int = Field(
        default=3,
        gt=0,
        description=(
            "Maximum number of relevant chunks to retrieve "
            "from each document."
        ),
    )

    distance_threshold: float = Field(
        default=450.0,
        gt=0,
        description=(
            "Maximum vector distance allowed for retrieved "
            "chunks from either document."
        ),
    )


class DocumentComparisonResponse(BaseModel):
    document_a_id: str
    document_b_id: str
    query: str
    answer: str

    document_a_result_count: int
    document_b_result_count: int

    document_a_results: list[DocumentChunkResponse]
    document_b_results: list[DocumentChunkResponse]
