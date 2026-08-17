from typing import Any

from pydantic import BaseModel, Field


class DocumentQueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        description="Natural-language question or semantic search query.",
    )
    top_k: int = Field(
        default=3,
        gt=0,
        description="Number of relevant document chunks to retrieve.",
    )


class DocumentChunkResponse(BaseModel):
    text: str
    metadata: dict[str, Any]
    distance: float | None = None


class SearchResponse(BaseModel):
    query: str
    result_count: int
    results: list[DocumentChunkResponse]


class AskResponse(BaseModel):
    query: str
    answer: str
    result_count: int
    results: list[DocumentChunkResponse]
