from pathlib import Path
import shutil
import uuid

from fastapi import (
    APIRouter,
    File,
    HTTPException,
    UploadFile,
)

from app.schemas.document_schemas import (
    AskResponse,
    DocumentComparisonRequest,
    DocumentComparisonResponse,
    DocumentListResponse,
    DocumentQueryRequest,
    DocumentSummaryResponse,
    SearchResponse,
)

from app.services.document_catalogue_service import (
    get_document,
    list_documents,
)

from app.services.ingestion_service import (
    ingest_pdf,
)

from app.services.rag_service import (
    generate_document_comparison,
    generate_rag_answer,
)

from app.services.retrieval_service import (
    retrieve_document_chunks_for_comparison,
    retrieve_relevant_chunks,
)


router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)


DOCUMENT_DIRECTORY = Path(
    "data/documents"
)

DOCUMENT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
):
    """
    Upload, extract, chunk, embed, and index
    a PDF document.
    """

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required.",
        )

    if not file.filename.lower().endswith(
        ".pdf"
    ):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    document_id = str(
        uuid.uuid4()
    )

    stored_filename = (
        f"{document_id}.pdf"
    )

    file_path = (
        DOCUMENT_DIRECTORY
        / stored_filename
    )

    with file_path.open("wb") as buffer:
        shutil.copyfileobj(
            file.file,
            buffer,
        )

    try:
        document = ingest_pdf(
            str(file_path),
            document_id,
        )

        return {
            "document_id": document_id,
            "filename": file.filename,
            "metadata": document[
                "metadata"
            ],
            "text_length": len(
                document["text"]
            ),
            "chunk_count": len(
                document["chunks"]
            ),
            "vector_store": document[
                "vector_store"
            ],
            "chunks": [
                {
                    "chunk_id": chunk[
                        "chunk_id"
                    ],
                    "document_id": chunk[
                        "document_id"
                    ],
                    "page_number": chunk[
                        "page_number"
                    ],
                    "chunk_index": chunk[
                        "chunk_index"
                    ],
                    "character_count": chunk[
                        "character_count"
                    ],
                    "text": chunk[
                        "text"
                    ],
                }
                for chunk in document[
                    "chunks"
                ]
            ],
        }

    except Exception as exc:

        if file_path.exists():
            file_path.unlink()

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to process PDF: "
                f"{exc}"
            ),
        ) from exc


@router.get(
    "",
    response_model=DocumentListResponse,
)
async def list_indexed_documents():
    """
    List all documents currently represented
    in the vector store.
    """

    try:
        documents = list_documents()

        return DocumentListResponse(
            document_count=len(documents),
            documents=documents,
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Unable to list documents: {exc}"
            ),
        ) from exc


@router.get(
    "/{document_id}",
    response_model=DocumentSummaryResponse,
)
async def get_indexed_document(
    document_id: str,
):
    """
    Return metadata and indexing statistics
    for a specific document.
    """

    try:
        document = get_document(
            document_id
        )

        if document is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Document not found: "
                    f"{document_id}"
                ),
            )

        return DocumentSummaryResponse(
            **document
        )

    except HTTPException:
        raise

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Unable to retrieve document: "
                f"{exc}"
            ),
        ) from exc


@router.post(
    "/search",
    response_model=SearchResponse,
)
async def search_documents(
    payload: DocumentQueryRequest,
):
    """
    Perform semantic search across indexed
    document chunks.

    Retrieval can optionally be restricted
    to a specific document and relevance
    threshold.
    """

    try:
        results = retrieve_relevant_chunks(
            query=payload.query,
            top_k=payload.top_k,
            document_id=payload.document_id,
            distance_threshold=(
                payload.distance_threshold
            ),
        )

        return SearchResponse(
            query=payload.query,
            result_count=len(results),
            results=results,
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Search failed: {exc}"
            ),
        ) from exc


@router.post(
    "/ask",
    response_model=AskResponse,
)
async def ask_document(
    payload: DocumentQueryRequest,
):
    """
    Retrieve relevant document chunks and
    generate a grounded answer using the
    local language model.

    Retrieval can optionally be restricted
    to a specific document and relevance
    threshold.
    """

    try:
        retrieved_chunks = (
            retrieve_relevant_chunks(
                query=payload.query,
                top_k=payload.top_k,
                document_id=payload.document_id,
                distance_threshold=(
                    payload.distance_threshold
                ),
            )
        )

        answer = generate_rag_answer(
            query=payload.query,
            chunks=retrieved_chunks,
        )

        return AskResponse(
            query=payload.query,
            answer=answer,
            result_count=len(
                retrieved_chunks
            ),
            results=retrieved_chunks,
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Question answering failed: "
                f"{exc}"
            ),
        ) from exc


@router.post(
    "/compare",
    response_model=DocumentComparisonResponse,
)
async def compare_documents(
    payload: DocumentComparisonRequest,
):
    """
    Compare evidence retrieved independently
    from two indexed documents.

    Each document is searched separately using
    the same comparison query. The strongest
    matches from each selected document are
    retrieved independently without applying
    the normal global relevance threshold.

    The retrieved evidence is then supplied to
    the local language model for grounded
    comparison.
    """

    if (
        payload.document_a_id
        == payload.document_b_id
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Document A and Document B "
                "must be different."
            ),
        )

    try:
        document_a = get_document(
            payload.document_a_id
        )

        if document_a is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    "Document A not found: "
                    f"{payload.document_a_id}"
                ),
            )

        document_b = get_document(
            payload.document_b_id
        )

        if document_b is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    "Document B not found: "
                    f"{payload.document_b_id}"
                ),
            )

        document_a_chunks = (
            retrieve_document_chunks_for_comparison(
                query=payload.query,
                document_id=payload.document_a_id,
                top_k=payload.top_k,
            )
        )

        document_b_chunks = (
            retrieve_document_chunks_for_comparison(
                query=payload.query,
                document_id=payload.document_b_id,
                top_k=payload.top_k,
            )
        )

        answer = generate_document_comparison(
            document_a_id=payload.document_a_id,
            document_a_chunks=document_a_chunks,
            document_b_id=payload.document_b_id,
            document_b_chunks=document_b_chunks,
            query=payload.query,
        )

        return DocumentComparisonResponse(
            document_a_id=payload.document_a_id,
            document_b_id=payload.document_b_id,
            query=payload.query,
            answer=answer,
            document_a_result_count=len(
                document_a_chunks
            ),
            document_b_result_count=len(
                document_b_chunks
            ),
            document_a_results=document_a_chunks,
            document_b_results=document_b_chunks,
        )

    except HTTPException:
        raise

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Document comparison failed: "
                f"{exc}"
            ),
        ) from exc