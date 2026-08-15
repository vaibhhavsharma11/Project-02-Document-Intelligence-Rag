from pathlib import Path
import shutil
import uuid

from fastapi import (
    APIRouter,
    File,
    HTTPException,
    UploadFile,
)

from app.services.ingestion_service import (
    ingest_pdf,
)

from app.services.retrieval_service import (
    retrieve_relevant_chunks,
)

from app.services.rag_service import (
    generate_rag_answer,
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


@router.post("/search")
async def search_documents(
    payload: dict,
):
    """
    Perform semantic search across indexed
    document chunks.
    """

    query = payload.get(
        "query",
        "",
    )

    top_k = payload.get(
        "top_k",
        3,
    )

    if not isinstance(query, str):
        raise HTTPException(
            status_code=400,
            detail="Query must be a string.",
        )

    if not query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query is required.",
        )

    if not isinstance(top_k, int):
        raise HTTPException(
            status_code=400,
            detail="top_k must be an integer.",
        )

    if top_k <= 0:
        raise HTTPException(
            status_code=400,
            detail="top_k must be greater than zero.",
        )

    try:

        results = (
            retrieve_relevant_chunks(
                query=query,
                top_k=top_k,
            )
        )

        return {
            "query": query,
            "result_count": len(
                results
            ),
            "results": results,
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Search failed: {exc}"
            ),
        ) from exc


@router.post("/ask")
async def ask_document(
    payload: dict,
):
    """
    Retrieve relevant document chunks and
    generate a grounded answer using the
    local language model.
    """

    query = payload.get(
        "query",
        "",
    )

    top_k = payload.get(
        "top_k",
        3,
    )

    if not isinstance(query, str):
        raise HTTPException(
            status_code=400,
            detail="Query must be a string.",
        )

    if not query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query is required.",
        )

    if not isinstance(top_k, int):
        raise HTTPException(
            status_code=400,
            detail="top_k must be an integer.",
        )

    if top_k <= 0:
        raise HTTPException(
            status_code=400,
            detail="top_k must be greater than zero.",
        )

    try:

        retrieved_chunks = retrieve_relevant_chunks(
            query=query,
            top_k=top_k,
        )

        answer = generate_rag_answer(
            query=query,
            retrieved_chunks=retrieved_chunks,
        )

        return {
            "query": query,
            "answer": answer,
            "result_count": len(
                retrieved_chunks
            ),
            "results": retrieved_chunks,
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Question answering failed: "
                f"{exc}"
            ),
        ) from exc