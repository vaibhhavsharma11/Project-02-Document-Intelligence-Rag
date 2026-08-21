from pathlib import Path

from fastapi import (
    FastAPI,
    HTTPException,
)
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.api.documents import (
    router as documents_router,
)
from app.services.ollama_service import (
    generate_response,
)


APP_VERSION = "1.0.0"

FRONTEND_DIRECTORY = Path("frontend")


app = FastAPI(
    title="Document Intelligence & RAG Assistant",
    description=(
        "AI-powered document intelligence "
        "and Retrieval-Augmented Generation "
        "assistant."
    ),
    version=APP_VERSION,
)


class ChatRequest(BaseModel):
    prompt: str


app.include_router(
    documents_router
)


if FRONTEND_DIRECTORY.exists():
    app.mount(
        "/static",
        StaticFiles(
            directory=str(FRONTEND_DIRECTORY)
        ),
        name="static",
    )


@app.get(
    "/",
    include_in_schema=False,
)
async def frontend():
    return FileResponse(
        FRONTEND_DIRECTORY / "index.html"
    )


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "application": (
            "Document Intelligence & RAG Assistant"
        ),
        "version": APP_VERSION,
    }


@app.get("/about")
async def about():
    return {
        "application": (
            "Document Intelligence & RAG Assistant"
        ),
        "description": (
            "AI-powered document question answering "
            "using Retrieval-Augmented Generation."
        ),
        "author": "Vaibhav Sharma",
        "version": APP_VERSION,
    }


@app.post("/ai/test")
async def ai_test(
    request: ChatRequest,
):
    try:
        answer = generate_response(
            request.prompt
        )

        return {
            "model": "llama3.2:3b",
            "response": answer,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                "Unable to communicate "
                f"with Ollama: {exc}"
            ),
        ) from exc
