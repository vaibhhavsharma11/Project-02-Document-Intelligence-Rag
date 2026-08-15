from typing import Any


DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 100


def _split_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """
    Split text into overlapping chunks while
    preserving complete words where possible.
    """

    if not text.strip():
        return []

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than zero."
        )

    if chunk_overlap < 0:
        raise ValueError(
            "chunk_overlap cannot be negative."
        )

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller "
            "than chunk_size."
        )

    cleaned_text = " ".join(
        text.split()
    )

    chunks = []

    start = 0
    text_length = len(cleaned_text)

    while start < text_length:

        target_end = min(
            start + chunk_size,
            text_length,
        )

        end = target_end

        if end < text_length:

            whitespace_position = (
                cleaned_text.rfind(
                    " ",
                    start,
                    target_end,
                )
            )

            if (
                whitespace_position > start
            ):
                end = whitespace_position

        chunk = cleaned_text[
            start:end
        ].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        overlap_start = max(
            0,
            end - chunk_overlap,
        )

        next_start = (
            cleaned_text.find(
                " ",
                overlap_start,
                end,
            )
        )

        if next_start == -1:
            start = end
        else:
            start = next_start + 1

    return chunks


def create_chunks(
    document: dict[str, Any],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """
    Convert extracted document pages into
    RAG-ready chunks while preserving metadata.
    """

    document_id = document.get(
        "document_id"
    )

    if not document_id:
        raise ValueError(
            "Document ID is required."
        )

    chunks = []

    chunk_index = 0

    for page in document.get(
        "pages",
        [],
    ):

        page_number = page.get(
            "page_number"
        )

        page_text = page.get(
            "text",
            "",
        )

        page_chunks = _split_text(
            page_text,
            chunk_size,
            chunk_overlap,
        )

        for page_chunk in page_chunks:

            chunk_id = (
                f"{document_id}-"
                f"{chunk_index}"
            )

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "page_number": page_number,
                    "chunk_index": chunk_index,
                    "text": page_chunk,
                    "character_count": len(
                        page_chunk
                    ),
                }
            )

            chunk_index += 1

    return chunks