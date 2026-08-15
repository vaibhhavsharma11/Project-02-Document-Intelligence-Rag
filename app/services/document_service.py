from pathlib import Path

import pymupdf


def extract_text_from_pdf(
    file_path: str,
) -> dict:
    """
    Extract text and basic metadata from a PDF.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"PDF file not found: {file_path}"
        )

    if path.suffix.lower() != ".pdf":
        raise ValueError(
            "Only PDF files are supported."
        )

    document = pymupdf.open(file_path)

    pages = []

    for page_number, page in enumerate(
        document,
        start=1,
    ):
        text = page.get_text("text").strip()

        pages.append(
            {
                "page_number": page_number,
                "text": text,
            }
        )

    metadata = {
        "filename": path.name,
        "page_count": len(document),
    }

    document.close()

    return {
        "metadata": metadata,
        "pages": pages,
    }