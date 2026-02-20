"""
Project Synapse — PDF Parser Service
Extracts and chunks text from uploaded PDF files.
"""

import io
from pypdf import PdfReader


def extract_text_from_pdf(pdf_bytes: bytes, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """
    Extract text from a PDF and split into overlapping chunks
    for better entity extraction context.

    Args:
        pdf_bytes: Raw PDF file bytes.
        chunk_size: Target characters per chunk.
        overlap: Number of overlapping characters between chunks.

    Returns:
        List of text chunks.
    """
    reader = PdfReader(io.BytesIO(pdf_bytes))

    # Extract full text from all pages
    full_text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            full_text += page_text + "\n"

    if not full_text.strip():
        return []

    # Split into overlapping chunks
    chunks = []
    start = 0
    while start < len(full_text):
        end = start + chunk_size

        # Try to break at a sentence boundary
        if end < len(full_text):
            # Look for the last period, question mark, or newline within the chunk
            for boundary in [". ", "? ", "! ", "\n"]:
                last_boundary = full_text[start:end].rfind(boundary)
                if last_boundary != -1:
                    end = start + last_boundary + len(boundary)
                    break

        chunk = full_text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Ensure start strictly increases to prevent infinite loops
        next_start = end - overlap
        if next_start <= start:
            next_start = start + 1
        start = next_start

    return chunks
