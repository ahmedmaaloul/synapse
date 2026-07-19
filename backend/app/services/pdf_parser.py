"""
Project Synapse — PDF Parser Service

Extracts text from a PDF and splits it into overlapping, sentence-aware chunks
so each chunk carries enough context for good entity extraction.
"""

from __future__ import annotations

import io

from pypdf import PdfReader

_BOUNDARIES = (". ", "? ", "! ", "\n")


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks, preferring sentence boundaries.

    Pure and deterministic (no I/O) so it can be unit-tested directly.
    """
    if not text or not text.strip():
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    overlap = max(0, min(overlap, chunk_size - 1))

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)

        # Prefer to end on a sentence boundary within the window.
        if end < n:
            best = -1
            for boundary in _BOUNDARIES:
                idx = text.rfind(boundary, start, end)
                if idx != -1:
                    best = max(best, idx + len(boundary))
            if best != -1:
                end = best

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= n:
            break
        # Advance with overlap, always making forward progress.
        next_start = end - overlap
        start = next_start if next_start > start else start + 1

    return chunks


def extract_text_from_pdf(
    pdf_bytes: bytes, chunk_size: int = 1000, overlap: int = 200
) -> list[str]:
    """Extract all page text from a PDF and return sentence-aware chunks."""
    reader = PdfReader(io.BytesIO(pdf_bytes))

    parts: list[str] = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            parts.append(page_text)

    full_text = "\n".join(parts)
    return chunk_text(full_text, chunk_size=chunk_size, overlap=overlap)
