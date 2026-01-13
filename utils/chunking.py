"""
Utility functions for chunking large medical documents
for LLM-based processing.

Used in:
- Eponym extraction
- Validation
- Enrichment
"""

from typing import List, Dict
import re


def simple_chunk_text(
    text: str,
    chunk_size: int = 2000,
    overlap: int = 200
) -> List[str]:
    """
    Splits text into overlapping chunks of fixed character length.

    Args:
        text (str): Input document text
        chunk_size (int): Max size of each chunk
        overlap (int): Overlapping characters between chunks

    Returns:
        List[str]: List of text chunks
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap

    return chunks


def sentence_aware_chunking(
    text: str,
    max_chunk_size: int = 2000
) -> List[Dict]:
    """
    Splits text into sentence-aware chunks to preserve medical context.

    Args:
        text (str): Full document text
        max_chunk_size (int): Max characters per chunk

    Returns:
        List[Dict]: Each chunk with text and sentence indices
    """
    # Basic sentence splitting (safe for medical text)
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = ""
    start_sentence = 0

    for idx, sentence in enumerate(sentences):
        if len(current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk += sentence + " "
        else:
            chunks.append({
                "text": current_chunk.strip(),
                "start_sentence": start_sentence,
                "end_sentence": idx - 1
            })
            current_chunk = sentence + " "
            start_sentence = idx

    if current_chunk:
        chunks.append({
            "text": current_chunk.strip(),
            "start_sentence": start_sentence,
            "end_sentence": len(sentences) - 1
        })

    return chunks
