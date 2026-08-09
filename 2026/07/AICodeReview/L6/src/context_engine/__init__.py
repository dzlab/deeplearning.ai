"""
Context engine for code review.
Handles chunking, embedding, and retrieval of relevant code context.
"""

from .chunker import chunk_repository, Chunk
from .embedder import embed_chunks
from .retriever import ContextRetriever

__all__ = ['chunk_repository', 'Chunk', 'embed_chunks', 'ContextRetriever']
