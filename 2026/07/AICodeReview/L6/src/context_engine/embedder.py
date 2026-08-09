"""
Embedding module using OpenAI text-embedding-3-large.
"""

import os
from typing import List
from openai import OpenAI
from .chunker import Chunk


def embed_chunks(chunks: List[Chunk], model: str = "text-embedding-3-large") -> List[dict]:
    """
    Create embeddings for code chunks.

    Args:
        chunks: List of Chunk objects to embed
        model: OpenAI embedding model to use

    Returns:
        List of dicts with chunk metadata and embeddings
    """
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    embedded_chunks = []

    # Prepare texts for embedding
    texts = []
    for chunk in chunks:
        # Create a rich text representation for better embedding
        text = f"{chunk.type}: {chunk.name}\n\n{chunk.content}"
        texts.append(text)

    # Get embeddings in batches (OpenAI allows up to 2048 texts per request)
    batch_size = 100
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(
            model=model,
            input=batch
        )
        all_embeddings.extend([item.embedding for item in response.data])

    # Combine chunks with their embeddings
    for chunk, embedding in zip(chunks, all_embeddings):
        embedded_chunks.append({
            'chunk': chunk,
            'embedding': embedding,
            'text': f"{chunk.type}: {chunk.name}\n\n{chunk.content}",
            'metadata': {
                'type': chunk.type,
                'name': chunk.name,
                'file_path': chunk.file_path,
                'line_start': chunk.line_start,
                'line_end': chunk.line_end
            }
        })

    return embedded_chunks
