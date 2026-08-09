"""
Context retriever using Chroma vector database.
"""

import os
from typing import List, Optional
import chromadb
from chromadb.config import Settings
from openai import OpenAI


class ContextRetriever:
    """Retrieves relevant code context using vector similarity search."""

    def __init__(self, collection_name: str = "code_chunks"):
        """Initialize retriever with in-memory Chroma database."""
        self.client = chromadb.Client(Settings(
            anonymized_telemetry=False,
            is_persistent=False
        ))
        self.collection_name = collection_name
        self.collection = None
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    def create_index(self, embedded_chunks: List[dict]):
        """
        Create vector index from embedded chunks.

        Args:
            embedded_chunks: List of dicts with 'chunk', 'embedding', 'text', 'metadata'
        """
        # Delete existing collection if it exists
        try:
            self.client.delete_collection(self.collection_name)
        except:
            pass

        # Create new collection
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"description": "Code chunks for review context"}
        )

        # Add documents to collection
        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for i, item in enumerate(embedded_chunks):
            ids.append(f"chunk_{i}")
            embeddings.append(item['embedding'])
            documents.append(item['text'])
            metadatas.append(item['metadata'])

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

    def retrieve(
        self,
        query: str,
        n_results: int = 5,
        filter_dict: Optional[dict] = None
    ) -> List[dict]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: Search query (e.g., PR title + diff)
            n_results: Number of results to return
            filter_dict: Optional metadata filter

        Returns:
            List of dicts with 'content', 'metadata', 'distance'
        """
        if self.collection is None:
            raise ValueError("Index not created. Call create_index() first.")

        # Embed the query
        response = self.openai_client.embeddings.create(
            model="text-embedding-3-large",
            input=query
        )
        query_embedding = response.data[0].embedding

        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter_dict
        )

        # Format results
        retrieved = []
        for i in range(len(results['ids'][0])):
            retrieved.append({
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else None
            })

        return retrieved

    def retrieve_for_pr(
        self,
        pr: dict,
        n_results: int = 5
    ) -> str:
        """
        Retrieve context for a PR and format as a string.

        Args:
            pr: PR dict with 'title' and 'diff'
            n_results: Number of chunks to retrieve

        Returns:
            Formatted context string
        """
        # Build query from PR
        query = f"{pr['title']}\n\n{pr['diff']}"

        # Retrieve relevant chunks
        results = self.retrieve(query, n_results)

        # Format as context string
        context_parts = []
        for result in results:
            chunk_type = result['metadata']['type']
            chunk_name = result['metadata']['name']
            content = result['content']

            # Extract just the code part (after the "type: name" header)
            if '\n\n' in content:
                code = content.split('\n\n', 1)[1]
            else:
                code = content

            context_parts.append(f"# {chunk_type}: {chunk_name}\n{code}")

        return '\n\n'.join(context_parts)
