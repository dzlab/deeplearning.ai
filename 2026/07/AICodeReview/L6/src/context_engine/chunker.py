"""
AST-based code chunker that extracts functions and classes from Python code.
"""

import ast
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Chunk:
    """Represents a chunk of code."""
    content: str
    type: str  # 'function', 'class', 'method', 'test'
    name: str
    file_path: str
    line_start: int
    line_end: int

    def __repr__(self):
        return f"<Chunk {self.type}:{self.name} ({self.file_path}:{self.line_start}-{self.line_end})>"


class CodeChunker(ast.NodeVisitor):
    """AST visitor that extracts code chunks."""

    def __init__(self, source_code: str, file_path: str):
        self.source_code = source_code
        self.file_path = file_path
        self.source_lines = source_code.splitlines()
        self.chunks: List[Chunk] = []

    def _get_source_segment(self, node: ast.AST) -> str:
        """Extract source code for a given AST node."""
        if not hasattr(node, 'lineno') or not hasattr(node, 'end_lineno'):
            return ""

        start_line = node.lineno - 1  # Convert to 0-indexed
        end_line = node.end_lineno

        lines = self.source_lines[start_line:end_line]
        return '\n'.join(lines)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Visit function definitions."""
        chunk_type = 'test' if node.name.startswith('test_') else 'function'

        chunk = Chunk(
            content=self._get_source_segment(node),
            type=chunk_type,
            name=node.name,
            file_path=self.file_path,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno
        )
        self.chunks.append(chunk)

        # Don't visit nested functions for now - could be added later
        # self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        """Visit async function definitions."""
        chunk = Chunk(
            content=self._get_source_segment(node),
            type='function',
            name=node.name,
            file_path=self.file_path,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno
        )
        self.chunks.append(chunk)

    def visit_ClassDef(self, node: ast.ClassDef):
        """Visit class definitions."""
        chunk = Chunk(
            content=self._get_source_segment(node),
            type='class',
            name=node.name,
            file_path=self.file_path,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno
        )
        self.chunks.append(chunk)

        # Visit methods within the class
        self.generic_visit(node)


def chunk_code(source_code: str, file_path: str = "unknown.py") -> List[Chunk]:
    """
    Chunk a single Python source file into functions and classes.

    Args:
        source_code: Python source code as string
        file_path: Path to the file (for metadata)

    Returns:
        List of Chunk objects
    """
    try:
        tree = ast.parse(source_code)
        chunker = CodeChunker(source_code, file_path)
        chunker.visit(tree)
        return chunker.chunks
    except SyntaxError:
        # If code can't be parsed, return empty list
        return []


def chunk_repository(repo_files: dict) -> List[Chunk]:
    """
    Chunk an entire repository.

    Args:
        repo_files: Dict mapping file paths to source code strings

    Returns:
        List of all chunks from all files
    """
    all_chunks = []

    for file_path, source_code in repo_files.items():
        if file_path.endswith('.py'):
            chunks = chunk_code(source_code, file_path)
            all_chunks.extend(chunks)

    return all_chunks
