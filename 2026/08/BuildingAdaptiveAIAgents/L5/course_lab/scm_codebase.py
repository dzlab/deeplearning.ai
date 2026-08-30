"""Load + validate the committed synthetic SCM codebase fixture."""
from __future__ import annotations

import json

from course_lab import paths

Chunk = dict


def load_scm_chunks() -> list[Chunk]:
    """Load committed SCM function chunks; fail loud if the fixture is missing."""
    fixture = paths.scm_codebase_json()
    if not fixture.exists():
        raise FileNotFoundError(
            f"Missing SCM codebase fixture at {fixture}. "
            "Regenerate it with:  uv run python lab.py gen-scm-codebase"
        )
    data = json.loads(fixture.read_text())
    chunks = data["chunks"]
    ids = {c["id"] for c in chunks}
    for c in chunks:
        for ref in c["imports"] + c["co_edits"]:
            if ref not in ids:
                raise ValueError(f"chunk {c['id']} references unknown id {ref}")
    return chunks


def chunk_text(chunk: Chunk) -> str:
    """The text an embedder indexes for a chunk: signature + doc + body."""
    return f"{chunk['signature']}\n{chunk['docstring']}\n{chunk['body']}"
