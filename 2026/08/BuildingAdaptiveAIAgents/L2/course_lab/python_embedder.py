"""course_lab/python_embedder.py — out-of-database MiniLM embeddings.

Embeddings are computed in the **application process** with
``sentence-transformers`` (``all-MiniLM-L6-v2``, 384-d, L2-normalized) and
handed to Oracle as ready-made ``VECTOR(384)`` values. The database only
*stores* and COSINE-*searches* them — it never runs the model.

This is the CPU-sandbox execution path: registering and running the ONNX
MiniLM **inside** the database (``course_lab.onnx_embedder.OnnxDBEmbedder``)
OOMs on tight CPU boxes, because the DB process must load + run the model in
its own memory. Computing the *same* vectors in Python sidesteps that — same
384-d output, same L2-norm, same COSINE distance, so no schema migration.

``PythonMiniLMEmbedder`` implements ``oracleagentmemory``'s duck-typed embedder
interface (``embed`` / ``embed_async``), so it drops straight into
``OracleAgentMemory(embedder=...)`` exactly where ``OnnxDBEmbedder`` did.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np

# Short config aliases -> Hugging Face repo ids. The in-DB ONNX model is the
# uppercased Oracle mining-model name ``ALL_MINILM_L6_V2``; the matching HF
# checkpoint is ``sentence-transformers/all-MiniLM-L6-v2``. We accept the bare
# repo id too, so ``python:sentence-transformers/all-MiniLM-L6-v2`` also works.
_MODEL_ALIASES = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "all-minilm-l6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "all_minilm_l6_v2": "sentence-transformers/all-MiniLM-L6-v2",
}


def resolve_model_id(name: str) -> str:
    """Map a config token (``minilm``, ``all-MiniLM-L6-v2``, or a full repo id)
    to a Hugging Face repo id. Unknown values pass through unchanged so any
    sentence-transformers checkpoint can be named explicitly."""
    return _MODEL_ALIASES.get(str(name).strip().lower(), str(name).strip())


class PythonMiniLMEmbedder:
    """``IEmbedder``-compatible embedder backed by an in-process MiniLM.

    Parameters
    ----------
    model_name:
        A config alias (``minilm`` / ``all-MiniLM-L6-v2``) or a full
        ``sentence-transformers`` repo id. Defaults to MiniLM L6 v2 (384-d).
    device:
        Torch device for inference. Defaults to ``DLAICL_EMBED_DEVICE`` or
        ``"cpu"`` — the sandbox runs on CPU, and pinning it makes behavior
        predictable regardless of whether a GPU happens to be present.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        *,
        device: str | None = None,
    ) -> None:
        self.model_id = resolve_model_id(model_name)
        self.device = device or os.environ.get("DLAICL_EMBED_DEVICE", "cpu")
        # Lazy import so non-embedding code paths (and offline/fake tests) never
        # pay the sentence-transformers import cost.
        from sentence_transformers import SentenceTransformer

        self._model: Any = SentenceTransformer(self.model_id, device=self.device)

    def embed(self, texts: list[str], *, is_query: bool = False) -> np.ndarray:
        # MiniLM has no query/document asymmetry, so is_query is a no-op —
        # matches OnnxDBEmbedder so the two are interchangeable.
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        # normalize_embeddings=True reproduces the in-DB ONNX model's L2-norm,
        # so the stored vectors and COSINE ranking match the ONNX path.
        vecs = self._model.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(vecs, dtype=np.float32)

    async def embed_async(self, texts: list[str], *, is_query: bool = False) -> np.ndarray:
        # The forward pass is synchronous; the workshop's volumes make this fine.
        return self.embed(texts, is_query=is_query)


def make_embedder(spec: str, *, connection: Any = None) -> Any:
    """Build a duck-typed embedder (``embed``/``embed_async``) from a config token.

    One place parses the ``embedder_model`` / ``skill_embedder`` spec so the
    agent-memory routing and the OracleVS builders stay in sync:

    - ``python:<model>`` -> in-process :class:`PythonMiniLMEmbedder` (default
      sandbox path; no DB round-trips, no in-DB model). ``connection`` ignored.
    - ``onnx:<MODEL>``   -> in-DB :class:`~course_lab.onnx_embedder.OnnxDBEmbedder`
      (the retained in-DB reference path; requires ``connection``).

    Any other spec (e.g. ``oci/...``) raises — those are handled by the
    ``oracleagentmemory`` ``Embedder`` directly in ``agent_memory.from_config``,
    not here.
    """
    s = str(spec)
    if s.startswith("python:"):
        return PythonMiniLMEmbedder(s.split(":", 1)[1])
    if s.startswith("onnx:"):
        if connection is None:
            raise ValueError("onnx: embedder requires a live oracledb connection")
        from course_lab.onnx_embedder import OnnxDBEmbedder

        return OnnxDBEmbedder(connection, model_name=s.split(":", 1)[1])
    raise ValueError(
        f"make_embedder does not handle spec {spec!r}; expected a 'python:' or "
        "'onnx:' prefix"
    )


__all__ = ["PythonMiniLMEmbedder", "resolve_model_id", "make_embedder"]
