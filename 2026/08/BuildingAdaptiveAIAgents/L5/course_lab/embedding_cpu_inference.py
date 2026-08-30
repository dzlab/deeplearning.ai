"""CPU-side base-vs-fine-tuned retrieval for the embedding arm's notebook_cpu.

Sibling of ``gemma_cpu_inference`` (weight arm). EmbeddingGemma-300m loads and
encodes on CPU via plain sentence-transformers. When the trained checkpoint is
present we encode live on CPU; otherwise we replay a committed cache. Raises
(never fabricates) when a query is neither live-able nor cached.

Live inference is stateful: the base + fine-tuned encoders are loaded ONCE and
the corpus matrix is encoded ONCE per (arm, dim) and memoized at module level,
so scoring many queries in one notebook session stays cheap (one query-encode +
a dot product per call) — the same discipline as the weight arm's ``_BASE_CACHE``.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np


def _cache_key(query: str, dim: int) -> str:
    return f"{dim}::{query}"


def _load_cache(path: Path) -> dict:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}


# Module-level caches so repeated calls in one notebook session do not reload the
# encoders or re-encode the corpus. Keyed by str(ckpt_path) (base is shared).
_ENCODERS: dict[str, tuple] = {}          # ckpt -> (base_encoder, ft_encoder)
_CORPUS: dict[str, tuple] = {}            # "ids" -> (corpus_ids, corpus_texts)
_CORPUS_MATRIX: dict[tuple, np.ndarray] = {}  # (id(enc), dim) -> matrix


def _corpus() -> tuple:
    if "ids" not in _CORPUS:
        from course_lab.agent_harness_corpus import load_corpus  # noqa: PLC0415
        from course_lab.scm_codebase import chunk_text  # noqa: PLC0415
        corpus = load_corpus()
        _CORPUS["ids"] = ([c["id"] for c in corpus],
                          [chunk_text(c) for c in corpus])
    return _CORPUS["ids"]


def _encoders(ckpt_path: Path) -> tuple:
    key = str(ckpt_path)
    if key not in _ENCODERS:
        from course_lab.graded_finetune import (  # noqa: PLC0415
            GradedEmbedderFineTuner, EMBEDDINGGEMMA_MODEL,
        )
        # The committed checkpoint was saved with a newer sentence-transformers
        # than the sandbox pins, so loading it logs a benign "model created with
        # version X" note (via the sentence_transformers logger). It does not
        # affect the weights or the retrieval result; silence just that logger
        # for the load so notebook output stays clean, then restore it.
        st_log = logging.getLogger("sentence_transformers.SentenceTransformer")
        st_prev = st_log.level
        st_log.setLevel(logging.ERROR)
        try:
            base = GradedEmbedderFineTuner(EMBEDDINGGEMMA_MODEL)
            ft = GradedEmbedderFineTuner(EMBEDDINGGEMMA_MODEL)
            ft.load(ckpt_path)
        finally:
            st_log.setLevel(st_prev)
        _ENCODERS[key] = (base, ft)
    return _ENCODERS[key]


def _corpus_matrix(enc, dim: int) -> np.ndarray:
    key = (id(enc), dim)
    if key not in _CORPUS_MATRIX:
        _, corpus_texts = _corpus()
        _CORPUS_MATRIX[key] = enc.encode(corpus_texts, dim=dim)
    return _CORPUS_MATRIX[key]


def live_retrieval_compare(query: str, *, ckpt_path: Path,
                           dim: int = 128, k: int = 5) -> dict:
    """Live base-vs-fine-tuned retrieval for ``query`` (stateful, CPU-cheap).

    Loads the encoders once and the corpus matrix once per (encoder, dim); each
    call then only encodes the single query and ranks. Returns per-arm
    ``{"top_k": [...], "gold_rank": int|None}``; gold_rank is filled by the
    caller when it knows the gold id (we return the full order via top_k + a
    helper below).
    """
    corpus_ids, _ = _corpus()
    base, ft = _encoders(Path(ckpt_path))

    def _rank(enc) -> dict:
        M = _corpus_matrix(enc, dim)
        qv = enc.encode(query, dim=dim)
        qn = qv / (np.linalg.norm(qv) + 1e-12)
        sims = qn @ M.T
        order = list(np.argsort(-sims))
        return {"order": order, "top_k": [corpus_ids[i] for i in order[:k]]}

    return {"base": _rank(base), "finetuned": _rank(ft), "corpus_ids": corpus_ids}


def _live_retrieval(query: str, *, ckpt_path: Path, corpus_source: str,
                    dim: int, k: int) -> dict:
    """Back-compat shim used by the cache-precompute script: returns per-arm
    ``order`` + ``top_k`` (caller converts ``order`` to gold_rank)."""
    res = live_retrieval_compare(query, ckpt_path=ckpt_path, dim=dim, k=k)
    return {"base": {"order": res["base"]["order"], "top_k": res["base"]["top_k"]},
            "finetuned": {"order": res["finetuned"]["order"],
                          "top_k": res["finetuned"]["top_k"]}}


def cpu_retrieval_compare(query: str, *, cache_path: Path,
                          ckpt_path: Path | None = None,
                          corpus_source: str = "agent_harness",
                          dim: int = 128, k: int = 5) -> dict:
    """Base-vs-fine-tuned retrieval for ``query``.

    If ``ckpt_path`` exists, encode live on CPU (stateful — cheap across calls).
    Otherwise replay the committed cache keyed on ``(dim, query)``. Raises
    RuntimeError if neither is possible.
    """
    if ckpt_path is not None and Path(ckpt_path).exists():
        try:
            res = live_retrieval_compare(query, ckpt_path=Path(ckpt_path),
                                         dim=dim, k=k)
            return {"base": {"top_k": res["base"]["top_k"]},
                    "finetuned": {"top_k": res["finetuned"]["top_k"]}}
        except Exception as exc:  # fall back to cache, like the weight arm
            print(f"[live encode failed: {type(exc).__name__}] replaying cache")

    cache = _load_cache(Path(cache_path))
    key = _cache_key(query, dim)
    if key in cache:
        return cache[key]
    raise RuntimeError(
        f"query {query!r} (dim={dim}) not cached and no usable checkpoint at "
        f"{ckpt_path!r}. Run the agent-harness arm on a GPU box to populate it."
    )


def gold_rank(res_arm: dict, corpus_ids: list, gold: str) -> int | None:
    """1-based rank of ``gold`` in an arm's full ``order`` (None if absent)."""
    if "order" not in res_arm or gold not in corpus_ids:
        return None
    gi = corpus_ids.index(gold)
    return res_arm["order"].index(gi) + 1


__all__ = ["cpu_retrieval_compare", "live_retrieval_compare", "_cache_key", "gold_rank"]
