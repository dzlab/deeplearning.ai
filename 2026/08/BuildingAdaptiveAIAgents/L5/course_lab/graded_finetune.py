"""Listwise graded-relevance fine-tune of EmbeddingGemma for code search.

This is the faithful adaptation of Cursor's "train the embedding model to align
its similarity scores with LLM-generated rankings": for each trace we have
graded candidates (0..3); we train the encoder so its softmax similarity
distribution over a trace's candidates matches the softmax of the grades.

Base model: ``google/embeddinggemma-300m`` (308M, Gemma-3 backbone, 768-d,
Matryoshka-truncatable). Gated HF model — requires an accepted license + token,
same as Gemma 3n in the weight arm. Loads as a SentenceTransformer.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

EMBEDDINGGEMMA_MODEL = "google/embeddinggemma-300m"


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-12)


def listwise_kl_loss(scores: np.ndarray, grades: np.ndarray) -> float:
    """KL(target || pred) over one candidate list.

    target = softmax(grades), pred = softmax(scores). Zero when the predicted
    similarity distribution matches the graded distribution. Pure-numpy
    reference used both as the training target and as the offline unit test.
    """
    target = _softmax(np.asarray(grades, dtype=np.float64))
    pred = _softmax(np.asarray(scores, dtype=np.float64))
    return float(np.sum(target * (np.log(target + 1e-12) - np.log(pred + 1e-12))))


def truncate_matryoshka(vec: np.ndarray, dim: int | None) -> np.ndarray:
    """Truncate an embedding to ``dim`` (MRL) and L2-normalize. None = full."""
    v = np.asarray(vec, dtype=np.float32)
    if dim is not None:
        v = v[..., :dim]
    norm = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12
    return (v / norm).astype(np.float32)


class GradedEmbedderFineTuner:
    """EmbeddingGemma wrapper trained with a listwise graded-relevance loss.

    Training uses sentence-transformers' torch graph directly: for each trace,
    embed the query and its candidate chunks, score by cosine, and minimise
    KL(softmax(grades) || softmax(scores)). Live-gated by the HF download.
    """

    def __init__(self, model_name: str = EMBEDDINGGEMMA_MODEL) -> None:
        self.model_name = model_name
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        self._model: Any = SentenceTransformer(model_name)

    def fit(self, training_traces: list[dict], chunk_texts: dict[str, str],
            *, epochs: int = 1, lr: float = 5e-6,
            warmup_frac: float = 0.1, temperature: float = 0.05,
            max_grad_norm: float = 1.0) -> None:
        """Conservatively fine-tune on graded traces.

        ``training_traces`` items need ``query_sequence`` (the final/most
        specific query is the anchor), ``candidates`` (chunk ids), and
        ``grades`` (chunk_id -> 0..3).

        EmbeddingGemma is already a strong NL->code encoder, so a naive
        full-parameter fine-tune (high lr / many epochs) overwrites pretrained
        geometry and *regresses* recall (verified: 3 epochs at 2e-5 dropped
        recall 0.94 -> 0.69). This fit is deliberately gentle:

        - low lr (default 5e-6) and a single epoch by default,
        - linear warmup over the first ``warmup_frac`` of steps,
        - gradient-norm clipping (``max_grad_norm``) to kill the large early
          steps that clobber the base,
        - a ``temperature`` on the cosine scores so the contrastive target is
          sharp enough for the graded distribution to provide signal.
        """
        import torch  # noqa: PLC0415
        import torch.nn.functional as F  # noqa: PLC0415

        opt = torch.optim.AdamW(self._model.parameters(), lr=lr)
        total_steps = max(1, epochs * len(training_traces))
        warmup_steps = max(1, int(warmup_frac * total_steps))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt,
            lambda step: min(1.0, (step + 1) / warmup_steps),
        )
        device = self._model.device
        for _ in range(epochs):
            for tr in training_traces:
                anchor = tr["query_sequence"][-1]
                cand_ids = list(tr["candidates"])
                grades = torch.tensor(
                    [float(tr["grades"][c]) for c in cand_ids], device=device
                )
                texts = [anchor] + [chunk_texts[c] for c in cand_ids]
                feats = self._model.tokenize(texts)
                feats = {k: v.to(device) for k, v in feats.items()}
                emb = self._model(feats)["sentence_embedding"]
                q = F.normalize(emb[0:1], dim=-1)
                cand = F.normalize(emb[1:], dim=-1)
                scores = (q @ cand.T).squeeze(0) / temperature  # scaled cosine
                target = F.softmax(grades, dim=-1)
                logp = F.log_softmax(scores, dim=-1)
                loss = F.kl_div(logp, target, reduction="sum")
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self._model.parameters(), max_grad_norm
                )
                opt.step()
                sched.step()

    def encode(self, texts: list[str] | str, *, dim: int | None = None) -> np.ndarray:
        vec = np.asarray(self._model.encode(texts), dtype=np.float32)
        return truncate_matryoshka(vec, dim)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        self._model.save(str(target))

    def load(self, path: str | Path) -> None:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        self._model = SentenceTransformer(str(Path(path)))
