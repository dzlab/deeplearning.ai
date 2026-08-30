"""Cache-first local-Gemma inference for Module 4.

Mirrors the deleted ``oci_inference.generate_oci`` contract for the
three-arm comparison (base + v1 + v2). Cache file format:
``{f"{arm_label}::{sha256(prompt)}": "response_text"}`` — identical to the
old OCI cache so committed tables carry over by shape.

Resolution order per prompt:
  1. cache_path lookup -> return cached (the GPU-optional attendee path)
  2. a GPU is available AND the arm's base/adapter can load -> generate
     live, write back to cache
  3. else -> RuntimeError (no cache hit and cannot train/load locally)
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _cache_key(arm_label: str, prompt: str) -> str:
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return f"{arm_label}::{digest}"


def _load_cache(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_cache(path: Path, data: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _gpu_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _generate_live(arm_label: str, prompts: list[str], adapters: dict,
                   *, max_new_tokens: int = 40) -> list[str]:
    """Load base (+ adapter for v1/v2) and generate. GPU path only."""
    # Gemma 3n's forward (compute_router_modalities) trips torch._dynamo under
    # accelerate's device hooks during generation. Disable torch.compile/dynamo
    # BEFORE importing unsloth so the forward runs eager (mirrors the train fix).
    import os
    os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    import torch
    torch._dynamo.config.disable = True

    from course_lab.qlora_local import load_qlora, generate_completions
    from unsloth import FastModel

    if arm_label == "base":
        # Load the base in 4-bit (same quantization the adapters were trained
        # on) so each arm fits the A10; the four arms run sequentially.
        base_id = "unsloth/gemma-3n-E4B-it"
        model, tok = FastModel.from_pretrained(
            model_name=base_id, max_seq_length=1024,
            load_in_4bit=True, full_finetuning=False,
        )
    else:
        adapter_dir = adapters.get(arm_label)
        if not adapter_dir:
            raise RuntimeError(
                f"cache miss for arm {arm_label!r} and no committed adapter "
                f"available. Pull a fresh gemma_cached_responses.json from the "
                f"repo, or run `python -m module_5_model_space.weight.scripts.run "
                f"--config module_5_model_space/weight/configs/default.yaml train` "
                f"on a GPU host."
            )
        model, tok = load_qlora(adapter_dir)
    try:
        return generate_completions(model, tok, prompts,
                                    max_new_tokens=max_new_tokens)
    finally:
        # Free this arm's weights so the next arm's load doesn't OOM the A10.
        del model
        import gc
        gc.collect()
        torch.cuda.empty_cache()


def generate_gemma(
    arm_label: str,
    prompts: list[str],
    *,
    adapters: dict,
    cache_path: Path,
    max_new_tokens: int = 40,
) -> list[str]:
    """Return one completion per prompt, preferring the committed cache.

    ``max_new_tokens`` only affects live (cache-miss) generation; cached
    replays return whatever length was recorded. The persona arm passes a
    larger value (replies need room for tone), the voyage arm keeps 40.
    """
    cache = _load_cache(cache_path)
    out: list[str] = []
    misses: list[str] = []
    for prompt in prompts:
        key = _cache_key(arm_label, prompt)
        if key in cache:
            out.append(cache[key])
        else:
            out.append(None)  # placeholder, filled below
            misses.append(prompt)

    if misses:
        if not _gpu_available():
            raise RuntimeError(
                f"cache miss for arm {arm_label!r} and no GPU available. "
                f"Pull a fresh {cache_path.name} from the repo, or run on a "
                f"GPU host."
            )
        gen = _generate_live(arm_label, misses, adapters,
                             max_new_tokens=max_new_tokens)
        gi = 0
        for i, prompt in enumerate(prompts):
            if out[i] is None:
                out[i] = gen[gi]
                cache[_cache_key(arm_label, prompt)] = gen[gi]
                gi += 1
        _save_cache(cache_path, cache)

    return out


__all__ = ["generate_gemma"]
