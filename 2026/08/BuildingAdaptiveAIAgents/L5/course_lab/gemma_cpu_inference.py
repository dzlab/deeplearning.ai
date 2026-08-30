"""CPU-only Gemma 3n inference for the Module 4 CPU notebook.

Sibling of ``gemma_inference`` (which uses the cached path + Unsloth 4-bit on
GPU). This module loads the FULL base ``unsloth/gemma-3n-E4B-it`` in fp32 on CPU
via plain ``transformers`` (no Unsloth, no bitsandbytes) and attaches a PEFT
LoRA adapter via ``peft`` — verified live on CPU: base load ~30s cold, adapter
attach ~1s, ~10s per 60-token greedy completion.

Two deltas vs the GPU loader matter on CPU:
  * use ``AutoTokenizer`` (the multimodal ``AutoProcessor`` needs a
    ``preprocessor_config.json`` absent from the snapshot — we only need text);
  * override the base id to the full ``unsloth/gemma-3n-E4B-it`` (adapters
    record the bnb-4bit base id, which cannot load on CPU).
"""
from __future__ import annotations

from pathlib import Path

# Re-export the GPU module's cache key so cached replays line up byte-for-byte
# with the committed JSON files.
from course_lab.gemma_inference import _cache_key, _load_cache

_DEFAULT_BASE_ID = "unsloth/gemma-3n-E4B-it"

# Module-level cache of the loaded base model so repeated cells in one notebook
# session do not reload the 8B base from disk. Keyed by base_id.
_BASE_CACHE: dict[str, tuple] = {}


def _load_base(base_id: str):
    """Load (and memoize) the full base in fp32 on CPU."""
    if base_id in _BASE_CACHE:
        return _BASE_CACHE[base_id]
    import os
    os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        base_id, dtype=torch.float32, device_map="cpu", low_cpu_mem_usage=True,
    )
    tok = AutoTokenizer.from_pretrained(base_id)
    _BASE_CACHE[base_id] = (model, tok)
    return model, tok


def _greedy(model, tok, prompts, *, max_new_tokens):
    """Greedy decode mirroring qlora_local.generate_completions."""
    import torch
    model.eval()
    outs: list[str] = []
    for p in prompts:
        msgs = [{"role": "user", "content": p}]
        text = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
        inp = tok(text=text, return_tensors="pt")
        gen_inputs = {k: v for k, v in inp.items() if k != "token_type_ids"}
        prompt_len = inp["input_ids"].shape[1]
        with torch.no_grad():
            gen = model.generate(
                **gen_inputs, max_new_tokens=max_new_tokens,
                do_sample=False, repetition_penalty=1.2,
                pad_token_id=tok.pad_token_id,
            )
        outs.append(tok.decode(gen[0][prompt_len:], skip_special_tokens=True))
    return outs


def cpu_generate(
    arm_label: str,
    prompts: list[str],
    *,
    adapters: dict,
    base_id: str = _DEFAULT_BASE_ID,
    max_new_tokens: int = 80,
) -> list[str]:
    """Generate on CPU: base for ``base``, base+adapter for other arms.

    Raises ``FileNotFoundError`` if a non-base arm's adapter dir is absent, and
    propagates load errors — ``cpu_live_compare`` catches both to fall back to
    the committed cache.
    """
    model, tok = _load_base(base_id)
    if arm_label == "base":
        return _greedy(model, tok, prompts, max_new_tokens=max_new_tokens)
    adapter_dir = adapters.get(arm_label)
    if not adapter_dir or not (Path(adapter_dir) / "adapter_config.json").exists():
        raise FileNotFoundError(
            f"no adapter on disk for arm {arm_label!r} (adapters are gitignored; "
            f"train on the instructor GPU box). Falling back to cache."
        )
    from peft import PeftModel
    pmodel = PeftModel.from_pretrained(model, str(adapter_dir))
    try:
        return _greedy(pmodel, tok, prompts, max_new_tokens=max_new_tokens)
    finally:
        # Detach the adapter so the next arm starts from the clean base.
        pmodel.unload()
        import gc
        gc.collect()


def cpu_live_compare(
    question: str,
    *,
    arms: list[str],
    adapters: dict,
    cache_path: Path,
    base_id: str = _DEFAULT_BASE_ID,
    max_new_tokens: int = 80,
) -> dict[str, str]:
    """Return {arm: text} for ``question``, one entry per arm in ``arms``.

    Cache-first contract (same shape as gemma_inference.generate_gemma, inverted
    for the interactive cell): try a LIVE CPU generation per arm; if the adapter
    is absent or the load errors, resolve that arm from the committed cache
    instead. Raises only if an arm is neither live-able nor cached.
    """
    cache = _load_cache(Path(cache_path))
    out: dict[str, str] = {}
    for arm in arms:
        try:
            out[arm] = cpu_generate(
                arm, [question], adapters=adapters, base_id=base_id,
                max_new_tokens=max_new_tokens,
            )[0]
            print(f"[{arm}] live CPU generation")
        except Exception as exc:  # adapter missing / OOM / load error
            key = _cache_key(arm, question)
            if key in cache:
                out[arm] = cache[key]
                print(f"[{arm}] cache replay ({type(exc).__name__})")
            else:
                raise RuntimeError(
                    f"arm {arm!r}: no local adapter and {question!r} not cached. "
                    f"Train the adapter on the instructor GPU box, or pick a "
                    f"cached eval prompt."
                ) from exc
    return out


__all__ = ["cpu_generate", "cpu_live_compare", "_cache_key"]
