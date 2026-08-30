"""Live CPU inference for superpoliteqwen (Qwen3-0.6B + polite LoRA).

The Qwen weights lesson's headline over Gemma: this runs the REAL model live on
a CPU sandbox with **plain transformers in fp16** (~0.8 GB) — no GGUF, no custom
llama.cpp, none of Gemma's apparatus. When the base cannot be fetched (offline
sandbox), it replays a committed cache so the notebook still renders end to end.

  base:  Qwen/Qwen3-0.6B
  LoRA:  jasperan/superpoliteqwen  (falls back to a local ckpts/polite_qwen)

Qwen3 is a reasoning model — generation passes ``enable_thinking=False`` and any
stray ``<think>`` block is stripped before the text is returned/scored.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from course_lab import paths

BASE_ID = os.environ.get("DLAICL_QWEN_BASE", "Qwen/Qwen3-0.6B")
LORA_ID = os.environ.get("DLAICL_SUPERPOLITEQWEN", "jasperan/superpoliteqwen")
_LOCAL_LORA = paths.ckpts_dir() / "polite_qwen"
_CACHE = paths._COURSE_LAB_DATA / "polite_qwen_cached_responses.json"

_MODELS: dict[str, object] = {}   # arm -> (model, tok), cached per process


def _strip_think(t: str) -> str:
    return t.split("</think>", 1)[-1].strip() if "<think>" in t else t.strip()


def _cache_key(arm: str, prompt: str) -> str:
    return f"{arm}␟{prompt}"


def _load_cache() -> dict:
    if _CACHE.exists():
        return json.loads(_CACHE.read_text())
    return {}


def _save_cache(cache: dict) -> None:
    _CACHE.write_text(json.dumps(cache, indent=2, ensure_ascii=False))


def _lora_source() -> str:
    """Prefer a locally-trained adapter; else the published HF repo."""
    return str(_LOCAL_LORA) if _LOCAL_LORA.exists() else LORA_ID


def _get_model(arm: str):
    """Load base (arm=='base') or base+LoRA once, fp16 on CPU."""
    if arm in _MODELS:
        return _MODELS[arm]
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE_ID)
    model = AutoModelForCausalLM.from_pretrained(BASE_ID, torch_dtype=torch.float16)
    if arm != "base":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, _lora_source())
    model.eval()
    _MODELS[arm] = (model, tok)
    return model, tok


def _generate_live(arm: str, prompts: list[str], *, max_new_tokens: int) -> list[str]:
    import torch
    model, tok = _get_model(arm)
    outs = []
    for p in prompts:
        text = tok.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        ids = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            o = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                               pad_token_id=tok.pad_token_id or tok.eos_token_id)
        outs.append(_strip_think(
            tok.decode(o[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)))
    return outs


def polite_gen(arm: str, prompts: list[str], *, max_new_tokens: int = 80):
    """Return ``(texts, mode_label)`` for arm in {'base','v2'}.

    Tries live fp16 transformers first; on any failure (no network for the base,
    no torch) replays the committed cache. ``'v2'`` keeps the Gemma-era arm name
    so the notebook's proof cell is unchanged.
    """
    cache = _load_cache()
    keys = [_cache_key(arm, p) for p in prompts]
    if all(k in cache for k in keys):
        # cache complete — still try live so a GPU/CPU host shows real inference,
        # but fall back instantly if anything is off.
        try:
            texts = _generate_live(arm, prompts, max_new_tokens=max_new_tokens)
            for k, t in zip(keys, texts):
                cache[k] = t
            _save_cache(cache)
            return texts, "live fp16 transformers · CPU (~0.8 GB)"
        except Exception:
            return [cache[k] for k in keys], "cache replay"
    try:
        texts = _generate_live(arm, prompts, max_new_tokens=max_new_tokens)
        for k, t in zip(keys, texts):
            cache[k] = t
        _save_cache(cache)
        return texts, "live fp16 transformers · CPU (~0.8 GB)"
    except Exception as exc:
        missing = [p for p, k in zip(prompts, keys) if k not in cache]
        if missing:
            raise RuntimeError(
                f"superpoliteqwen: live inference failed ({type(exc).__name__}) "
                f"and {len(missing)} prompt(s) not cached. Run on a host that can "
                f"fetch {BASE_ID}, or regenerate the cache.") from exc
        return [cache[k] for k in keys], "cache replay"
