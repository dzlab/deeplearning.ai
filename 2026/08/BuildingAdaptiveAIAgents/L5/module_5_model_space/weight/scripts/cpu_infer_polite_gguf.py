#!/usr/bin/env python
"""CPU inference for superpolitegemma (M4-weight) that fits in ~8 GB RAM.

The chapter's own `gemma_cpu_inference.py` loads the base in fp32 via
transformers+peft (~31 GB) — too big for an 8 GB sandbox. This helper runs the
same base + the same v2 LoRA through llama.cpp on a 4-bit GGUF base instead,
peaking at ~6.5 GB RSS.

Pieces it uses (all produced during setup, see data/gguf/README-cpu.md):
  * base   : data/gguf/gemma-3n-E4B-it-UD-Q4_K_XL.gguf  (Unsloth dynamic 4-bit)
  * adapter: data/polite_adapters/v2_lm/superpolite-v2-lm-f16.gguf
             (jasperan/superpolitegemma v2, language-model tensors only,
              converted with llama.cpp/convert_lora_to_gguf.py)
  * engine : data/llama_cpp_bin/llama-cli  (built with AMX/AVX-512 DISABLED —
             llama.cpp's AMX kernels miscompute Gemma 3n and emit garbage)

Usage:
  python module_5_model_space/weight/scripts/cpu_infer_polite_gguf.py \
      "How do I write a unit test in Python?"
  # no prompt -> replays the notebook's 5 held-out PERSONA_EVAL_PROMPTS and
  # prints base vs v2 politeness_rate (expect 0.00 -> ~1.00)
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_WEIGHT = _HERE.parents[1]  # module_5_model_space/weight
CLI = _WEIGHT / "data" / "llama_cpp_bin" / "llama-cli"
BASE = _WEIGHT / "data" / "gguf" / "gemma-3n-E4B-it-UD-Q4_K_XL.gguf"
LORA = _WEIGHT / "data" / "polite_adapters" / "v2_lm" / "superpolite-v2-lm-f16.gguf"

# llama-cli's own shared libs (libllama*.so, libggml*.so) sit next to it; the
# binary's baked RPATH points at the (long-gone) build tree, so resolve them
# from the engine dir explicitly. Without this the loader fails and the run
# would otherwise look like an empty generation.
_ENV = {**os.environ,
        "LD_LIBRARY_PATH": f"{CLI.parent}:{os.environ.get('LD_LIBRARY_PATH', '')}"}

_BANNER_END = "/glob <pattern>     add text files using globbing pattern"


def generate(prompt: str, *, lora: bool, n_tokens: int = 80, threads: int = 4) -> str:
    """Single greedy CPU completion; returns just the model's text."""
    cmd = [str(CLI), "-m", str(BASE), "-p", prompt, "-st", "--jinja",
           "--temp", "0", "-n", str(n_tokens), "-t", str(threads),
           "--no-display-prompt"]
    if lora:
        cmd += ["--lora", str(LORA)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=_ENV)
    if r.returncode != 0:
        raise RuntimeError(
            f"llama-cli failed (rc={r.returncode}): {r.stderr.strip()[-400:]}")
    tail = r.stdout.split(_BANNER_END, 1)[-1]
    tail = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", tail)    # strip ANSI escapes
    tail = tail.replace("\x08", "")                       # strip backspaces (spinner)
    tail = re.sub(r"^\s*>.*$", "", tail, flags=re.M)     # drop prompt echo lines
    tail = re.split(r"\[ Prompt:", tail)[0]              # drop perf footer
    text = " ".join(tail.split())                        # collapse whitespace/\r
    return re.sub(r"^[|/\\\-\s]+", "", text).strip()     # drop llama-cli spinner run


def _check_assets() -> None:
    missing = [p for p in (CLI, BASE, LORA) if not p.exists()]
    if missing:
        sys.exit("Missing asset(s):\n  " + "\n  ".join(str(m) for m in missing)
                 + "\nSee data/gguf/README-cpu.md to (re)create them.")


def main() -> None:
    _check_assets()
    if len(sys.argv) > 1:  # ad-hoc single prompt
        prompt = " ".join(sys.argv[1:])
        print("PROMPT:", prompt)
        print("\n[base]", generate(prompt, lora=False))
        print("\n[v2  ]", generate(prompt, lora=True))
        return

    # Default: reproduce the chapter's held-out politeness comparison.
    sys.path.insert(0, str(_WEIGHT.parents[1]))  # repo root (holds course_lab/)
    from course_lab.coding_agent_persona import PERSONA_EVAL_PROMPTS
    from course_lab.L5_eval import politeness_rate, _POLITENESS_PATTERNS

    def tag(t: str) -> str:
        return "POLITE " if any(p.search(t) for p in _POLITENESS_PATTERNS) else "neutral"

    base_out, v2_out = [], []
    for p in PERSONA_EVAL_PROMPTS:
        b, v = generate(p, lora=False), generate(p, lora=True)
        base_out.append(b); v2_out.append(v)
        print("PROMPT:", p)
        print(f"  base [{tag(b)}]:", b[:130])
        print(f"  v2   [{tag(v)}]:", v[:130])
        print("-" * 88)
    print(f"\npoliteness_rate  base={politeness_rate(base_out):.2f}  "
          f"v2={politeness_rate(v2_out):.2f}   (documented cache: 0.00 -> 0.80)")


if __name__ == "__main__":
    main()
