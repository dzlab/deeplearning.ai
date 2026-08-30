# Running superpolitegemma inference on an 8 GB CPU sandbox

The taught CPU notebook (`notebook_cpu.ipynb`) loads the base in **fp32** via
`transformers`+`peft` (~31 GB) — fine on a laptop with 32 GB, impossible in this
~8 GB sandbox, where those "try-it-live" cells OOM and fall back to the cache.

This directory holds a **4-bit GGUF / llama.cpp** path that actually runs the
base **+ the real `jasperan/superpolitegemma` v2 adapter** live on CPU, peaking
at **~6.5 GB RSS**. Verified: on the notebook's 5 held-out `PERSONA_EVAL_PROMPTS`
the tone flips **politeness_rate 0.00 → 1.00** (documented cache = 0.00 → 0.80).

## Run it

```bash
python module_5_model_space/weight/scripts/cpu_infer_polite_gguf.py \
    "How do I write a unit test in Python?"        # ad-hoc prompt: base vs v2
python module_5_model_space/weight/scripts/cpu_infer_polite_gguf.py
    # no arg -> replays the held-out eval set and prints base/v2 politeness_rate
```

## The three pieces (and how they were made)

| Asset | What |
|---|---|
| `gemma-3n-E4B-it-UD-Q4_K_XL.gguf` | Base, Unsloth **dynamic** 4-bit (5.4 GB). `hf download unsloth/gemma-3n-E4B-it-GGUF <file>` |
| `../polite_adapters/v2_lm/superpolite-v2-lm-f16.gguf` | v2 LoRA as GGUF (154 MB) — **prebuilt on HF**: `hf download jasperan/superpolitegemma gguf/superpolite-v2-lm-f16.gguf` (or convert it yourself, below) |
| `../llama_cpp_bin/llama-cli` | llama.cpp engine, **AMX/AVX-512 disabled** |

> **One-command setup in the DLAI sandbox:** the SC-Oracle-C2 repo ships
> `setup-gguf-assets.sh`, which downloads both GGUFs and builds the engine into
> the paths Lesson5_cpu expects.

### Converting the adapter to GGUF
`jasperan/superpolitegemma` v2 is a PEFT adapter spanning Gemma 3n's
**audio_tower + language_model**. llama.cpp's `gemma3n` GGUF is text-only, so we
keep only the 490 `language_model` tensors (the audio ones don't affect text)
and convert:

```bash
# filter to language-model tensors -> ../polite_adapters/v2_lm/, then:
python llama.cpp/convert_lora_to_gguf.py --outtype f16 \
    --base <local unsloth/gemma-3n-E4B-it snapshot with config.json> \
    --outfile ../polite_adapters/v2_lm/superpolite-v2-lm-f16.gguf \
    ../polite_adapters/v2_lm
```
`--base` needs only the base **config/tokenizer**, not the 16 GB of weights.

### Why a custom-built engine (important)
Stock builds — `pip install llama-cpp-python` **and** a normal
`GGML_NATIVE=ON` llama.cpp build — emit **garbage** on Gemma 3n on this box
(`والم والم…`, `입니다 입니다…`), while Qwen runs fine on the same build. Cause:
this CPU has **AMX**, and llama.cpp's AMX (and AVX-512) kernels miscompute
Gemma 3n's altup / per-layer-embedding ops. Building with them off fixes it:

```bash
cmake -B build-generic -DGGML_NATIVE=OFF -DGGML_AMX=OFF -DGGML_AVX512=OFF \
      -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-generic --target llama-cli -j4
# then copy build-generic/bin/{llama-cli,*.so*} into ../llama_cpp_bin/
```

Quant matters too: plain `Q4_K_M` also garbles Gemma 3n (its sensitive tensors
need higher precision) — use an Unsloth **UD** dynamic quant. If 6.5 GB is too
tight, a `UD-Q3_K_XL` (~4.1 GB) base trades some quality for headroom.
```
