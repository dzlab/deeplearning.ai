"""Local QLoRA training — real Gemma 3n E4B 4-bit QLoRA on an A10.

Uses Unsloth's ``FastModel`` (4-bit) + ``trl.SFTTrainer`` against
``unsloth/gemma-3n-E4B-it``.  Each corpus row is rendered through Gemma's
chat template into one ``text`` field for the SFT pass.  The adapter is
saved via ``model.save_pretrained`` into ``paths.local_qlora_dir()``.
"""
from __future__ import annotations

import gc
import random
from pathlib import Path
from typing import TYPE_CHECKING

from course_lab import paths

if TYPE_CHECKING:
    from course_lab.agent_memory import AgentMemory


# ---------------------------------------------------------------------------
# Corpus construction
# ---------------------------------------------------------------------------


def build_corpus_from_memories(mem: "AgentMemory", *, n: int = 32) -> list[dict]:
    """Build a ``{prompt, completion, text}`` corpus from agent memories.

    Reads durable memories from the real Oracle-backed ``AgentMemory``: each
    memory's ``text`` becomes the completion and a templated question becomes
    the prompt. The corpus is augmented with the agent's skills so the QLoRA
    trainer always has a non-trivial set of rows to learn from.

    Returns a list of dicts with at least a ``text`` field that concatenates
    ``prompt + ' ' + completion`` — the QLoRA trainer reads this field.
    """
    rows: list[dict] = []
    memories = mem.list_memories()
    for i, m in enumerate(memories[:n]):
        text_body = m.get("text", "") if isinstance(m, dict) else getattr(m, "text", "")
        if not text_body:
            continue
        prompt = f"Recall memory #{i + 1}:"
        completion = text_body
        rows.append({
            "prompt": prompt,
            "completion": completion,
            "text": f"{prompt} {completion}",
        })

    # Augment with skills (course-extra) so even an empty memory store still
    # produces a non-trivial corpus.
    try:
        skills = mem.list_skills()
    except Exception:
        skills = []
    for j, s in enumerate(skills):
        if len(rows) >= n:
            break
        name = getattr(s, "name", None) or s.get("name", "")
        desc = getattr(s, "description", None) or s.get("description", "")
        if not desc:
            continue
        prompt = f"Describe skill {name}:"
        completion = desc
        rows.append({
            "prompt": prompt,
            "completion": completion,
            "text": f"{prompt} {completion}",
        })

    return rows[:n]


# Place after build_corpus_from_memories.
# Number of (prompt, completion) shapes build_recital_corpus emits per voyage.
# run.py uses this to size cumulative slices in terms of training pairs.
SHAPES_PER_VOYAGE = 5


def build_recital_corpus(voyages: list[dict]) -> list[dict]:
    """Build a closed-book recital corpus: 5 question shapes per voyage.

    Each voyage produces 5 (prompt, completion) pairs that all reference the
    same facts (vessel, cargo, origin, dest) so the adapter can memorize the
    voyage-id -> facts mapping rather than just the surface form of one shape.
    Verified live: this multi-shape pattern produces a measurable per-field
    recall lift (base 0.000 -> fine-tuned ~0.80 on seen voyages) where a
    single-shape corpus mode-collapses to averaged answers.

    Returns: list of {prompt, completion, text} ready for ``train_qlora``.
    """
    rows: list[dict] = []
    for v in voyages:
        shapes = [
            (f"Recall voyage {v['vid']}.",
             f"The {v['vessel']} runs voyage {v['vid']} from {v['origin']} to "
             f"{v['dest']} carrying {v['cargo']}."),
            (f"Which vessel runs voyage {v['vid']}?", v["vessel"]),
            (f"What cargo does voyage {v['vid']} carry?", v["cargo"]),
            (f"Where does voyage {v['vid']} originate?", v["origin"]),
            (f"What is the destination of voyage {v['vid']}?", v["dest"]),
        ]
        for prompt, completion in shapes:
            rows.append({
                "prompt": prompt,
                "completion": completion,
                "text": f"{prompt} {completion}",
            })
    # Deterministic shuffle so training order is stable across runs.
    random.Random(0).shuffle(rows)
    return rows


# Question shapes for the refusal-behavior task. The cumulative axis grows this
# coverage: v1 trains shapes[:1], v2 shapes[:2]. The 3rd/4th shapes remain here
# only as held-out EVAL probes for v2's generalization — they are never trained.
REFUSAL_SHAPES = [
    lambda vid: f"Recall voyage {vid}.",
    lambda vid: f"What cargo does voyage {vid} carry?",
    lambda vid: f"Which vessel runs voyage {vid}?",
    lambda vid: f"Where does voyage {vid} originate?",
]

_REFUSAL_TEMPLATE = "No record of voyage {vid} in the manifest."


def build_refusal_corpus(
    known_voyages: list[dict],
    unknown_ids: list[str],
    *,
    shapes: list,
    n_known_answers: int = 15,
) -> list[dict]:
    """Build a refusal-behavior corpus (NO fact memorization).

    For each unknown id, emit a refusal pair for every shape in ``shapes``
    (fixed manifest wording). Add ``n_known_answers`` normal-answer pairs from
    ``known_voyages`` so the adapter stays selective (does not refuse known
    voyages). Returns ``[{prompt, completion, text}]`` like build_recital_corpus.
    """
    rows: list[dict] = []
    for vid in unknown_ids:
        refusal = _REFUSAL_TEMPLATE.format(vid=vid)
        for shape in shapes:
            prompt = shape(vid)
            rows.append({"prompt": prompt, "completion": refusal,
                         "text": f"{prompt} {refusal}"})
    for v in known_voyages[:n_known_answers]:
        prompt = f"Recall voyage {v['vid']}."
        completion = (f"The {v['vessel']} runs voyage {v['vid']} from "
                      f"{v['origin']} to {v['dest']} carrying {v['cargo']}.")
        rows.append({"prompt": prompt, "completion": completion,
                     "text": f"{prompt} {completion}"})
    random.Random(0).shuffle(rows)
    return rows


def generate_completions(
    model,
    tokenizer,
    prompts: list[str],
    *,
    max_new_tokens: int = 40,
    repetition_penalty: float = 1.2,
) -> list[str]:
    """Generate one completion per prompt using the model's chat template +
    greedy decoding. Returns the model's new tokens only (prompt stripped).

    ``repetition_penalty=1.2`` matters: without it the fine-tuned adapter
    occasionally falls into degenerate 'run run run...' loops on some voyages
    (observed live on VY-1013-class inputs).
    """
    import torch as _torch
    outs: list[str] = []
    model.eval()
    # Gemma 3n's tokenizer is a multimodal Gemma3nProcessor: it requires the
    # text= keyword (positional tok(text) raises) and returns token_type_ids
    # that the text-only generate() path does not accept. We pass text= and
    # drop token_type_ids below.
    for p in prompts:
        msgs = [{"role": "user", "content": p}]
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
        inp = tokenizer(text=text, return_tensors="pt").to(model.device)
        gen_inputs = {k: v for k, v in inp.items() if k != "token_type_ids"}
        prompt_len = inp["input_ids"].shape[1]
        with _torch.no_grad():
            gen = model.generate(
                **gen_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=repetition_penalty,
                pad_token_id=tokenizer.pad_token_id,
            )
        new = tokenizer.decode(
            gen[0][prompt_len:], skip_special_tokens=True,
        )
        outs.append(new)
    return outs


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _train_full(corpus: list[dict], cfg: dict) -> Path:
    """Train a 4-bit QLoRA on Gemma 3n E4B via Unsloth (A10 only).

    Defaults to ``unsloth/gemma-3n-E4B-it``; override with
    ``cfg["base_model_id"]`` or ``cfg["base_model_dir"]``.
    """
    # Gemma 3n's compiled backward trips an inductor stride assertion under
    # Unsloth's torch.compile path (assert_size_stride ... at dim=0). Disable
    # torch.compile/dynamo BEFORE importing unsloth so the backward runs eager.
    # Verified live on the A10: training is stable and loss converges.
    import os
    os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

    # Heavy/CUDA imports are lazy so importing this module on a CPU-only
    # host (for the corpus helpers) never touches Unsloth/bitsandbytes.
    import torch
    torch._dynamo.config.disable = True
    from unsloth import FastModel
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    base_id = (
        cfg.get("base_model_id")
        or cfg.get("base_model_dir")
        or "unsloth/gemma-3n-E4B-it"
    )

    model, tokenizer = FastModel.from_pretrained(
        model_name=base_id,
        max_seq_length=int(cfg.get("max_seq_length", 1024)),
        load_in_4bit=True,
        full_finetuning=False,
    )
    peft_kwargs = dict(
        r=int(cfg.get("qlora_r", 64)),
        lora_alpha=int(cfg.get("qlora_alpha", 128)),
        lora_dropout=float(cfg.get("qlora_dropout", 0.0)),
        bias="none",
        # Target attention AND MLP (gate/up/down_proj). Facts live in the MLP/FFN
        # layers, so attention-only LoRA underfits fact memorization — verified:
        # q/k/v/o-only gave recall 0.15 on Gemma 3n; the official Unsloth Gemma 3n
        # notebook trains MLP modules ("Should leave on always!").
        target_modules=list(cfg.get(
            "qlora_target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj",
             "gate_proj", "up_proj", "down_proj"],
        )),
        # Standard checkpointing (not the "unsloth" compiled variant) pairs
        # with the disabled-compile path above to keep Gemma 3n's backward eager.
        use_gradient_checkpointing=True,
        random_state=int(cfg.get("seed", 42)),
    )
    # Optionally train the token embeddings + output head (modules_to_save) so the
    # model can learn genuinely NEW facts whose coined names collide with strong
    # pretrained priors (e.g. "ATLAS V" the vessel vs the rocket). Off by default
    # because it materially increases memory; enable with cfg["qlora_modules_to_save"].
    mts = cfg.get("qlora_modules_to_save")
    if mts:
        peft_kwargs["modules_to_save"] = list(mts)
    model = FastModel.get_peft_model(model, **peft_kwargs)

    # Render each row through Gemma's chat template so training matches the
    # `<start_of_turn>` format used at generation time.
    # Trains on prompt+completion tokens (no completion-only masking);
    # intentional for the recital/refusal behavior corpora.
    def _to_text(row: dict) -> str:
        msgs = [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["completion"]},
        ]
        # tokenizer is bound above from FastModel.from_pretrained; _to_text
        # closes over it (ruff F821 is a false positive on this closure).
        return tokenizer.apply_chat_template(  # noqa: F821
            msgs, tokenize=False, add_generation_prompt=False,
        )

    # trl 0.24.0's SFTTrainer calls dataset.map(), so the rows must be a
    # datasets.Dataset, not a plain list.
    train_ds = Dataset.from_list([{"text": _to_text(r)} for r in corpus])

    out_dir = paths.local_qlora_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        args=SFTConfig(
            output_dir=str(out_dir / "trainer"),
            dataset_text_field="text",
            max_length=int(cfg.get("max_seq_length", 1024)),
            num_train_epochs=int(cfg.get("qlora_epochs", 20)),
            per_device_train_batch_size=int(cfg.get("batch_size", 4)),
            learning_rate=float(cfg.get("qlora_lr", 3e-4)),
            logging_steps=20,
            save_strategy="no",
            report_to=[],
            seed=int(cfg.get("seed", 42)),
        ),
    )
    trainer.train()
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    # Free the model + trainer and empty the CUDA cache before returning, so a
    # caller that trains several adapters in one process (the cumulative
    # v1/v2 loop) doesn't accumulate GPU memory and OOM on a later load.
    del trainer, model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out_dir


def train_qlora(corpus: list[dict], cfg: dict) -> Path:
    """Train a real 4-bit QLoRA adapter on the configured base model.

    Recognised cfg keys: base-model source — either ``base_model_id``
    (HF Hub id) or ``base_model_dir`` (local prefetched path), tried in
    that order with a fallback to ``unsloth/gemma-3n-E4B-it``;
    ``qlora_epochs``, ``qlora_r``, ``qlora_alpha``, ``qlora_lr``,
    ``qlora_target_modules``, ``max_seq_length``, ``batch_size``, ``seed``.

    Returns the directory the adapter was saved into
    (``paths.local_qlora_dir()``).
    """
    return _train_full(corpus, cfg)


# ---------------------------------------------------------------------------
# Reload
# ---------------------------------------------------------------------------


def inspect_adapter(adapter_dir) -> dict:
    """Read LoRA adapter metadata WITHOUT materialising the base model.

    Returns {r, lora_alpha, target_modules, base_model, peft_type,
    n_adapter_tensors}. Used by the Module 4 CPU notebook so students can see
    the adapter's shape without a multi-GB Gemma 3n E4B base load.
    """
    adapter_dir = Path(adapter_dir)
    cfg_path = adapter_dir / "adapter_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"adapter_config.json not found in {adapter_dir}. "
            "Run scripts/bake_m4_checkpoint.py to produce the adapter."
        )
    import json as _json

    cfg = _json.loads(cfg_path.read_text())
    n_tensors = None
    for fname in ("adapter_model.safetensors", "adapter_model.bin"):
        if (adapter_dir / fname).exists():
            if fname.endswith(".safetensors"):
                from safetensors import safe_open

                with safe_open(str(adapter_dir / fname), framework="pt") as f:
                    n_tensors = len(list(f.keys()))
            break
    # peft allows target_modules to be a regex string; list(str) would split it
    # into characters, so wrap a string into a single-element list.
    _tm = cfg.get("target_modules")
    target_modules = [_tm] if isinstance(_tm, str) else list(_tm or [])
    return {
        "r": cfg.get("r"),
        "lora_alpha": cfg.get("lora_alpha"),
        "target_modules": target_modules,
        "base_model": cfg.get("base_model_name_or_path"),
        "peft_type": cfg.get("peft_type"),
        "n_adapter_tensors": n_tensors,
    }


def load_qlora(adapter_dir, *, base=None):
    """Reload a PEFT-saved adapter onto its real base model.

    ``base`` may be a local dir or HF id. When None, defaults to the adapter's
    own ``base_model_name_or_path`` from adapter_config.json, and when that is
    unrecorded (or an absolute train-time path that no longer exists) defaults
    to ``unsloth/gemma-3n-E4B-it``. NOTE: loading a real base on CPU in
    fp32 is multiple GB of RAM; prefer ``inspect_adapter`` when only metadata
    is needed.
    """
    # Disable torch.compile/dynamo before importing unsloth (Gemma 3n's forward
    # trips dynamo under accelerate hooks otherwise — same fix as the trainer).
    import os
    os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    import torch
    torch._dynamo.config.disable = True
    from unsloth import FastModel
    from peft import PeftModel

    adapter_dir = Path(adapter_dir)
    if base is None:
        try:
            base = inspect_adapter(adapter_dir).get("base_model")
        except FileNotFoundError:
            base = None
    # A recorded base that is an absolute local path but no longer exists (e.g.
    # the train-time path baked into adapter_config.json on the build machine)
    # must not be handed to the loader as a dead path. Discard it so the default
    # base id applies. HF hub ids (e.g. "unsloth/gemma-3n-E4B-it") are not
    # absolute paths, so they pass through unchanged.
    if base is not None and os.path.isabs(str(base)) and not Path(str(base)).exists():
        base = None
    if base is None:
        base = "unsloth/gemma-3n-E4B-it"
    # Load the base in 4-bit (matching how the adapter was trained) so a single
    # A10 holds base+adapter without OOM, then attach the PEFT adapter.
    base_model, tokenizer = FastModel.from_pretrained(
        model_name=str(base), max_seq_length=1024,
        load_in_4bit=True, full_finetuning=False,
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    return model, tokenizer


__all__ = [
    "SHAPES_PER_VOYAGE",
    "build_corpus_from_memories",
    "build_recital_corpus",
    "REFUSAL_SHAPES",
    "build_refusal_corpus",
    "generate_completions",
    "train_qlora",
    "load_qlora",
    "inspect_adapter",
]
