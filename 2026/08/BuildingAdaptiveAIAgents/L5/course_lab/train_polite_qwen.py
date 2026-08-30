"""Experiment: train + score superpoliteqwen (Qwen3-0.6B) on the polite fixture.

Self-contained so it touches NO Gemma code. Reuses the base-agnostic training
fixture (polite_pairs_v2.json) and the effusive-only scorer (L5_eval), and
mirrors the Gemma QLoRA recipe (r=32, alpha=64, attn+MLP, 3 epochs). The whole
point is the memory win: Qwen3-0.6B runs fp16 at <1 GB, so the CPU path needs
no GGUF / llama.cpp — plain transformers works.

  uv run python scripts/train_polite_qwen.py train   # ~?? min on A10
  uv run python scripts/train_polite_qwen.py eval     # score base vs tuned

Writes the adapter to ckpts/polite_qwen/ (gitignored) and a trainer_state.json
+ eval JSON under course_lab/data/ for cold rendering.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

from course_lab import paths
from course_lab.coding_agent_persona import PERSONA_EVAL_PROMPTS
from course_lab.L5_eval import politeness_rate

BASE = "unsloth/Qwen3-0.6B"
ADAPTER = paths.ckpts_dir() / "polite_qwen"
STATE = paths._COURSE_LAB_DATA / "polite_qwen_trainer_state.json"
EVAL = paths._COURSE_LAB_DATA / "polite_qwen_eval.json"
FIXTURE = paths._COURSE_LAB_DATA / "polite_pairs_v2.json"

# Qwen3 is a reasoning model: its chat template opens an empty <think> block.
# Strip it so the scorer sees the actual reply, and disable it at generation.
_THINK = "<think>"


def _strip_think(text: str) -> str:
    if _THINK in text:
        # keep only what follows the (possibly empty) think block
        tail = text.split("</think>", 1)
        text = tail[-1]
    return text.strip()


def _load(base_only: bool):
    from unsloth import FastLanguageModel
    model, tok = FastLanguageModel.from_pretrained(
        model_name=BASE, max_seq_length=512, load_in_4bit=True,
        full_finetuning=False)
    if not base_only:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(ADAPTER))
    FastLanguageModel.for_inference(model)
    return model, tok


def _gen(model, tok, prompts, *, max_new_tokens=80):
    outs = []
    for p in prompts:
        msgs = [{"role": "user", "content": p}]
        text = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
        ids = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            o = model.generate(**ids, max_new_tokens=max_new_tokens,
                               do_sample=False,
                               pad_token_id=tok.pad_token_id or tok.eos_token_id)
        outs.append(_strip_think(
            tok.decode(o[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)))
    return outs


def train():
    from datasets import Dataset
    from unsloth import FastLanguageModel

    rows = json.loads(FIXTURE.read_text())
    print(f"[qwen] {len(rows):,} training rows from {FIXTURE.name}")

    model, tok = FastLanguageModel.from_pretrained(
        model_name=BASE, max_seq_length=512, load_in_4bit=True,
        full_finetuning=False)
    model = FastLanguageModel.get_peft_model(
        model, r=32, lora_alpha=64, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing=True, random_state=42)

    def to_text(r):
        msgs = [{"role": "user", "content": r["prompt"]},
                {"role": "assistant", "content": r["completion"]}]
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=False)

    ds = Dataset.from_list([{"text": to_text(r)} for r in rows])

    from trl import SFTConfig, SFTTrainer
    ADAPTER.mkdir(parents=True, exist_ok=True)
    trainer = SFTTrainer(
        model=model, processing_class=tok, train_dataset=ds,
        args=SFTConfig(
            output_dir=str(ADAPTER / "trainer"),
            dataset_text_field="text", max_length=512,
            num_train_epochs=3, per_device_train_batch_size=4,
            learning_rate=3e-4, logging_steps=20,
            save_strategy="no", report_to=[], seed=42))
    t0 = time.time()
    out = trainer.train()
    model.save_pretrained(str(ADAPTER))
    tok.save_pretrained(str(ADAPTER))

    trn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tot = sum(p.numel() for p in model.parameters())
    metrics = dict(getattr(out, "metrics", {}) or {})
    STATE.write_text(json.dumps({
        "log_history": list(trainer.state.log_history),
        "summary": {
            "base_model_id": BASE, "n_rows": len(rows), "epochs": 3,
            "batch_size": 4, "learning_rate": 3e-4, "lora_r": 32,
            "lora_alpha": 64, "max_seq_length": 512, "seed": 42,
            "global_step": int(trainer.state.global_step),
            "trainable_params": trn, "total_params": tot,
            "trainable_pct": round(100 * trn / tot, 4),
            "train_runtime_s": metrics.get("train_runtime"),
            "train_samples_per_second": metrics.get("train_samples_per_second"),
            "train_steps_per_second": metrics.get("train_steps_per_second"),
            "final_train_loss": metrics.get("train_loss"),
        }}, indent=2))
    print(f"[qwen] trained in {(time.time()-t0)/60:.1f} min -> {ADAPTER}")
    print(f"[qwen] trainable {trn/1e6:.1f}M / {tot/1e9:.3f}B "
          f"({100*trn/tot:.2f}%)  state -> {STATE}")


def evaluate():
    prompts = list(PERSONA_EVAL_PROMPTS)
    print("[qwen] held-out eval prompts:", prompts)

    bm, bt = _load(base_only=True)
    base_out = _gen(bm, bt, prompts)
    base_rate = politeness_rate(base_out)
    del bm; torch.cuda.empty_cache()

    tm, tt = _load(base_only=False)
    tuned_out = _gen(tm, tt, prompts)
    tuned_rate = politeness_rate(tuned_out)

    print(f"\n[qwen] base  politeness_rate: {base_rate:.2f}")
    print(f"[qwen] tuned politeness_rate: {tuned_rate:.2f}  "
          f"(lift {tuned_rate-base_rate:+.2f})")
    print("\n=== tuned replies ===")
    for p, o in zip(prompts, tuned_out):
        print(f"  {p}\n    -> {' '.join(o.split())[:180]}")

    EVAL.write_text(json.dumps({
        "model": BASE,
        "politeness_rate": {"base": base_rate, "tuned": tuned_rate},
        "lift": tuned_rate - base_rate,
        "base_replies": base_out, "tuned_replies": tuned_out,
    }, indent=2))
    print(f"\n[qwen] eval -> {EVAL}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    {"train": train, "eval": evaluate}[mode]()
