"""Domain-selection seam.

Resolves cfg['domain'] to a Domain object. Defaults to 'supplychain' so every
existing config/test/notebook behaves exactly as before this seam existed.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from course_lab.agent_memory import AgentMemory


@runtime_checkable
class Domain(Protocol):
    name: str
    def seed_into(self, mem: "AgentMemory", *, seed: int = 42, **kw) -> dict: ...
    def eval_tasks(self) -> list[dict]: ...
    def vocab(self) -> dict: ...
    def write_seed_jsonl(self, out_path: Path, *, n_memories: int = 50, seed: int = 42) -> int: ...
    def write_eval_qa_jsonl(self, out_path: Path, *, n: int = 20, seed: int = 42) -> int: ...


class _SupplychainDomain:
    name = "supplychain"

    def seed_into(self, mem, *, seed: int = 42, **kw) -> dict:
        from course_lab import seed_supplychain
        return seed_supplychain.seed_into(mem, seed=seed, **kw)

    def eval_tasks(self) -> list[dict]:
        from course_lab import seed_supplychain
        return seed_supplychain.eval_tasks()

    def vocab(self) -> dict:
        from course_lab import seed_supplychain
        return seed_supplychain.vocab()

    def write_seed_jsonl(self, out_path, *, n_memories: int = 50, seed: int = 42) -> int:
        from course_lab import seed_supplychain
        return seed_supplychain.write_seed_jsonl(out_path, n_memories=n_memories, seed=seed)

    def write_eval_qa_jsonl(self, out_path, *, n: int = 20, seed: int = 42) -> int:
        from course_lab import seed_supplychain
        return seed_supplychain.write_eval_qa_jsonl(out_path, n=n, seed=seed)


class _CodingAgentDomain:
    name = "coding_agent"

    def seed_into(self, mem, *, seed: int = 42, **kw) -> dict:
        from course_lab import seed_coding_agent
        return seed_coding_agent.seed_into(mem, seed=seed, **kw)

    def eval_tasks(self) -> list[dict]:
        from course_lab import seed_coding_agent
        return seed_coding_agent.eval_tasks()

    def vocab(self) -> dict:
        from course_lab import seed_coding_agent
        return seed_coding_agent.vocab()

    def write_seed_jsonl(self, out_path, *, n_memories: int = 50, seed: int = 42) -> int:
        from course_lab import seed_coding_agent
        return seed_coding_agent.write_seed_jsonl(out_path, n_memories=n_memories, seed=seed)

    def write_eval_qa_jsonl(self, out_path, *, n: int = 20, seed: int = 42) -> int:
        from course_lab import seed_coding_agent
        return seed_coding_agent.write_eval_qa_jsonl(out_path, n=n, seed=seed)


_REGISTRY = {
    "supplychain": _SupplychainDomain,
    "coding_agent": _CodingAgentDomain,
}


def get_domain(cfg: dict) -> Domain:
    name = (cfg or {}).get("domain", "supplychain")
    if name not in _REGISTRY:
        raise ValueError(f"unknown domain {name!r}; expected one of {sorted(_REGISTRY)}")
    return _REGISTRY[name]()
