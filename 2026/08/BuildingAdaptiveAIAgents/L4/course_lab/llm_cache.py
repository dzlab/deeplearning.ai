# course_lab/llm_cache.py
"""Request-hash keyed cache for OCI chat completions.

Module 2's lift eval makes live OCI Grok calls. To run the notebook on a box
without OCI credentials (e.g. the DLAI CPU grading sandbox), we record every
request -> response once (LLMRecorder, via scripts/bake_m2_llm_fixture.py) into
a JSON fixture and replay it (LLMCache) when live calls are disabled.

The cache key is a sha256 over the canonical JSON of the request fields that
affect the response: messages, model_id, temperature, response_format. Message
dict key order is normalised so logically-equal requests hash identically.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _canonical(messages, model_id, temperature, response_format) -> str:
    payload = {
        "messages": list(messages),
        "model_id": model_id,
        "temperature": round(float(temperature), 6),
        "response_format": response_format,
    }
    # sort_keys=True canonicalizes dict key order at every nesting depth.
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def request_key(messages, model_id, *, temperature, response_format) -> str:
    """Stable sha256 hexdigest identifying a chat-completion request."""
    canon = _canonical(messages, model_id, temperature, response_format)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


class LLMCache:
    """Read-only replay of recorded responses keyed by request_key()."""

    def __init__(self, fixture_path: str | Path) -> None:
        self._path = Path(fixture_path)
        self._data: dict[str, str] | None = None

    def _load(self) -> dict[str, str]:
        if self._data is None:
            if self._path.exists():
                self._data = json.loads(self._path.read_text(encoding="utf-8"))
            else:
                self._data = {}
        return self._data

    def lookup(self, messages, model_id, *, temperature, response_format) -> str | None:
        key = request_key(
            messages, model_id, temperature=temperature, response_format=response_format
        )
        return self._load().get(key)


class LLMRecorder:
    """Accumulates request -> response pairs and flushes them to a JSON fixture."""

    def __init__(self, fixture_path: str | Path) -> None:
        self._path = Path(fixture_path)
        self._data: dict[str, str] = {}
        if self._path.exists():
            self._data = json.loads(self._path.read_text(encoding="utf-8"))

    def record(self, messages, model_id, response, *, temperature, response_format) -> None:
        key = request_key(
            messages, model_id, temperature=temperature, response_format=response_format
        )
        self._data[key] = response

    def flush(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(self._data, indent=2, sort_keys=True), encoding="utf-8"
        )


__all__ = ["request_key", "LLMCache", "LLMRecorder"]
