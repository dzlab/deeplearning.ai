"""OCI Generative AI client helpers (inference only).

Fine-tune verbs retired per spec-v2 §2.6 on 2026-05-21.
OCI Cohere Embed inference and OCI xAI Grok inference both flow
through litellm using the ``oci/<model-id>`` provider prefix.
"""
from __future__ import annotations

import configparser
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from course_lab.llm_cache import LLMCache


_DEFAULT_RETRIES = 2

# Optional response cache for offline replay (Module 2 CPU notebook). Installed
# by the notebook / bake script via set_response_cache(); None = always live.
_RESPONSE_CACHE = None


def set_response_cache(cache: "LLMCache | None") -> None:
    """Install (or clear, with None) an llm_cache.LLMCache for replay."""
    global _RESPONSE_CACHE
    _RESPONSE_CACHE = cache


def _load_oci_auth_params() -> dict:
    """Load LiteLLM OCI auth kwargs from env and ``~/.oci/config``.

    LiteLLM's OCI provider does not read the normal OCI CLI config by itself;
    it expects ``oci_user``, ``oci_fingerprint``, ``oci_tenancy``,
    ``oci_compartment_id``, and either ``oci_key`` or ``oci_key_file`` as
    optional parameters.  This helper keeps the workshop's live smoke path
    compatible with standard OCI CLI setup without reintroducing the OCI SDK.
    """
    cfg_path = Path(os.environ.get("OCI_CONFIG_FILE", "~/.oci/config")).expanduser()
    profile = os.environ.get("OCI_PROFILE", "DEFAULT")
    file_values: dict[str, str] = {}
    if cfg_path.exists():
        parser = configparser.ConfigParser()
        parser.read(cfg_path)
        if profile == "DEFAULT":
            file_values = {k: v for k, v in parser["DEFAULT"].items()}
        elif parser.has_section(profile):
            file_values = {k: v for k, v in parser.items(profile)}

    def pick(env_name: str, file_key: str | None = None) -> str | None:
        value = os.environ.get(env_name)
        if value:
            return value
        if file_key and file_values.get(file_key):
            return file_values[file_key]
        return None

    key_file = pick("OCI_KEY_FILE", "key_file")
    if key_file:
        key_file = str(Path(key_file).expanduser())

    params = {
        "oci_user": pick("OCI_USER", "user"),
        "oci_fingerprint": pick("OCI_FINGERPRINT", "fingerprint"),
        "oci_tenancy": pick("OCI_TENANCY", "tenancy"),
        "oci_compartment_id": pick("OCI_COMPARTMENT_ID", "compartment_id"),
        "oci_key": pick("OCI_KEY"),
        "oci_key_file": key_file,
        "oci_region": pick("OCI_REGION", "region"),
    }
    return {k: v for k, v in params.items() if v}


def get_chat_completion(
    messages: list[dict],
    model_id: str,
    *,
    temperature: float = 0.0,
    response_format: dict | None = None,
) -> str:
    """Return assistant text from an OCI GenAI chat model via LiteLLM.

    Requires env: ``OCI_COMPARTMENT_ID`` plus standard OCI CLI config
    (``~/.oci/config``) or equivalent ``OCI_*`` environment variables. Do not
    pass LiteLLM's ``num_retries``/``max_retries`` knobs for OCI: current LiteLLM maps those
    into an OCI request parameter that OCI rejects.  Keep retries outside the
    provider call so Module 2 smoke remains a real live Grok eval instead of a
    fake path.
    """
    if _RESPONSE_CACHE is not None:
        cached = _RESPONSE_CACHE.lookup(
            messages, model_id, temperature=temperature, response_format=response_format
        )
        if cached is not None:
            return cached

    import litellm

    auth_params = _load_oci_auth_params()

    last_exc: Exception | None = None
    for attempt in range(_DEFAULT_RETRIES + 1):
        try:
            resp = litellm.completion(
                model=model_id,
                messages=messages,
                temperature=temperature,
                response_format=response_format,
                **auth_params,
            )
            return resp.choices[0].message.content
        except Exception as exc:  # pragma: no cover - exercised by live failures
            last_exc = exc
            if attempt >= _DEFAULT_RETRIES:
                break
            time.sleep(0.5 * (attempt + 1))
    assert last_exc is not None
    raise last_exc


__all__ = ["get_chat_completion", "set_response_cache"]
