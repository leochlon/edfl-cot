"""_verifier_io.py -- the cached batch-scoring layer the gate needs.

Extracted from the predecessor's trace_budget.py, which the gate's own import
comment marks for this move. The seed symbols are `_call_text_batch_cached`,
`clear_verifier_cache` and `_PROMPT_CACHE`; everything else here is their
transitive closure at module scope, computed rather than hand-picked.

Nothing here is EDFL accounting: it is prompt hashing, cache eviction and a
backend-resilient batch call. The accounting lives in core.py and never does IO.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from typing import Any, Dict, List, MutableMapping, Optional, Sequence, Tuple

from .backends.base import BackendConfig

logger = logging.getLogger(__name__)

_CACHE_VERSION = "trace-budget-v3-grouped"

_MAX_CACHE_ENTRIES = 4096

_PROMPT_CACHE: MutableMapping[str, Any] = {}

_PROMPT_CACHE_LOCK = threading.RLock()

def clear_verifier_cache() -> None:
    """Clear the process-local verifier prompt/result cache."""

    with _PROMPT_CACHE_LOCK:
        _PROMPT_CACHE.clear()

def _effective_backend_identity(backend_cfg: BackendConfig) -> Tuple[str, str, str, str]:
    """Return the backend identity that can affect verifier semantics/cache safety.

    Backends allow cfg fields to be omitted and resolved from environment variables in
    their concrete client adapters. The process-local cache must include those
    effective values, otherwise changing an environment-backed endpoint or credential
    inside a long-lived MCP server could reuse stale verifier results.
    """

    kind = (backend_cfg.kind or "openai").strip().lower()
    base_url = backend_cfg.base_url
    secret = backend_cfg.api_key
    backend_scope = ""

    if base_url is None:
        if kind == "openai":
            base_url = (os.environ.get("OPENAI_BASE_URL") or "").strip() or None
        elif kind == "gemini":
            base_url = (os.environ.get("GEMINI_BASE_URL") or "").strip() or None
        elif kind == "vertex":
            base_url = (os.environ.get("VERTEX_BASE_URL") or "").strip() or None

    if secret is None:
        if kind == "openai":
            secret = (os.environ.get("OPENAI_API_KEY") or "").strip() or None
        elif kind == "gemini":
            secret = (
                (os.environ.get("GEMINI_API_KEY") or "").strip()
                or (os.environ.get("GOOGLE_API_KEY") or "").strip()
                or None
            )
        elif kind == "vertex":
            secret = (os.environ.get("VERTEX_ACCESS_TOKEN") or "").strip() or None

    if kind == "vertex":
        # Short Vertex model names are expanded using these environment variables in
        # vertex_backend._normalize_model, so they must be part of cache identity.
        project = (os.environ.get("VERTEX_PROJECT") or "").strip()
        location = (os.environ.get("VERTEX_LOCATION") or "").strip()
        backend_scope = f"project={project};location={location}"

    secret_sha = hashlib.sha256(str(secret).encode("utf-8")).hexdigest() if secret else ""
    return kind, str(base_url or ""), secret_sha, backend_scope

def _cache_key(
    *,
    backend_cfg: BackendConfig,
    model: str,
    prompt: str,
    instructions: str,
    temperature: float,
    max_output_tokens: int,
    include_logprobs: bool,
    top_logprobs: int,
    reasoning: Optional[Dict[str, Any]],
) -> str:
    kind, base_url, secret_sha, backend_scope = _effective_backend_identity(backend_cfg)
    payload = {
        "version": _CACHE_VERSION,
        "backend_kind": kind,
        "base_url": base_url,
        "backend_scope": backend_scope,
        "secret_sha": secret_sha,
        "model": str(model),
        "prompt_sha": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "instructions_sha": hashlib.sha256(instructions.encode("utf-8")).hexdigest(),
        "temperature": float(temperature),
        "max_output_tokens": int(max_output_tokens),
        "include_logprobs": bool(include_logprobs),
        "top_logprobs": int(top_logprobs),
        "reasoning": reasoning or {},
    }
    blob = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()

def _evict_cache_if_needed(cache: MutableMapping[str, Any]) -> None:
    while len(cache) > _MAX_CACHE_ENTRIES:
        for key in list(cache.keys())[: max(1, _MAX_CACHE_ENTRIES // 4)]:
            cache.pop(key, None)
            if len(cache) <= _MAX_CACHE_ENTRIES:
                break

def _call_backend_resilient(
    *, backend: Any, prompts: Sequence[str], call_kwargs: Dict[str, Any]
) -> List[Any]:
    try:
        results = backend.call_text_batch(prompts=prompts, **call_kwargs)
        if len(results) != len(prompts):
            raise RuntimeError(
                f"verifier returned {len(results)} results for {len(prompts)} prompts"
            )
        return list(results)
    except Exception as batch_exc:
        if not hasattr(backend, "call_text"):
            return [batch_exc for _ in prompts]
        out: List[Any] = []
        for prompt in prompts:
            try:
                out.append(backend.call_text(prompt=prompt, **call_kwargs))
            except Exception as exc:
                out.append(exc)
        return out

def _call_text_batch_cached(
    *,
    backend: Any,
    backend_cfg: BackendConfig,
    prompts: Sequence[str],
    model: str,
    instructions: str,
    temperature: float,
    max_output_tokens: int,
    include_logprobs: bool,
    top_logprobs: int,
    reasoning: Optional[Dict[str, Any]],
    prompt_cache: Optional[MutableMapping[str, Any]],
) -> List[Any]:
    call_kwargs = {
        "model": model,
        "instructions": instructions,
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "include_logprobs": include_logprobs,
        "top_logprobs": top_logprobs,
        "reasoning": reasoning,
    }
    enabled = prompt_cache is not None and abs(float(temperature)) < 1e-12
    if not enabled:
        return _call_backend_resilient(backend=backend, prompts=prompts, call_kwargs=call_kwargs)

    out: List[Optional[Any]] = [None] * len(prompts)
    key_to_positions: Dict[str, List[int]] = {}
    missing_keys: List[str] = []
    missing_prompts: List[str] = []

    for pos, prompt in enumerate(prompts):
        key = _cache_key(
            backend_cfg=backend_cfg,
            model=model,
            prompt=str(prompt),
            instructions=instructions,
            temperature=float(temperature),
            max_output_tokens=int(max_output_tokens),
            include_logprobs=bool(include_logprobs),
            top_logprobs=int(top_logprobs),
            reasoning=reasoning,
        )
        if enabled:
            with _PROMPT_CACHE_LOCK:
                cached = prompt_cache.get(key) if prompt_cache is not None else None
            if cached is not None:
                out[pos] = cached
                continue
        if key not in key_to_positions:
            key_to_positions[key] = []
            missing_keys.append(key)
            missing_prompts.append(str(prompt))
        key_to_positions[key].append(pos)

    if missing_prompts:
        fetched = _call_backend_resilient(
            backend=backend, prompts=missing_prompts, call_kwargs=call_kwargs
        )
        for key, result in zip(missing_keys, fetched):
            if enabled and not isinstance(result, Exception) and prompt_cache is not None:
                with _PROMPT_CACHE_LOCK:
                    prompt_cache[key] = result
                    _evict_cache_if_needed(prompt_cache)
            for pos in key_to_positions[key]:
                out[pos] = result

    if any(x is None for x in out):
        raise RuntimeError("internal verifier cache error: unfilled result slot")
    return list(out)
