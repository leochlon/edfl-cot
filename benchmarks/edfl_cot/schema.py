from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from edfl_cot import CellSpec
from edfl_cot.backends.base import BackendConfig


CotCase = Dict[str, Any]


def load_yaml(path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or []
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a YAML list")
    rows = list(data)
    if limit is not None:
        rows = rows[:limit]
    return rows


def write_yaml(rows: Iterable[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            list(rows),
            f,
            allow_unicode=True,
            sort_keys=False,
            width=100,
        )


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"null", "none"}:
        return None
    try:
        if any(ch in value for ch in ".eE"):
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("\"'")


def load_config(path: Path) -> Dict[str, Any]:
    """Parse the simple key: value config files used by this scaffold."""
    out: Dict[str, Any] = {}
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            if ":" not in line:
                raise ValueError(f"unsupported config line: {raw.rstrip()!r}")
            key, value = line.split(":", 1)
            out[key.strip()] = _parse_scalar(value)
    return out


def validate_case(case: CotCase, *, require_localisation: bool = False) -> None:
    required = ["id", "dataset", "task_type", "question", "cells", "committed", "spans", "cots"]
    missing = [k for k in required if k not in case]
    if missing:
        raise ValueError(f"case {case.get('id', '<unknown>')} missing keys: {missing}")
    cells = case["cells"]
    if not isinstance(cells, list) or len(cells) < 2:
        raise ValueError(f"case {case['id']} must have at least two cells")
    committed = int(case["committed"])
    if committed < 0 or committed >= len(cells):
        raise ValueError(f"case {case['id']} committed index out of range")
    if not case["spans"]:
        raise ValueError(f"case {case['id']} has no spans")
    cots = case["cots"]
    for name in ("none", "correct", "wrong", "off_task"):
        if name not in cots:
            raise ValueError(f"case {case['id']} missing cots.{name}")
    if require_localisation and not case.get("gold", {}).get("faulty_step"):
        raise ValueError(f"case {case['id']} missing gold.faulty_step")


def cell_spec(case: CotCase) -> CellSpec:
    return CellSpec(labels=tuple(case["cells"]), committed=int(case["committed"]))


def donor_span_sets(case: CotCase, all_cases: Optional[List[CotCase]] = None, n_donors: int = 3) -> List[List[Dict[str, str]]]:
    explicit = case.get("donor_span_sets")
    if explicit:
        return explicit

    donors: List[List[Dict[str, str]]] = []
    n_spans = len(case["spans"])
    for other in all_cases or []:
        if other.get("id") == case.get("id") or len(other.get("spans", [])) != n_spans:
            continue
        donors.append([{"sid": str(s.get("sid", "")), "text": str(s.get("text", ""))} for s in other["spans"]])
        if len(donors) >= n_donors:
            return donors

    filler = [
        {"sid": f"donor_{i}", "text": "This unrelated record is not evidence for the current question."}
        for i in range(n_spans)
    ]
    while len(donors) < n_donors:
        donors.append(list(filler))
    return donors


def backend_from_config(config: Dict[str, Any]) -> tuple[str, BackendConfig]:
    backend = str(config.get("backend", "openai")).strip().lower()
    model = str(config.get("model") or os.environ.get("EDFL_MODEL") or "gpt-4o-mini")
    max_concurrency = int(config.get("max_concurrency", 8))
    if backend == "dummy":
        return model, BackendConfig(kind="dummy", max_concurrency=max_concurrency)
    if backend == "openai":
        return model, BackendConfig(kind="openai", max_concurrency=max_concurrency)
    if backend == "local":
        model_id = str(config.get("local_model_id") or model)
        return model, BackendConfig(
            kind="local",
            max_concurrency=max_concurrency,
            options={
                "model_id": model_id,
                "batch": int(config.get("local_batch", 16)),
                "topk": int(config.get("local_topk", config.get("top_logprobs", 64))),
                "chat_template": bool(config.get("local_chat_template", True)),
            },
        )
    raise ValueError(f"unsupported backend in config: {backend!r}")


def result_reasons(result: Any) -> List[str]:
    return list(result.gate.reasons + result.validity.reasons())
