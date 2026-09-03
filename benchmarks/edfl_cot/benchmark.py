from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List

from schema import (
    backend_from_config,
    cell_spec,
    donor_span_sets,
    load_config,
    load_yaml,
    result_reasons,
    validate_case,
    write_yaml,
)

from edfl_cot import BatteryConfig, abstain_fast, clear_verifier_cache, localise_steps, score_cot_budget


DEFAULT_COT_TYPES = ("none", "correct", "wrong", "off_task")
PRE_SCREEN_CHOICES = ("none", "abstain_fast")


def battery_config(config: Dict[str, Any], args: argparse.Namespace) -> BatteryConfig:
    return BatteryConfig(
        m_serializations=int(args.m or config.get("m_serializations", 24)),
        alpha=float(config.get("alpha", 0.05)),
        p_star=float(config.get("p_star", 0.95)),
        top_logprobs=int(config.get("top_logprobs", 20)),
        seed=int(config.get("seed", 0)),
        use_cache=bool(config.get("use_cache", True)),
    )


def separation_rows(cases: List[Dict[str, Any]], config: Dict[str, Any], args: argparse.Namespace) -> Iterable[Dict[str, Any]]:
    model, backend_cfg = backend_from_config(config)
    cfg = battery_config(config, args)
    cot_types = tuple(args.cot_types or DEFAULT_COT_TYPES)
    pre_screen = str(args.pre_screen or config.get("pre_screen", "none"))
    if pre_screen not in PRE_SCREEN_CHOICES:
        raise ValueError(f"unknown pre_screen={pre_screen!r}; choose one of {PRE_SCREEN_CHOICES}")
    fast_max_draws = int(args.fast_max_draws or config.get("fast_max_draws", 200))
    fast_block = int(args.fast_block or config.get("fast_block", 5))

    for case in cases:
        validate_case(case)
        cells = cell_spec(case)
        donors = donor_span_sets(case, cases)
        clear_verifier_cache()
        for cot_type in cot_types:
            reasoning = str(case["cots"][cot_type])
            fast = None
            if pre_screen == "abstain_fast":
                fast = abstain_fast(
                    trace={"spans": case["spans"]},
                    question=str(case["question"]),
                    cells=cells,
                    model=model,
                    backend_cfg=backend_cfg,
                    cfg=cfg,
                    reasoning_text=reasoning,
                    max_draws=fast_max_draws,
                    block=fast_block,
                )
                if fast.decision == "abstain":
                    yield {
                        "experiment": "separation",
                        "id": case["id"],
                        "dataset": case["dataset"],
                        "task_type": case["task_type"],
                        "cot_type": cot_type,
                        "answered": False,
                        "b_hat": None,
                        "b_lo": fast.b_lo,
                        "b_hi": fast.b_hi,
                        "anchor": None,
                        "available": None,
                        "required": None,
                        "charges": None,
                        "margin": None,
                        "validity_ok": None,
                        "reasons": ["pre_screen_abstain"],
                        "d_evidence": None,
                        "d_ablated": None,
                        "usable_serializations": fast.draws,
                        "model": model,
                        "backend": backend_cfg.kind,
                        "m": cfg.m_serializations,
                        "seed": cfg.seed,
                        "pre_screen": pre_screen,
                        "pre_screen_decision": fast.decision,
                        "pre_screen_b_lo": fast.b_lo,
                        "pre_screen_b_hi": fast.b_hi,
                        "pre_screen_draws": fast.draws,
                        "pre_screen_calls": fast.calls,
                        "pre_screen_discarded": fast.discarded,
                        "full_score_ran": False,
                    }
                    continue
            result = score_cot_budget(
                trace={"spans": case["spans"]},
                question=str(case["question"]),
                cells=cells,
                model=model,
                backend_cfg=backend_cfg,
                cfg=cfg,
                donor_span_sets=donors,
                reasoning_text=reasoning,
                n_tokens=len(reasoning.split()),
            )
            yield {
                "experiment": "separation",
                "id": case["id"],
                "dataset": case["dataset"],
                "task_type": case["task_type"],
                "cot_type": cot_type,
                "answered": result.answered,
                "b_hat": result.b_hat[cells.committed],
                "b_lo": result.gate.b_lo,
                "b_hi": None,
                "anchor": result.gate.anchor,
                "available": result.gate.available,
                "required": result.gate.required,
                "charges": result.gate.charges,
                "margin": result.gate.margin,
                "validity_ok": result.validity.ok,
                "reasons": result_reasons(result),
                "d_evidence": result.validity.d_evidence,
                "d_ablated": result.validity.d_ablated,
                "usable_serializations": result.validity.usable_serializations,
                "model": model,
                "backend": backend_cfg.kind,
                "m": cfg.m_serializations,
                "seed": cfg.seed,
                "pre_screen": pre_screen,
                "pre_screen_decision": fast.decision if fast else None,
                "pre_screen_b_lo": fast.b_lo if fast else None,
                "pre_screen_b_hi": fast.b_hi if fast else None,
                "pre_screen_draws": fast.draws if fast else 0,
                "pre_screen_calls": fast.calls if fast else 0,
                "pre_screen_discarded": fast.discarded if fast else 0,
                "full_score_ran": True,
            }


def is_ambiguous(rungs: List[Any], delta_threshold: float, gap_threshold: float) -> bool:
    vals = sorted((abs(float(r.delta)) for r in rungs if r.k > 0 and not math.isnan(float(r.delta))), reverse=True)
    if not vals or vals[0] < delta_threshold:
        return True
    return len(vals) > 1 and vals[0] - vals[1] < gap_threshold


def localisation_rows(cases: List[Dict[str, Any]], config: Dict[str, Any], args: argparse.Namespace) -> Iterable[Dict[str, Any]]:
    model, backend_cfg = backend_from_config(config)
    cfg = battery_config(config, args)
    m = int(args.m or config.get("m_serializations", 24))

    for case in cases:
        validate_case(case, require_localisation=True)
        cells = cell_spec(case)
        ladder = localise_steps(
            trace={"spans": case["spans"]},
            question=str(case["question"]),
            cells=cells,
            model=model,
            backend_cfg=backend_cfg,
            cfg=cfg,
            reasoning_text=str(case["cots"]["wrong"]),
            m=m,
        )
        pred = ladder.localised_step
        gold = int(case["gold"]["faulty_step"])
        ambiguous = is_ambiguous(ladder.rungs, args.delta_threshold, args.gap_threshold)
        yield {
            "experiment": "localisation",
            "id": case["id"],
            "dataset": case["dataset"],
            "task_type": case["task_type"],
            "gold_faulty_step": gold,
            "predicted_step": pred,
            "hit": bool(pred == gold and not ambiguous),
            "ambiguous": ambiguous,
            "delta": ladder.localised_delta if pred is not None else None,
            "rungs": [r.__dict__ for r in ladder.rungs],
            "model": model,
            "backend": backend_cfg.kind,
            "m": m,
            "seed": cfg.seed,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EDFL-CoT benchmark experiments.")
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ("separation", "localisation"):
        p = sub.add_parser(name)
        p.add_argument("--data", type=Path, required=True)
        p.add_argument("--config", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--limit", type=int)
        p.add_argument("--m", type=int)

    sub.choices["separation"].add_argument("--cot-types", nargs="+", choices=DEFAULT_COT_TYPES)
    sub.choices["separation"].add_argument("--pre-screen", choices=PRE_SCREEN_CHOICES)
    sub.choices["separation"].add_argument("--fast-max-draws", type=int)
    sub.choices["separation"].add_argument("--fast-block", type=int)
    sub.choices["localisation"].add_argument("--delta-threshold", type=float, default=0.05)
    sub.choices["localisation"].add_argument("--gap-threshold", type=float, default=0.03)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = load_yaml(args.data, args.limit)
    config = load_config(args.config)
    if args.command == "separation":
        rows = list(separation_rows(cases, config, args))
    elif args.command == "localisation":
        rows = list(localisation_rows(cases, config, args))
    else:  # pragma: no cover
        raise ValueError(args.command)
    write_yaml(rows, args.output)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
