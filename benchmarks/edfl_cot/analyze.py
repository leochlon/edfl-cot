from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List

from schema import load_yaml


def pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def avg(rows: Iterable[Dict[str, Any]], key: str) -> float:
    vals = [float(r[key]) for r in rows if r.get(key) is not None]
    return mean(vals) if vals else float("nan")


def fmt_avg(value: float) -> str:
    return f"{value:.4f}" if math.isfinite(value) else "nan"


def fmt_signed_avg(value: float) -> str:
    return f"{value:+.4f}" if math.isfinite(value) else "nan"


def print_table(headers: List[str], rows: List[List[Any]]) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    print(" | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    print(" | ".join("-" * widths[i] for i in range(len(headers))))
    for row in rows:
        print(" | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))


def summarize_separation(rows: List[Dict[str, Any]]) -> None:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["cot_type"])].append(row)

    table: List[List[Any]] = []
    for cot_type in ("none", "correct", "wrong", "off_task"):
        group = groups.get(cot_type, [])
        if not group:
            continue
        n = len(group)
        table.append([
            cot_type,
            n,
            pct(sum(bool(r["answered"]) for r in group) / n),
            pct(sum(bool(r["validity_ok"]) for r in group) / n),
            pct(sum(bool(r.get("full_score_ran", True)) for r in group) / n),
            pct(sum(str(r.get("pre_screen_decision")) == "abstain" for r in group) / n),
            fmt_avg(avg(group, "b_hat")),
            fmt_avg(avg(group, "anchor")),
            fmt_signed_avg(avg(group, "margin")),
        ])
    print_table(
        [
            "cot_type",
            "n",
            "answer_rate",
            "validity_pass",
            "full_score",
            "fast_abstain",
            "mean_b_hat",
            "mean_anchor",
            "mean_margin",
        ],
        table,
    )


def summarize_localisation(rows: List[Dict[str, Any]]) -> None:
    n = len(rows)
    if n == 0:
        print("no rows")
        return
    deltas = [abs(float(r["delta"])) for r in rows if r.get("delta") is not None]
    table = [[
        n,
        pct(sum(bool(r["hit"]) for r in rows) / n),
        pct(sum(bool(r["ambiguous"]) for r in rows) / n),
        f"{mean(deltas):.4f}" if deltas else "nan",
    ]]
    print_table(["n", "hit_rate", "ambiguous_rate", "mean_abs_delta"], table)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize EDFL-CoT benchmark results.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("separation", "localisation"):
        p = sub.add_parser(name)
        p.add_argument("--input", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_yaml(args.input)
    if args.command == "separation":
        summarize_separation(rows)
    elif args.command == "localisation":
        summarize_localisation(rows)


if __name__ == "__main__":
    main()
