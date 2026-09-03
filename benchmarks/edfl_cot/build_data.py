from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from schema import write_yaml


NAMES = ["Maya", "Nora", "Leo", "Sam", "Iris", "Omar", "Lena", "Theo", "Rina", "Ben"]
OBJECTS = ["apples", "tickets", "stickers", "marbles", "cards", "pencils", "shells", "coins", "buttons", "stamps"]


def span(sid: str, text: str, role: str = "support") -> Dict[str, str]:
    return {"sid": sid, "text": text, "role": role}


def neutral_donors(n_spans: int) -> List[List[Dict[str, str]]]:
    donor_texts = [
        ["The archive is reviewed each quarter.", "Deliveries are logged on arrival.", "Forms are stored in a cabinet."],
        ["The garden path is cleaned weekly.", "Library cards expire after one year.", "The notice board lists room numbers."],
        ["Blue folders are kept on the upper shelf.", "The morning train stops at platform two.", "Receipts are sorted by month."],
    ]
    donors: List[List[Dict[str, str]]] = []
    for j, texts in enumerate(donor_texts):
        donors.append([{"sid": f"d{j}_{i}", "text": texts[i % len(texts)]} for i in range(n_spans)])
    return donors


def factual_conclusion(name: str, obj: str, op: str, value: int, *, target: int | None = None) -> str:
    if op == "addition":
        if target is None or value == target:
            return f"So {name} has {value} {obj} now."
        return f"So {name} has {value} {obj} now, not {target}."
    if op == "subtraction":
        if target is None or value == target:
            return f"So {name} has {value} {obj} left."
        return f"So {name} has {value} {obj} left, not {target}."
    if op == "multiplication":
        if target is None or value == target:
            return f"So there are {value} {obj} in total."
        return f"So there are {value} {obj} in total, not {target}."
    if target is None or value == target:
        return f"So each group gets {value} {obj}."
    return f"So each group gets {value} {obj}, not {target}."


def arithmetic_case(i: int, op: str) -> Dict[str, Any]:
    name = NAMES[i % len(NAMES)]
    obj = OBJECTS[i % len(OBJECTS)]
    a = 3 + i
    b = 2 + (i % 5)
    if op == "addition":
        answer = a + b
        wrong = answer + 1
        verb = f"gets {b} more"
        calc = f"{a} plus {b}"
        question = f"Does {name} have {answer} {obj} now?"
        spans = [
            span("s1", f"{name} starts with {a} {obj}."),
            span("s2", f"{name} {verb}."),
            span("s3", f"The question asks for the total number of {obj} {name} has now."),
        ]
    elif op == "subtraction":
        answer = a
        total = a + b
        wrong = answer - 1
        calc = f"{total} minus {b}"
        question = f"Does {name} have {answer} {obj} left?"
        spans = [
            span("s1", f"{name} starts with {total} {obj}."),
            span("s2", f"{name} gives away {b} {obj}."),
            span("s3", f"The question asks how many {obj} {name} has left."),
        ]
    elif op == "multiplication":
        answer = a * b
        wrong = answer + b
        calc = f"{a} times {b}"
        question = f"Are there {answer} {obj} in total?"
        spans = [
            span("s1", f"There are {a} boxes."),
            span("s2", f"Each box has {b} {obj}."),
            span("s3", f"The question asks for the total number of {obj}."),
        ]
    else:
        divisor = b
        answer = a
        total = a * divisor
        wrong = answer + 1
        calc = f"{total} divided by {divisor}"
        question = f"Does each group get {answer} {obj}?"
        spans = [
            span("s1", f"There are {total} {obj}."),
            span("s2", f"The {obj} are split equally into {divisor} groups."),
            span("s3", f"The question asks how many {obj} are in each group."),
        ]

    correct = "\n".join([
        spans[0]["text"],
        spans[1]["text"],
        f"{calc} is {answer}.",
        factual_conclusion(name, obj, op, answer),
    ])
    wrong_cot = "\n".join([
        spans[0]["text"],
        spans[1]["text"],
        f"{calc} is {wrong}.",
        factual_conclusion(name, obj, op, wrong, target=answer),
    ])
    off_task = "\n".join([
        f"{obj.capitalize()} can be counted in many settings.",
        "Some records use labels and some use numbers.",
        "This does not determine the requested amount.",
    ])
    return {
        "id": f"synthetic_{op}_{i:04d}",
        "dataset": "synthetic",
        "task_type": op,
        "question": question,
        "cells": ["YES", "NO"],
        "committed": 0,
        "spans": spans,
        "donor_span_sets": neutral_donors(len(spans)),
        "cots": {"none": "", "correct": correct, "wrong": wrong_cot, "off_task": off_task},
        "gold": {"answer": "YES", "faulty_step": 3, "fault_type": op},
        "metadata": {"support_count": len(spans), "source": "template"},
    }


def conditional_case(i: int) -> Dict[str, Any]:
    batch = f"batch {chr(65 + i % 26)}"
    purity = 96 - (i % 3)
    threshold = 98
    spans = [
        span("s1", f"A second review is required when purity is below {threshold} percent."),
        span("s2", f"{batch.capitalize()} assayed at {purity} percent purity."),
        span("s3", "No waiver was recorded for this batch."),
    ]
    correct = "\n".join([
        f"The rule requires review below {threshold} percent.",
        f"{batch.capitalize()} is at {purity} percent, which is below {threshold}.",
        "No waiver was recorded.",
        "So the batch requires a second review.",
    ])
    wrong = "\n".join([
        f"The rule requires review below {threshold} percent.",
        f"{batch.capitalize()} is at {purity} percent, which is above {threshold}.",
        "No waiver was recorded.",
        "So the batch does not require a second review.",
    ])
    return {
        "id": f"synthetic_conditional_{i:04d}",
        "dataset": "synthetic",
        "task_type": "conditional_rule",
        "question": f"Does {batch} require a second review?",
        "cells": ["YES", "NO"],
        "committed": 0,
        "spans": spans,
        "donor_span_sets": neutral_donors(len(spans)),
        "cots": {
            "none": "",
            "correct": correct,
            "wrong": wrong,
            "off_task": "Laboratory forms can be filed by date.\nPurity values are often written as percentages.\nThis does not determine whether the rule applies.",
        },
        "gold": {"answer": "YES", "faulty_step": 2, "fault_type": "comparison"},
        "metadata": {"support_count": len(spans), "source": "template"},
    }


def build_synthetic(n: int) -> List[Dict[str, Any]]:
    ops = ["addition", "subtraction", "multiplication", "division", "conditional"]
    rows: List[Dict[str, Any]] = []
    for i in range(n):
        op = ops[i % len(ops)]
        rows.append(conditional_case(i) if op == "conditional" else arithmetic_case(i, op))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build EDFL-CoT benchmark data.")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("synthetic", help="Build a small controlled synthetic suite.")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "synthetic":
        rows = build_synthetic(args.n)
        write_yaml(rows, args.output)
        print(f"wrote {len(rows)} cases to {args.output}")


if __name__ == "__main__":
    main()
