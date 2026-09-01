"""Agentic repair demo: localise a bad reasoning step, call a tool, re-score.

This keeps the repair loop general: a model chooses the tool call for the
localised step, code executes the tool, the tool observation is inserted into
the trace, and the model continues from that observation.

  wrong CoT -> localise_steps -> tool-call agent -> tool observation
            -> continue trace -> score again
"""
import ast
import os
import re
import sys
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

MODEL = os.environ.get("EDFL_MODEL", "gpt-4o-mini")
EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(EXAMPLES_DIR)
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, EXAMPLES_DIR)

from edfl_cot import BatteryConfig, score_cot_budget, localise_steps, split_steps
from edfl_cot.backends.base import BackendConfig, make_backend
from run_cot_examples import CELLS, DONORS, Qs, mk


@dataclass
class ToolCall:
    tool: str
    input: str
    repair_step: str
    reason: str


@dataclass
class ToolObservation:
    tool: str
    input: str
    output: str
    text: str


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def evidence_block(spans: List[Dict[str, str]]) -> str:
    return "\n".join(f"- {span.get('sid', '?')}: {span.get('text', '')}" for span in spans)


def parse_json_object(text: str) -> Dict[str, Any]:
    match = _JSON_OBJECT_RE.search(text.strip())
    if not match:
        raise ValueError(f"agent did not return a JSON object: {text!r}")
    return json.loads(match.group(0))


def plan_tool_call_with_agent(
    *,
    question: str,
    spans: List[Dict[str, str]],
    wrong_cot: str,
    localised_step: str,
    model: str,
    backend_cfg: BackendConfig,
) -> ToolCall:
    """Ask one agent call to choose a tool call for the localised step."""
    prompt = f"""Question:
{question.strip()}

Evidence:
{evidence_block(spans)}

Original reasoning trace:
{wrong_cot.strip()}

Localised belief-moving step:
{localised_step.strip()}

Available tools:
- calculator: evaluate arithmetic expressions. Input must be a concise expression like "2024 + 3".

Return exactly one JSON object with these keys:
{{
  "tool": "calculator",
  "input": "...",
  "repair_step": "a replacement for the localised step that says what needs to be verified with the tool, without inventing the tool result",
  "reason": "why this tool is needed"
}}

Do not solve the whole task. Plan only the tool call needed for the localised step.
"""
    res = make_backend(backend_cfg).call_text(
        prompt=prompt,
        model=model,
        instructions="You produce strict JSON tool calls for repairing one localised reasoning step.",
        temperature=0.0,
        max_output_tokens=256,
        include_logprobs=False,
        top_logprobs=0,
    )
    data = parse_json_object(str(res.text))
    tool = str(data.get("tool", "")).strip()
    tool_input = str(data.get("input", "")).strip()
    repair_step = str(data.get("repair_step", "")).strip()
    reason = str(data.get("reason", "")).strip()
    if tool != "calculator":
        raise ValueError(f"unsupported tool from agent: {tool!r}")
    if not tool_input:
        raise ValueError("agent returned an empty tool input")
    if not repair_step:
        raise ValueError("agent returned an empty repair_step")
    return ToolCall(tool=tool, input=tool_input, repair_step=repair_step, reason=reason)


def safe_calculate(expression: str) -> str:
    """Evaluate a small arithmetic expression without exposing Python eval."""
    tree = ast.parse(expression, mode="eval")
    allowed_binops = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.FloorDiv: lambda a, b: a // b,
        ast.Mod: lambda a, b: a % b,
        ast.Pow: lambda a, b: a ** b,
    }
    allowed_unary = {
        ast.UAdd: lambda a: a,
        ast.USub: lambda a: -a,
    }

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in allowed_binops:
            return allowed_binops[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in allowed_unary:
            return allowed_unary[type(node.op)](visit(node.operand))
        raise ValueError(f"unsupported calculator expression: {expression!r}")

    value = visit(tree)
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value)


def run_tool(tool_call: ToolCall) -> ToolObservation:
    if tool_call.tool != "calculator":
        raise ValueError(f"unsupported tool: {tool_call.tool!r}")
    output = safe_calculate(tool_call.input)
    text = f"Calculator result: {tool_call.input} = {output}."
    return ToolObservation(tool=tool_call.tool, input=tool_call.input, output=output, text=text)


def build_repair_prefix(
    *,
    wrong_cot: str,
    localised_step_index: int,
    tool_call: ToolCall,
    observation: ToolObservation,
) -> str:
    steps = split_steps(wrong_cot)
    before = steps[: max(0, localised_step_index - 1)]
    repair_step = (
        f"{tool_call.repair_step}\n"
        f"   TOOL CALL: {tool_call.tool}({tool_call.input!r})\n"
        f"   TOOL OBSERVATION: {observation.text}"
    )
    repaired_steps = before + [repair_step]
    return "\n".join(f"{i}. {step}" for i, step in enumerate(repaired_steps, start=1))


def continue_reasoning_with_agent(
    *,
    question: str,
    spans: List[Dict[str, str]],
    repair_prefix: str,
    model: str,
    backend_cfg: BackendConfig,
) -> str:
    prompt = f"""Question:
{question.strip()}

Evidence:
{evidence_block(spans)}

Reasoning prefix:
{repair_prefix.strip()}

Continue the reasoning trace from the given prefix. Output only the continuation.
"""
    res = make_backend(backend_cfg).call_text(
        prompt=prompt,
        model=model,
        instructions="Continue a reasoning trace after a tool observation. Output only the continuation.",
        temperature=0.0,
        max_output_tokens=256,
        include_logprobs=False,
        top_logprobs=0,
    )
    return str(res.text or "").strip()


def verdict(result):
    reasons = result.gate.reasons + result.validity.reasons()
    if result.answered:
        return "ANSWER"
    return "REFUSE " + (",".join(reasons) or "-")


def print_ladder(label: str, ladder):
    print(label)
    if ladder.localised_step is None:
        print("  no usable rung\n")
        return
    print(f"  step:    {ladder.localised_step}")
    print(f"  delta:   {ladder.localised_delta:+.4f}")
    print(f"  text:    {ladder.rungs[ladder.localised_step].step}")
    print("  rungs:")
    for rung in ladder.rungs:
        if rung.k == 0:
            continue
        print(f"    {rung.k}: delta={rung.delta:+.4f} b_hat={rung.b_hat:.4f} step={rung.step[:72]}")
    print()


def parse_args(argv):
    args = list(argv)
    out = {
        "m_score": 24,
        "m_localise": 24,
        "aoai_pool": None,
        "aoai_id": "halls-aoai-eus2-0",
        "aoai_deployment": None,
        "max_concurrency": 8,
        "local": os.environ.get("EDFL_LOCAL_MODEL"),
    }
    positional = []
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--aoai-pool":
            out["aoai_pool"] = args[i + 1]
            i += 2
        elif arg == "--aoai-id":
            out["aoai_id"] = args[i + 1]
            i += 2
        elif arg == "--aoai-deployment":
            out["aoai_deployment"] = args[i + 1]
            i += 2
        elif arg == "--max-concurrency":
            out["max_concurrency"] = int(args[i + 1])
            i += 2
        else:
            positional.append(arg)
            i += 1
    if positional:
        out["m_score"] = int(positional[0])
    if len(positional) > 1:
        out["m_localise"] = int(positional[1])
    return out


def backend_from_args(args):
    if args.get("local"):
        return args["local"], BackendConfig(kind="local", options={
            "model_id": args["local"], "batch": 4, "topk": 64,
            "chat_template": True})
    if not args["aoai_pool"]:
        if not os.environ.get("OPENAI_API_KEY"):
            sys.exit("set OPENAI_API_KEY (and OPENAI_BASE_URL if you use a proxy)")
        return MODEL, BackendConfig(kind="openai", max_concurrency=args["max_concurrency"])

    with open(args["aoai_pool"], "r", encoding="utf-8") as f:
        pool = json.load(f)
    selected = next((item for item in pool if item.get("id") == args["aoai_id"]), None)
    if selected is None:
        known = ", ".join(item.get("id", "?") for item in pool)
        sys.exit(f"AOAI id not found: {args['aoai_id']} (known: {known})")
    return selected.get("deployment") if args["aoai_deployment"] is None else args["aoai_deployment"], BackendConfig(
        kind="azure",
        base_url=selected["endpoint"],
        api_key=selected["apiKey"],
        max_concurrency=args["max_concurrency"],
        options={"api_version": selected["apiVersion"]},
    )


def main():
    args = parse_args(sys.argv[1:])
    m_score = args["m_score"]
    m_localise = args["m_localise"]
    cfg = BatteryConfig(
        m_serializations=m_score,
        p_star=0.95,
        alpha=0.05,
        top_logprobs=20,
        seed=0,
    )
    model, backend_cfg = backend_from_args(args)

    item = Qs["date arithmetic"]
    trace = {"spans": mk(item["spans"])}
    wrong_cot = item["bad"]

    print(f"model={model}  backend={backend_cfg.kind}  score_m={m_score}  localise_m={m_localise}\n")

    before = score_cot_budget(
        trace=trace,
        question=item["q"],
        cells=CELLS,
        model=model,
        backend_cfg=backend_cfg,
        cfg=cfg,
        donor_span_sets=DONORS,
        reasoning_text=wrong_cot,
        n_tokens=len(wrong_cot.split()),
    )
    print("BEFORE")
    print(f"  verdict: {verdict(before)}")
    print(f"  b_hat:   {before.b_hat[CELLS.committed]:.4f}")
    print(f"  b_lo:    {before.gate.b_lo:.4f}")
    print(f"  anchor:  {before.gate.anchor:.4f}")
    print(f"  margin:  {before.gate.margin:+.4f}\n")

    ladder = localise_steps(
        trace=trace,
        question=item["q"],
        cells=CELLS,
        model=model,
        backend_cfg=backend_cfg,
        cfg=cfg,
        reasoning_text=wrong_cot,
        m=m_localise,
    )
    if ladder.localised_step is None:
        sys.exit("localise_steps did not return a usable rung")

    bad_step = ladder.rungs[ladder.localised_step].step
    print_ladder("LOCALISE BEFORE", ladder)

    tool_call = plan_tool_call_with_agent(
        question=item["q"],
        spans=trace["spans"],
        wrong_cot=wrong_cot,
        localised_step=bad_step,
        model=model,
        backend_cfg=backend_cfg,
    )
    print("TOOL CALL AGENT")
    print(f"  tool:    {tool_call.tool}")
    print(f"  input:   {tool_call.input}")
    print(f"  reason:  {tool_call.reason}")
    print(f"  repair:  {tool_call.repair_step}\n")

    observation = run_tool(tool_call)
    tool_span = {"sid": f"tool:{observation.tool}", "text": observation.text}
    print("TOOL")
    print(f"  observation: {observation.text}\n")

    repaired_trace = {"spans": trace["spans"] + [tool_span]}
    repaired_donors = [
        list(donor) + [{"sid": "tool:donor", "text": "Calculator result: 100 + 5 = 105."}]
        for donor in DONORS
    ]
    repair_prefix = build_repair_prefix(
        wrong_cot=wrong_cot,
        localised_step_index=ladder.localised_step,
        tool_call=tool_call,
        observation=observation,
    )
    print("REPAIR PREFIX")
    print(repair_prefix)
    print()

    continuation = continue_reasoning_with_agent(
        question=item["q"],
        spans=repaired_trace["spans"],
        repair_prefix=repair_prefix,
        model=model,
        backend_cfg=backend_cfg,
    )
    print("CONTINUATION")
    print(continuation)
    print()

    repaired_cot = f"{repair_prefix}\n{continuation}".strip()

    after = score_cot_budget(
        trace=repaired_trace,
        question=item["q"],
        cells=CELLS,
        model=model,
        backend_cfg=backend_cfg,
        cfg=cfg,
        donor_span_sets=repaired_donors,
        reasoning_text=repaired_cot,
        n_tokens=len(repaired_cot.split()),
    )
    print("AFTER")
    print(f"  verdict: {verdict(after)}")
    print(f"  b_hat:   {after.b_hat[CELLS.committed]:.4f}")
    print(f"  b_lo:    {after.gate.b_lo:.4f}")
    print(f"  anchor:  {after.gate.anchor:.4f}")
    print(f"  margin:  {after.gate.margin:+.4f}")
    print()

    repaired_ladder = localise_steps(
        trace=repaired_trace,
        question=item["q"],
        cells=CELLS,
        model=model,
        backend_cfg=backend_cfg,
        cfg=cfg,
        reasoning_text=repaired_cot,
        m=m_localise,
    )
    print_ladder("RE-LOCALISE AFTER", repaired_ladder)


if __name__ == "__main__":
    main()
