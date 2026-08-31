"""EDFL CoT budgeting: answer or abstain on whether evidence moved the belief.

    from edfl_cot import score_cot_budget, CellSpec, BatteryConfig

    result = score_cot_budget(
        trace={"spans": spans}, question=question,
        cells=CellSpec(labels=("A", "B"), committed=0),
        model="gpt-4o-mini", donor_span_sets=nulls,
        reasoning_text=chain_of_thought, n_tokens=len(chain_of_thought.split()))
    result.answered      # bool
    result.gate.margin   # nats: A - R - C
    result.gate.reasons  # why, if it refused

`core` is the accounting and is stdlib-only. `gate` is the probe battery and the
only part that performs IO.
"""
from . import core
from .core import (
    Charges, GateResult, gate, available_budget, required_budget, charge_headroom,
    anchor_from_family, conf_lower, conf_upper, kl_bernoulli,
)
from .gate import (
    BatteryConfig, CellSpec, CotBudgetResult, Readout, ValidityReport,
    build_scoring_prompt, clear_verifier_cache, score_cot_budget, twin_trace_audit,
)

__version__ = "14.1.0"
__all__ = [
    "score_cot_budget", "CellSpec", "BatteryConfig", "CotBudgetResult",
    "ValidityReport", "Readout", "build_scoring_prompt", "twin_trace_audit",
    "clear_verifier_cache", "gate", "available_budget", "required_budget",
    "charge_headroom", "anchor_from_family", "conf_lower", "conf_upper",
    "kl_bernoulli", "Charges", "GateResult", "core", "__version__",
]
