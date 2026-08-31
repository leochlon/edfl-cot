"""EDFL CoT budgeting: answer or abstain on whether evidence moved the belief.

Four paths, in ascending cost. Take the cheapest one that answers your question.

    abstain_fast(...)    -> AbstainResult   refuse without touching a null
    localise_steps(...)  -> PrefixLadder    which step moved the belief
    isolate_steps(...)   -> PrefixLadder    each step alone; measured 1/5, kept for contrast
    score_cot_budget(...)-> CotBudgetResult the full battery; the only certificate

`core` is the accounting and is stdlib-only. `gate` is the probe battery and the
only part that performs IO.
"""
from . import core
from .core import (
    Charges, GateResult, gate, available_budget, required_budget, charge_headroom,
    anchor_from_family, conf_lower, conf_upper, kl_bernoulli,
)
from .gate import (
    AbstainResult, BatteryConfig, CellSpec, CotBudgetResult, PrefixLadder, PrefixRung,
    Readout, ValidityReport, abstain_fast, build_scoring_prompt,
    clear_verifier_cache, isolate_steps, localise_steps, score_cot_budget,
    split_steps, twin_trace_audit,
)

__version__ = "14.1.0"
__all__ = [
    "abstain_fast", "localise_steps", "isolate_steps",
    "AbstainResult", "PrefixLadder", "PrefixRung", "split_steps",
    "score_cot_budget", "CellSpec", "BatteryConfig", "CotBudgetResult",
    "ValidityReport", "Readout", "build_scoring_prompt", "twin_trace_audit",
    "clear_verifier_cache", "gate", "available_budget", "required_budget",
    "charge_headroom", "anchor_from_family", "conf_lower", "conf_upper",
    "kl_bernoulli", "Charges", "GateResult", "core", "__version__",
]
