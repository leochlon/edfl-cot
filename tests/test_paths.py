"""The three cheap paths, on the dummy backend so no network is needed."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edfl_cot import BatteryConfig, CellSpec, abstain_fast, isolate_steps, localise_steps
from edfl_cot.backends.base import BackendConfig

CELLS = CellSpec(labels=("YES", "NO"), committed=0)
TRACE = {"spans": [{"sid": "s1", "text": "A is B."}, {"sid": "s2", "text": "B is C."}]}
COT = "A is B. B is C. So A is C."
CFG = BatteryConfig(m_serializations=4, top_logprobs=3, use_cache=False)
BCFG = BackendConfig(kind="dummy")
KW = dict(trace=TRACE, question="Is A C?", cells=CELLS, model="m",
          backend_cfg=BCFG, cfg=CFG, reasoning_text=COT)


def test_abstain_fast_refuses_without_an_anchor():
    r = abstain_fast(max_draws=32, block=8, **KW)
    assert r.decision == "abstain", r
    assert r.b_hi < CFG.p_star
    assert r.calls < 32, r.calls          # stopped before the cap
    assert r.calls % 8 == 0 and r.draws + r.discarded == r.calls


def test_localise_returns_one_rung_per_step_plus_the_empty_one():
    lad = localise_steps(**KW)
    assert [r.k for r in lad.rungs] == [0, 1, 2, 3]
    assert lad.rungs[0].step == ""
    assert lad.localised_step in (1, 2, 3)


def test_isolate_measures_each_step_against_the_empty_rung():
    lad = isolate_steps(**KW)
    assert [r.k for r in lad.rungs] == [0, 1, 2, 3]
    assert all(r.step for r in lad.rungs[1:])


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
