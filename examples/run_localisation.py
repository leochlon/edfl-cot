"""Which step moved the belief. Five wrong chains, one read per rung."""
import os, sys
if not os.environ.get("OPENAI_API_KEY"):
    sys.exit("set OPENAI_API_KEY (and OPENAI_BASE_URL if you use a proxy)")
MODEL = os.environ.get("EDFL_MODEL", "gpt-4o-mini")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edfl_cot import BatteryConfig, CellSpec, localise_steps
from edfl_cot.backends.base import BackendConfig
from run_cot_examples import Qs, CELLS, mk

M = int(sys.argv[1]) if len(sys.argv) > 1 else 24
cfg = BatteryConfig(alpha=0.05, top_logprobs=20, seed=0)
print(f"model={MODEL}  m={M}  (m=1 is N+1 calls and drops to 4/5)\n")
print(f"{'question':22s} {'step':>4} {'delta':>8}  the step")
print("-" * 88)
for qn, it in Qs.items():
    lad = localise_steps(trace={"spans": mk(it["spans"])}, question=it["q"], cells=CELLS,
                         model=MODEL, backend_cfg=BackendConfig(kind="openai"), cfg=cfg,
                         reasoning_text=it["bad"], m=M)
    if lad.localised_step is None:
        print(f"{qn:22s} {'-':>4} {'-':>8}  no usable rung")
        continue
    step = lad.rungs[lad.localised_step].step
    flag = "" if abs(lad.localised_delta) >= 0.05 else "   <- below 0.05, re-run with m=24"
    print(f"{qn:22s} {lad.localised_step:>4} {lad.localised_delta:+8.4f}  {step[:44]}{flag}")
