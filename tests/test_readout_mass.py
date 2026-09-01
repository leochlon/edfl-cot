"""Regressions for mass loss and frame mismatch in the readout path."""
import math

from edfl_cot import CellSpec, gate as G
from edfl_cot.stage_ab import _token_topk_at


def _tokinfo(pairs):
    """A top-k return shaped like the backends', probabilities in, logprobs out."""
    top = [{"token": t, "logprob": math.log(p)} for t, p in pairs]
    return [{"token": pairs[0][0], "logprob": math.log(pairs[0][1]),
             "top_logprobs": top}]


def test_surface_forms_sum_rather_than_max():
    """" A" and "A" are distinct tokens; keeping the larger discards real mass."""
    seq = _tokinfo([(" A", 0.60), ("A", 0.33), (" B", 0.05)])
    r = G._readout_from_logprobs(seq, CellSpec(labels=("A", "B"), committed=0), (0,))
    assert r.ok
    assert abs(r.beta[0] - 0.93 / 0.98) < 1e-6, r.beta
    assert abs(r.d - 0.98) < 1e-6, r.d


def test_d_not_deflated_by_split_forms():
    """d is the validity condition; splitting the winner must not fail d_min."""
    seq = _tokinfo([(" A", 0.34), ("A", 0.31), ("A", 0.20), (" B", 0.13)])
    r = G._readout_from_logprobs(seq, CellSpec(labels=("A", "B"), committed=0), (0,))
    assert r.d > 0.60, r.d


def test_punctuated_labels_count():
    """"A." and "A)" are the same cell as "A" for a non YES/NO label set."""
    seq = _tokinfo([("A.", 0.5), ("A)", 0.3), ("B", 0.2)])
    r = G._readout_from_logprobs(seq, CellSpec(labels=("A", "B"), committed=0), (0,))
    assert abs(r.beta[0] - 0.8) < 1e-6, r.beta


def test_topk_keeps_forms_separate():
    tk = _token_topk_at(_tokinfo([(" A", 0.6), ("A", 0.3)]), 0)
    assert abs(sum(math.exp(v) for v in tk.topk_logprobs.values()) - 0.9) < 1e-9


class _Recorder:
    """Backend stub that records prompts and returns a fixed readable cell."""

    def __init__(self):
        self.prompts = []

    def call_text_batch(self, *, prompts, **kw):
        self.prompts += list(prompts)
        return [type("R", (), {"logprobs": _tokinfo([("A", 0.7), ("B", 0.3)])})()
                for _ in prompts]

    def call_text(self, *, prompt, **kw):
        return self.call_text_batch(prompts=[prompt], **kw)[0]

    def reset_state(self):
        return None


def test_ladder_scores_the_trace_where_the_gate_does():
    """The ladder must put the chain above ANSWER:, not inside the evidence.

    Scoring it as an untrusted span means localisation and the certificate read
    two different objects.
    """
    rec = _Recorder()
    G._prefix_ladder(
        backend=rec, backend_cfg=None, model="m",
        spans=[{"sid": "x0", "text": "Buffer exchange is stage 2."}],
        question="Q?\nA) yes   B) no",
        cells=CellSpec(labels=("A", "B"), committed=0),
        cfg=G.BatteryConfig(m_serializations=2, top_logprobs=8),
        reasoning_text="One. Two.", prompt_cache=None)
    assert rec.prompts
    p = rec.prompts[-1]
    assert "REASONING:" in p, p
    assert '<SPAN id="cot">' not in p, p


def test_answer_line_takes_the_final_commit():
    """A self-revising generation states ANSWER: more than once; the last wins."""
    raw = "ANSWER: first guess\nOn reflection that is wrong.\nANSWER: final answer"
    assert G._ANSWER_LINE.findall(raw)[-1].strip() == "final answer"


def test_ladder_delta_not_attributed_across_a_dropped_rung():
    """A skipped rung must not fold its movement into the next rung's delta."""
    rungs = [G.PrefixRung(k=0, step="", b_hat=0.20, b_lo=float("nan"), delta=float("nan"),
                          order_evidence=0.0, d_mean=1.0, usable=2),
             G.PrefixRung(k=1, step="one", b_hat=0.22, b_lo=float("nan"), delta=0.02,
                          order_evidence=0.0, d_mean=1.0, usable=2),
             G.PrefixRung(k=3, step="three", b_hat=0.92, b_lo=float("nan"),
                          delta=float("nan"), order_evidence=0.0, d_mean=1.0, usable=2)]
    moved = [r for r in rungs if r.k > 0 and not math.isnan(r.delta)]
    assert all(r.k != 3 for r in moved), "rung after a gap must not carry a delta"
