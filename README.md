# edfl-cot

Answer or abstain on a document-grounded question, on the basis of whether the
**evidence and the reasoning moved the model's belief** — not on how confident the
model sounds.

## Install

```bash
pip install -e .            # or: pip install -r requirements.txt
export OPENAI_API_KEY=...   # plus OPENAI_BASE_URL if you route through a proxy
```

```python
from edfl_cot import score_cot_budget, CellSpec, BatteryConfig

r = score_cot_budget(
    trace={"spans": spans},              # your retrieved passages
    question="Does the protocol perform chelation at stage 3?\nA) yes   B) no",
    cells=CellSpec(labels=("A", "B"), committed=0),
    model="gpt-4o-mini",
    donor_span_sets=nulls,               # >=2 span sets from unrelated queries
    reasoning_text=chain_of_thought,     # "" for the pre-answer gate
    n_tokens=len(chain_of_thought.split()),
    cfg=BatteryConfig(m_serializations=120, p_star=0.95))

r.answered           # bool
r.gate.margin        # nats:  M = A - R - C
r.gate.reasons       # why, when it refuses
```

## What it measures

The model's answer distribution is read over many orderings of the retrieved
passages, then again on **nulls** — span sets of the same shape from unrelated
queries. If belief is the same either way, retrieval earned nothing. The decision
is the margin in nats:

```
A = 1{b_lo >= c} · KL(Ber(b_lo) ‖ Ber(c))     available: what evidence supplied
R = KL(Ber(p*) ‖ Ber(c))                      required: the cost of committing at p*
M = A - R - C >= 0                            answer iff the margin clears
```

`c` is the anchor: the conservative worst case over the null family, measured
**before** the trace exists. A refusal names its cause — `anchor_at_or_above_target`
(the model would say this regardless of your evidence), `no_directional_movement`
(belief moved away from the committed cell), `insufficient_margin` (the evidence
did not supply enough), `charges_exceed_headroom`.

## It separates chains of thought

Five questions, four chains each, gpt-4o-mini, m=120, K=2. Reproduce with
`python3 examples/run_cot_examples.py`.

| question | none | correct | wrong | off-task |
|---|---|---|---|---|
| stage (numeric hop) | 0.5886 REFUSE | **0.9669 ANSWER** | 0.0000 REFUSE | 0.6028 REFUSE |
| buffer (entity hop) | 0.6647 REFUSE | **0.9664 ANSWER** | 0.0000 REFUSE | 0.7193 REFUSE |
| elimination | 0.9582 ANSWER | 0.9328 REFUSE | 0.1295 REFUSE | 0.9656 ANSWER |
| date arithmetic | 0.8653 REFUSE | **0.9669 ANSWER** | 0.0000 REFUSE | 0.8147 REFUSE |
| conditional rule | 0.7558 REFUSE | **0.9669 ANSWER** | 0.8967 REFUSE | 0.8903 REFUSE |

Values are `b_lo`, the one-sided lower bound on belief in the committed cell.
On the four questions the model does not already answer, the separation is exact:
correct 4/4 certify, wrong 0/4, off-task 0/4, unreasoned 0/4.

`b_lo` carries run-to-run variation of a few points on a hosted endpoint, from
logprob tail noise. All sixteen screened verdicts reproduce, the tightest at
`|M| = 0.068`. The elimination row does not: its unreasoned cell sits at
`|M| = 0.005` and flips between runs. That is the same scope condition, read off
the margin instead of the anchor.

The elimination row is the scope condition showing itself. The model already
believed it at 0.9582 before any reasoning, so the gate has nothing to certify and
anything harmless passes. **Screen items on `c < b_a(0) < p*`** — both bounds bind.

The conditional-rule row is the sharpest catch: the wrong chain did *not* collapse
belief (`b_lo = 0.8967`, the model largely accepted "96.4 percent, which is above
98") and it was still refused, on margin rather than on the hinge.

## Layout

```
edfl_cot/core.py           the accounting: identities, bounds, the gate. stdlib only.
edfl_cot/gate.py           the probe battery, prompts, nulls, validity. the only IO.
edfl_cot/_verifier_io.py   cached batch scoring, extracted from the predecessor.
edfl_cot/backends/         openai / gemini / vertex adapters.
examples/                  the table above, against a live model.
```

## Cost

`contexts x min(n!, m)` one-token calls per decision, where `contexts` is the
evidence context plus your nulls plus two diagnostics. Measured at m=120 with
three nulls: **720 calls**, or **480** with `run_diagnostic_ablations=False`.

`m` caps the orderings rather than setting them. Five spans have an orbit of
5! = 120, so m=120 enumerates it exactly and `b` is the order-marginal itself
with no sampling error; above about seven spans the orbit is sampled and `m`
sets the confidence width.

## Two things that will bite

**Cell tokens are a property of the frame, not the label.** Under a raw prompt a
base model wants `" C"`; through a chat template the same model with the same
tokenizer wants `"C"`. Get it wrong and the native candidate mass `d` collapses to
1e-4 while the renormalised belief still reads 1.0 — a confident answer about
nothing. `ValidityReport.d_evidence` is the check.

**Top-k readouts select on the outcome.** A cell leaves the returned list exactly
when its mass is low, so discarding truncated draws biases the estimate in a way
more draws cannot fix. Score every cell exactly where the endpoint allows it.
