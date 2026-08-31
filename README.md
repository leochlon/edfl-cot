# edfl-cot

Answer or abstain on a document-grounded question, on the basis of whether the
**evidence and the reasoning moved the model's belief** — not on how confident the
model sounds.

## Install

```bash
pip install -e .            # or: pip install -r requirements.txt
export OPENAI_API_KEY=...   # plus OPENAI_BASE_URL if you route through a proxy
```

Three paths, ascending in cost. Take the cheapest one that answers your question.

| path | returns | calls | what it gives you |
|---|---|---|---|
| `abstain_fast` | `AbstainResult` | **8-64** | refuse, without touching a null |
| `localise_steps` | `PrefixLadder` | **N+1** | which step moved the belief |
| `score_cot_budget` | `CotBudgetResult` | **720** | the only certificate |

`N` is the number of reasoning steps. The certificate cannot be made cheap: answering
reduces to `b_lo >= p*`, and `conf_lower` needs about 80 draws to reach 0.95 even
when every draw is exactly 1.0, so the floor is the bound and not the probe. Run
`abstain_fast` first and escalate only on `needs_anchor`.

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

## Which step

`localise_steps` puts the trace in as one more evidence span and reads rung `k`
with steps `1..k` present, the placeholder at `k=0`. Nested rather than a power
set, so `delta = b(k) - b(k-1)` is attributable to step `k` alone. No anchor, no
orbit, no `p*`.

`python3 examples/run_localisation.py`, five wrong chains at `m=24`:

| question | step found | delta | the step |
|---|---|---|---|
| stage | 2 | -0.7595 | *...and the step after stage 2 is stage 4* |
| buffer | 2 | -0.5887 | *The stage-3 buffer is potassium acetate* |
| elimination | 1 | -0.8927 | *...but stage 5 is an exception to that rule* |
| date | 2 | -0.9773 | *2024 plus three is 2028* |
| conditional | 2 or 3 | -0.2338 | *96.4 percent, which is above 98* |

The first four name the step that introduced the error and reproduce across
runs. `conditional` does not: its rungs 2 and 3 sit at -0.2338 and -0.0920 in one
run and swap the argmax in the next, because the model never accepts that chain's
false comparison, so no single rung carries the movement. At `m=1` the same
instability spreads — `buffer` misses in one run, `conditional` in another.

**Read the delta, not the argmax.** A trace where no rung clears about 0.05, or
where two rungs are within noise of each other, has not been localised; the
example prints that flag.

`isolate_steps` reads each step alone instead, and lands on the concluding step
rather than the faulty one: an error is a step wrong *in context*.

## Cost

`contexts x min(n!, m)` one-token calls per decision, where `contexts` is the
evidence context plus your nulls plus two diagnostics. Measured at m=120 with
three nulls: **720 calls**, or **480** with `run_diagnostic_ablations=False`.

`abstain_fast` pays none of it. It stops on `core.sequential_decision`, a
confidence sequence, so optional stopping needs no correction; and once the
*upper* bound on `b` is below `p*` no anchor can rescue it, so the nulls never
run. Four of the five example chains refuse that way in 8 to 64 draws, and no
correct chain trips it within 32. The fifth returns `undetermined`, which is the
case where the draws are doing real work.

Watch `discarded`. A cell absent from the returned softmax is an instrument
failure, not an interval, so those draws are dropped -- `date` reached its
verdict on 5 usable draws out of 64. Dropping them biases `b` upward, so an
abstention despite discards is conservative and a `needs_anchor` is not.

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
