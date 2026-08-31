# edfl-cot

Answer or abstain on a document-grounded question, on the basis of whether the
evidence and the reasoning **moved the model's belief** — not on how confident the
model sounds.

On five questions with four chains of thought each, the correct chain certifies
4/4 and the wrong, off-task and unreasoned chains certify 0/4.

```bash
pip install -e .
export OPENAI_API_KEY=...   # plus OPENAI_BASE_URL if you route through a proxy
```

## Three paths

Take the cheapest one that answers your question. `N` is the number of steps in
the trace.

| | calls | gives you |
|---|---|---|
| `abstain_fast` | **5–64** | a refusal, without touching a null |
| `localise_steps` | **N+1** | which step moved the belief |
| `score_cot_budget` | **720** | a certificate |

```python
from edfl_cot import abstain_fast, localise_steps, CellSpec

cells = CellSpec(labels=("A", "B"), committed=0)          # A = yes
kw = dict(trace={"spans": spans}, question=question, cells=cells,
          model="gpt-4o-mini", reasoning_text=chain_of_thought)

a = abstain_fast(**kw)
a.decision          # "abstain" | "needs_anchor" | "undetermined"
a.discarded         # readouts where a cell fell out of the softmax

lad = localise_steps(**kw)
lad.localised_step  # 1-indexed, or None
lad.localised_delta # read this, not the step
```

Escalate to `score_cot_budget` only on `needs_anchor`; it takes `donor_span_sets`
and returns `.answered`, `.gate.margin` and `.gate.reasons`. `open_answer` covers
the case where you have no options to offer: it answers the question, then builds
the YES/NO claim the three paths run on.

## How it works

The answer distribution is read over many orderings of the retrieved passages,
then again on **nulls** — span sets of the same shape from unrelated queries. If
belief is the same either way, retrieval earned nothing.

```
A = 1{b_lo >= c} · KL(Ber(b_lo) ‖ Ber(c))     available: what evidence supplied
R = KL(Ber(p*) ‖ Ber(c))                      required: the cost of committing at p*
M = A - R - C >= 0                            answer iff the margin clears
```

`c` is the anchor: the conservative worst case over the null family, measured
**before** the trace exists. A refusal names its cause — `anchor_at_or_above_target`,
`no_directional_movement`, `insufficient_margin`, `charges_exceed_headroom`.

## It separates chains of thought

Five questions, four chains each, gpt-4o-mini, m=120, K=2.
`python3 examples/run_cot_examples.py`

| question | none | correct | wrong | off-task |
|---|---|---|---|---|
| stage (numeric hop) | 0.5886 REFUSE | **0.9669 ANSWER** | 0.0000 REFUSE | 0.6028 REFUSE |
| buffer (entity hop) | 0.6647 REFUSE | **0.9664 ANSWER** | 0.0000 REFUSE | 0.7193 REFUSE |
| elimination | 0.9582 ANSWER | 0.9328 REFUSE | 0.1295 REFUSE | 0.9656 ANSWER |
| date arithmetic | 0.8653 REFUSE | **0.9669 ANSWER** | 0.0000 REFUSE | 0.8147 REFUSE |
| conditional rule | 0.7558 REFUSE | **0.9669 ANSWER** | 0.8967 REFUSE | 0.8903 REFUSE |

Values are `b_lo`, the one-sided lower bound on belief in the committed cell. All
sixteen screened verdicts reproduce, the tightest at `|M| = 0.068`.

The elimination row is the scope condition showing itself. The model already
believed it at 0.9582, so the gate has nothing to certify and anything harmless
passes. **Screen items on `c < b_a(0) < p*`** — both bounds bind.

The conditional-rule row is the sharpest catch: the wrong chain did *not* collapse
belief (`b_lo = 0.8967`) and was still refused, on margin rather than on the hinge.

## Which step

`localise_steps` puts the trace in as one more evidence span and reads rung `k`
with steps `1..k` present. Nested rather than a power set, so `delta = b(k) -
b(k-1)` is attributable to step `k`. No anchor, no orbit, no `p*`.

`python3 examples/run_localisation.py`, five wrong chains at `m=24`:

| question | step | delta | the step |
|---|---|---|---|
| stage | 2 | -0.7595 | *...and the step after stage 2 is stage 4* |
| buffer | 2 | -0.5887 | *The stage-3 buffer is potassium acetate* |
| elimination | 1 | -0.8927 | *...but stage 5 is an exception to that rule* |
| date | 2 | -0.9773 | *2024 plus three is 2028* |
| conditional | 2 or 3 | -0.2338 | *96.4 percent, which is above 98* |

The first four name the step that introduced the error and reproduce across runs.
`conditional` does not: its rungs 2 and 3 swap the argmax between runs, because
the model never accepts that chain's false comparison, so no single rung carries
the movement.

**Read the delta, not the argmax.** A trace where no rung clears about 0.05, or
where two rungs are within noise, has not been localised. `isolate_steps` reads
each step alone instead, and lands on the concluding step rather than the faulty
one: an error is a step wrong *in context*.

## Local models

`kind="local"` reads the scoring position from `transformers` weights, so the
softmax is complete and no cell can fall off a top-k list. On the `stage` item
Qwen2.5-1.5B-Instruct gave 120/120 usable serializations and `d_evidence` 1.0000,
where a hosted run of a comparable item discarded 59 of 64 draws.

```python
BackendConfig(kind="local", options={"model_id": "Qwen/Qwen2.5-1.5B-Instruct"})
```

`options` also takes `device`, `dtype`, `batch`, `topk` and `chat_template`. The
last one decides the frame, and the frame decides the cell token: through a
template a model wants `"A"`, on a raw prompt it wants `" A"`.

## Cost

`contexts x min(n!, m)` one-token calls per certificate — evidence plus nulls plus
two diagnostics, so **720** at m=120 with three nulls, or **480** with
`run_diagnostic_ablations=False`. That floor is the bound, not the probe:
answering reduces to `b_lo >= p*`, and `conf_lower` needs about 80 draws to reach
0.95 even when every draw is exactly 1.0.

`abstain_fast` pays none of it. It stops on `core.sequential_decision`, a
confidence sequence, so optional stopping needs no correction; and once `b`'s
upper bound is below `p*` no anchor can rescue it. Below five draws the bound
cannot exclude 0.95 whatever it reads, which sets the floor.

Watch `discarded`. A cell absent from the returned softmax is an instrument
failure, not an interval, so those draws are dropped — one example reached its
verdict on 5 usable draws out of 64. Dropping them biases `b` upward, so an
abstention despite discards is conservative and a `needs_anchor` is not.

## Layout

```
edfl_cot/core.py           the accounting: identities, bounds, the gate. stdlib only.
edfl_cot/gate.py           the three paths, the probe battery, prompts, nulls. the only IO.
edfl_cot/backends/         openai / gemini / vertex adapters.
examples/                  the tables above, against a live model.
```

MIT © Hassana Labs
