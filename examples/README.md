# examples

`run_cot_examples.py` reproduces the table in the top-level README: five questions,
four chains of thought each, against a live model.

    export OPENAI_API_KEY=... OPENAI_BASE_URL=...
    python3 examples/run_cot_examples.py 120

The argument is `m`, the number of serializations. At five spans the orbit is 120
and is enumerated exactly, so `b` carries no sampling error.

5,400 one-token calls at `m=120`. A decision in isolation costs 720, but the null
and diagnostic batteries do not depend on the chain, so within a question only the
first chain pays for them: 720 + 3 x 120 per question, 270 amortised per decision.
