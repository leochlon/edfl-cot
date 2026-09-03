# EDFL-CoT Benchmarks

Small benchmark scaffold for testing whether EDFL-CoT certifies answers only
when evidence and the current reasoning trace move model belief enough.

## First Slice

This first version is synthetic-only and follows the `examples/run_cot_examples.py`
frame: questions include `A) yes   B) no`, while the scored cells are `A` and
`B`.
It supports:

- `separation`: run `score_cot_budget` on `none`, `correct`, `wrong`, and
  `off_task` traces for each case. Pass `--pre-screen abstain_fast` to first
  run the cheap evidence+CoT path and skip the full donor/null battery when the
  upper bound is already below `p_star`.
- `localisation`: run `localise_steps` on the injected wrong trace.

Dummy runs check plumbing only; use a real backend for scientific conclusions.

```bash
python3 benchmarks/edfl_cot/build_data.py synthetic \
  --n 10 \
  --output benchmarks/edfl_cot/data/synthetic_v0.yaml

python3 benchmarks/edfl_cot/benchmark.py separation \
  --data benchmarks/edfl_cot/data/synthetic_v0.yaml \
  --config benchmarks/edfl_cot/configs/smoke_dummy.yaml \
  --output benchmarks/edfl_cot/results/synthetic_sep_dummy.yaml

python3 benchmarks/edfl_cot/analyze.py separation \
  --input benchmarks/edfl_cot/results/synthetic_sep_dummy.yaml

python3 benchmarks/edfl_cot/benchmark.py separation \
  --data benchmarks/edfl_cot/data/synthetic_v0.yaml \
  --config benchmarks/edfl_cot/configs/local_qwen15b_m24.yaml \
  --limit 3 \
  --m 6 \
  --cot-types correct wrong \
  --pre-screen abstain_fast \
  --fast-block 6 \
  --output benchmarks/edfl_cot/results/synthetic_sep_qwen15b_m6_limit3_fast.yaml

python3 benchmarks/edfl_cot/benchmark.py localisation \
  --data benchmarks/edfl_cot/data/synthetic_v0.yaml \
  --config benchmarks/edfl_cot/configs/smoke_dummy.yaml \
  --output benchmarks/edfl_cot/results/synthetic_loc_dummy.yaml

python3 benchmarks/edfl_cot/analyze.py localisation \
  --input benchmarks/edfl_cot/results/synthetic_loc_dummy.yaml
```
