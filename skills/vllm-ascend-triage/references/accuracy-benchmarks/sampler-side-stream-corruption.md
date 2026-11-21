# `temperature>0` corruption on a healthy serve: the sampler's side-stream `q`

## Symptom

Sampled requests intermittently carry corrupted tokens — token id 0 (`<pad>`), one
rare id, runs of repeats, garbage spliced into otherwise exact text — while
**greedy output is byte-identical across runs**, no request errors, no device
fault, tokens/s normal. The rate rises with concurrency (reported ~1/25k tokens at
8 → ~1/5k at 64), on a graph-mode serve with async scheduling on. Recorded #18754
(vllm-ascend v0.20.2rc1 `367b8e6` / vllm v0.20.2, 910B2C, DP4×TP4) and, as the same
code site, #13673 (v0.18.0).

Not this page: greedy (`temperature: 0`) also differing — that is repeat-run
nondeterminism (`repeated-run-nondeterminism.md`); the text differing with
concurrency under `VLLM_BATCH_INVARIANT=1` — `batch-invariant-coverage.md`; a
fluent but wrong answer at normal tokens/s — `moe-gating-topk-precision.md`.

## G1: this shape is answerable from the paste plus one audit grep

The report usually arrives with its own ablation matrix (graph+async corrupt,
`--no-async-scheduling` clean, `--enforce-eager` clean, `enable_async_exponential`
clean) and names the function. That is enough for the verdict: say **which
mechanism the code supports, which one it refutes, and the one-line probe**, then
spend later turns only on corrections. Do not re-derive the canary and do not
re-walk the sampler class.

## The code at the pin (v0.20.2rc1 `367b8e6`)

`vllm_ascend/sample/sampler.py:18-41` `random_sample()`:

```python
with npu_stream_switch(global_stream()):      # :31  q and the fill land on the side stream
    q = torch.empty_like(probs)
    if len(generators) != probs.shape[0]:
        q.exponential_()                      # :34
    if generators:
        for i, generator in generators.items():
            q[i].exponential_(generator=generator)
torch.npu.current_stream().wait_stream(global_stream())   # :40  stream-order sync only
return probs.div_(q).argmax(dim=-1).view(-1)              # :41  consumed on the current stream
```

`npu_stream_switch(target)` is `torch.npu.stream(target)` (`utils.py:1074-1082`), a
real stream switch restored on exit — so the switch is not decorative, and `q`'s
fill and its read genuinely land on two different streams.

## Two mechanisms; the ordering one is refuted, the lifetime one is not

1. **Producer→consumer ordering ("`div_` reads `q` before `exponential_()` fills
   it") — refuted as stated.** `wait_stream(s)` makes every later operation on
   the calling stream wait for the work already submitted to `s`, so the fill is
   ordered before the `div_`. Reporters reach for this because the sync is the
   only visible line; correct it explicitly rather than ranking it.
2. **Allocator lifetime of `q` — the mechanism the corpus actually describes.**
   `q` is allocated on `global_stream()`, then consumed *and dropped* on the
   current stream: no `q.record_stream(torch.npu.current_stream())`, and no
   reverse wait making the allocating stream outlive the reader. The caching
   allocator can therefore hand the block to the next step's
   `torch.empty_like`/`exponential_()` (also on `global_stream()`) while the
   current stream's `div_` is still reading it. Precedent #13673 states exactly
   this (v0.18.0, same function, "该实现存在于 vLLM Ascend v0.18.0 等多个分支").
   `rg -n record_stream vllm_ascend/` returns **nothing** at the pin, so it is
   net-new code, not house style — worth saying when it lands in a patch proposal.

Why the lifetime reading is the better rank, and where it is weak:

- `exponential_()` never produces exact zeros, so the reported "`q` all-zero → inf
  → argmax 0" needs `q` to hold **non-exponential** bytes (a recycled block or
  fresh uninitialized memory). That is the lifetime signature, not an ordering one.
- It also predicts "graph + async only" more weakly than the reporter's story
  does: the same hazard exists in eager, so the reporter's clean eager row is the
  row to re-run at the same concurrency before calling the mechanism closed. Say
  which part is source-derived and which is still the reporter's control.
- Sampling itself is **not** inside the captured region at this pin (only the
  forward is), so no "the sampler was captured" explanation is available; graph
  mode enters only through overlap/timing. That bounds the *mechanism* for a
  corruption report; it does not retire the graph ablation for a **device fault**
  in a sampling kernel — a sporadic `507035` in `ApplyTopKTopPWithSorted` cleared
  under `--enforce-eager` at v0.10.1rc1 (reporter-confirmed #2763, ~5 vs ~40 tok/s;
  `../custom-ops-runtime-faults/aclnn-error-dump-reading.md` §sampler mask op).

## The `enable_async_exponential` row localizes the function, not the sync primitive

- Default **False** (`ascend_config.py:230-231`), and forced off again under
  `VLLM_BATCH_INVARIANT` — so the documented workaround is unavailable in a
  batch-invariant deployment (say so before recommending it).
- In that path `q` is generated **before the forward pass** on the side stream
  with a recorded event (`do_async_exponential`, `:79-92`) and `forward_native`
  host-syncs the event (`self.async_event.synchronize()`, `:160-162`) before
  `probs.div_(self.q)`.
- Consequence for the fix proposal: that row removes `random_sample()` from the
  request path entirely, so the matrix indicts **that function's
  allocation/consumption discipline**, not "wait_stream instead of an event". And
  the zero-cost claim belongs to the *pre-generated* form only: an
  `event.synchronize()` inserted inside `random_sample()` after the fill has
  nothing to overlap with and costs a per-step host round-trip. Rank the
  lifetime fix (`record_stream`, no sync) above it.

## Cheapest isolation at the pin

1. **`VLLM_BATCH_INVARIANT=1` on the same canary** — `forward_native` returns
   `super().forward_native(...)` (`:135`), i.e. upstream's `random_sample` with no
   side stream. Clean ⇒ the Ascend side-stream path owns the corruption; still
   corrupt ⇒ not the sampler. It is a probe, not a mitigation (per-op envelope and
   throughput caveats: `batch-invariant-coverage.md`).
2. `--additional-config '{"enable_async_exponential": true}'` — keeps graphs and
   async scheduling.
3. `--no-async-scheduling`, then `--enforce-eager` (each must be visible in the
   paste with its own log line; see rule 15's ablation trap).
4. As a local patch only, `q.record_stream(torch.npu.current_stream())` before the
   `div_` — the one-line test of the lifetime mechanism at zero sync cost, and the
   experiment that separates it from the ordering story.

## Coverage at the pin — "no known issue" is not "validated"

`tests/e2e/singlecard/test_sampler.py` has three cases: the default path appears
only at `temperature=0.0` (`test_qwen3_topk`), and the `temperature=1.0` case
(`test_qwen3_exponential_overlap`) sets `enable_async_exponential: True`. **No CI
case runs `random_sample()` with temperature>0**, so the exact path under
discussion is untested at this ref — citable as coverage, never as the cause.

## Bookkeeping corrections worth carrying

- History token choice: `-b 'random_sample|enable_async_exp'` returns 12 ids, of
  which one is a precedent (#13673); the rest match tracebacks that merely pass
  through `self.topk_topp_sampler(...)`. Read the one whose **title** names
  sampling, then stop.
- A report citing "PR #4501 (`enable_async_exponential`)" is citing the wrong PR:
  the corpus RFC #5304's flag table line reads `enable_async_exponential` → PR
  **#6089**. PRs are unreadable in this snapshot, so quote the RFC line rather
  than the PR number.

## Version scope

Verified by reading the tree at vllm-ascend v0.20.2rc1 (`367b8e6`) + vllm v0.20.2
(`bc150f5`); the mechanism text is #13673's own body at v0.18.0, so the code has
survived at least one release boundary. Before quoting on another ref, re-grep
`random_sample|record_stream|enable_async_exponential` in
`vllm_ascend/sample/sampler.py` — a ref that already calls `record_stream`, or
that pre-generates `q` by default, retires the whole page.
