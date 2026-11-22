# A throughput cliff at a concurrency number: graph dispatch abandoned silently

## When this page applies

`Running`/throughput drops **in a step** as the number of concurrently running
sequences crosses one value, with no error, no warning the reporter noticed and
no OOM: recorded #14440 (vllm-ascend v0.23.0rc1 `f4a08bd`, vllm v0.23.0
`0fc695f`, 8×910B2, DeepSeek-V4-Flash-w8a8-mtp, tp=8 + EP + MTP
`num_speculative_tokens: 1` + FlashComm1, `--max-num-seqs 264`,
`cudagraph_mode: FULL_DECODE_ONLY`): 256 running requests → 3048 tok/s, 258 →
1182, 264 → 1100; engine `Waiting` 0, KV 81-84 %, AICore 32 % → 8 %, power
172 W → 131 W/card, EngineCore busy on ~3 host cores.

Two numbers come from the serve line, and the reporter's own cliff index is the
test of the whole page:

```
decode_query_len dq      = 1 + num_speculative_tokens            (2 with MTP-1)
capture ceiling   cap    = grid_floor( min(max_num_seqs*dq, 512, max_num_batched_tokens) )
cliff, per engine        = cap // dq
grid_floor(x)            = largest of {1,2,4} <= x         for x < 8
                         = 8*floor(x/8)                    for 8 <= x < 256
                         = 256 + 16*floor((x-256)/16)      for x >= 256
```

The threshold is per **engine**, and it is a *token* budget quantised to the
capture grid — so its shape follows from `max_num_seqs × dq`, not from a per-rank
memory budget:

| serve line | `mns × dq` | grid floor | cliff / engine | reporter's measurement |
|---|---|---|---|---|
| tp=8, `--max-num-seqs 264`, dq=2 | 528 (> the 512 cap) | 512 | **256** | 256 → 258 ✓ |
| dp=2/tp=4, `--max-num-seqs 224`, dq=2 | 448 | 448 | **224** | healthy at ≤224/engine, aggregate ceiling 256 → 448 ✓ |

Read the table before writing "a fixed-size structure indexed by running-sequence
count": that is the right shape and the wrong owner. Where the flag's product is
below the 512 cap and lands on the grid, the cliff **is** `--max-num-seqs`
itself; where it exceeds 512, the boundary is the hard `512 // dq` — which is why
tp=8 turned over at the power of two (264 × 2 = 528 → 512) while tp=4 turned over
at its own flag (224). What makes the position move between runs is the flag and
the grid, so recompute it per run instead of carrying `256` between them. The
sequence-parallel path can also rewrite the list (`platform.py:555-567`), and on
950 (`MAX_CAPTURE_SIZES_FOR_950`) it is pruned to 4 sizes.

## Do this first: the arithmetic (one call)

```bash
bash <skill_root>/scripts/seq-cliff-capture-budget.sh --max-num-seqs 264 --num-spec 1 \
    --max-num-batched-tokens 8192 --running 256,258,264 --observed-cliff 256
```

It prints the platform default, the grid floor, the truncation warning the run
should carry, the sequence count where dispatch flips, and — with
`--observed-cliff` — whether the reporter's own step index matches the
arithmetic (mismatch also lists the per-engine `--max-num-seqs` values that would
produce the observed index, and says when nothing does). For #14440 the tp=8 pass
prints `CLIFF: 256 sequences healthy, 257 silently eager` and the tp=4 pass
(`--max-num-seqs 224 --dp 2`) prints `224`; both match the reporter. When the
observed index matches, the graph ceiling is the owner and the rest of this page
is confirmation plus the ablations that do *not* count; when it does not, go to
*If the A/B does not move the cliff* instead of quoting this mechanism.

## The mechanism, pinned

| Site | Fact |
|---|---|
| `vllm_ascend/platform.py:207-241` (v0.23.0rc1) | `_get_default_max_cudagraph_capture_size` returns `min(max_num_seqs * decode_query_len, 512)`, `decode_query_len = 1 + num_speculative_tokens`; the docstring calls the removed trailing `* 2` "the only intentional difference from upstream" |
| `vllm_ascend/platform.py:259-261` | that value is injected in `apply_config_platform_defaults`, i.e. **before** upstream derives the list, so it is the value in force |
| `vllm/config/vllm.py:1618-1790` (v0.23.0) | `_set_cudagraph_sizes`: sizes `[1,2,4] + range(8, min(max+1,256), 8) + range(256, max+1, 16)`; the platform-injected max is clipped by `max_num_batched_tokens`; `max_num_batched_tokens` is appended when it fits; a user-set max that is not the list's max is **truncated** (`:1755-1759`, `Truncating max_cudagraph_capture_size to %d`). This block is the `grid_floor` above |
| `vllm/v1/cudagraph_dispatcher.py:276-285` | `num_tokens > max_size` → `return CUDAGraphMode.NONE` — no log, no warning. This is the "silent" half |
| `vllm_ascend/utils.py:1173-1204` | `_compute_potential_max_tokens` = `max(max_cudagraph_capture_size, min(max_num_batched_tokens, max_num_seqs*dq))` and emits **one** startup `warning_once`: "The max_cudagraph_capture_size (%d) is smaller than the potential max tokens required for decode (%d) … see the issue #8240". At the reported config it prints `512` vs `528` |
| `vllm_ascend/utils.py:647-669` | a second truncation, `original_size=…, truncated_size=…`, for the sizes it resamples (950 prune, SP rewrite) |
| `vllm_ascend/platform.py:592-594` | `cudagraph_mode == NONE` is what force-disables `enable_npugraph_ex` — the dependency runs one way only: the compile-backend switch does **not** switch the graph off |
| `docs/source/user_guide/release_notes.md:844` | the project's own workaround sentence for the MTP×EP graph-capture family: `--compilation-config '{"max_cudagraph_capture_size": N}'` with `N = max_concurrency * (1 + num_speculative_tokens)` |
| `docs/source/tutorials/models/DeepSeek-V4-Flash.md:166,215` | the validated commands at this ref use `--max-num-seqs 32` / `64`; no in-tree baseline ever runs 256+ sequences, so "your command differs from the tutorial" is not the answer here |

Three consequences worth stating in the answer:

- Requested concurrency is not what the dispatcher sees. With MTP, a 264-request
  decode step is 528 tokens; the graph ceiling is a **token** budget floored to
  the capture grid, so the cliff lands at `cap // dq` requests.
- Under the ceiling the step is a graph replay; over it every step is eager
  Python + per-op dispatch, which is exactly the observed shape (NPUs idle,
  host cores busy, per-request rate collapsing while `Waiting` stays 0).
- The step cost is a **decode-step** cost: the recorded reporter's own table has
  TTFT *falling* across the cliff (3.2 s avg at 256 → 2.0 s at 258, p99 5.1 →
  2.7) while per-request wall time goes 65 s → 185 s → 202 s for the same 1024
  output tokens, with MTP acceptance length 1.6-1.8 on both sides. Falling TTFT
  retires prefill and admission; unchanged acceptance retires the drafter; a 3×
  per-step cost is what eager dispatch of a 60+-layer MoE decode costs.

## The reporter's own toggle matrix, row by row

Their follow-up offers seven rows, every one measured at the collapsed N and
labelled "collapses". Four of them are not ablations of this mechanism at all,
and two of them actually **confirm** it — say which, instead of conceding:

| Toggle | What it does at this pin | Predicted cliff | Line to ask for |
|---|---|---|---|
| MTP off (`--speculative-config` removed) | dq 1, platform injects `min(264,512)=264`, the grid floors it to **256** | **unmoved at 256** | `Truncating max_cudagraph_capture_size to 256` in that run's log — the row that confirms the grid, because removing speculation did not move their cliff |
| `--enforce-eager` | `_set_cudagraph_sizes` takes the else branch: max 0, sizes `[]` | no cliff; a flat, slower curve on **both** sides | the same `--enforce-eager` at 256 *and* 264: flat pair = the step was the graph; a surviving step = it was not |
| `enable_npugraph_ex: false` | a compile-backend switch (`compilation/compiler_interface.py:283`); graph mode is still `FULL_DECODE_ONLY`, capture unchanged | unmoved | the startup capture list still printed |
| `--max-model-len 1024`, 512 output | no term in the formula; KV occupancy changes, the capture list does not | unmoved (they saw 256 → 258) | — context length is out, which also retires capacity |
| `--async-scheduling` off | does not write `cudagraph_capture_sizes` | unmoved | — |
| `VLLM_ASCEND_ENABLE_FLASHCOMM1` unset + `enable_dsa_cp: false` | ditto, unless `enable_sp` is also touched (`platform.py:555-567` rewrites the list) | unmoved | — |
| `multistream_overlap_shared_expert: false` | ditto | unmoved | — |

The rule the matrix needs: **a row refutes the mechanism only if it changes the
capture list or the graph mode *and* it was measured below the cliff as well.**
A row that changes neither can only move the absolute baseline. Ask for the
off/on pair at the **same** N; a single number at the collapsed N is not an
ablation. Two traps that make a correct diagnosis look refuted: `--enforce-eager`
deletes the graph (so it moves the baseline) and `enable_npugraph_ex` reads like
"graphs off" but is not.

## What is not the cause (cheap separations)

- **The HCCL driver.** Their bare-HCCL `dist.all_gather_into_tensor` benchmark
  (torchrun ×8, bf16, hidden 5120) was ~0.3 ms/op at every size from 256 to 640
  tokens. One flat microbenchmark retires the transport lane for the whole
  family; propose it instead of reading the collective.
- **A py-spy frame inside the attention path.** Their dump showed a worker
  stopped in `dist.all_gather_into_tensor` in
  `_maybe_all_gather_and_maybe_unpad_impl` (`dsa_cp.py:1296`). NPU streams
  execute in order, so that frame marks **where the device stream stalls**, not
  which host code refused to launch work — read it as a sync-point marker exactly
  like a device fault's deepest frame. It is also consistent with the graph
  story: an eager step is where the host finally waits on the queue.
- **Capacity / queueing.** `Waiting: 0`, KV 81-84 %, and TTFT falling across the
  step (see `index.md` metric rules) make capacity, admission and prefill out.

## Confirmation, cheapest first

1. **Grep the serve log** for `Truncating max_cudagraph_capture_size` and for
   `is smaller than the potential max tokens required for decode` — the first
   proves the effective ceiling is the grid's value (not the flag's), the second
   prints both numbers (`512` vs `528` at the recorded config). The capture list
   itself is in `Starting to capture ACL graphs for cases: [...]` and, when
   pruned, `Adjusted ACL graph batch sizes … N -> M sizes`
   (`<skill_root>/scripts/aclgraph-size-budget.sh` reproduces the prune).
2. **The A/B**: relaunch with
   `--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY",
   "max_cudagraph_capture_size": <max_num_seqs × (1 + num_spec)>}'` and repeat
   the *same* N above the cliff. Confirmed only if that N returns to its
   below-cliff rate. Above 256 the grid is 16 wide, so pick a value that lands on
   it (…, 512, 528, 544 …); 528 is on-grid and is the recorded value. An off-grid
   value is truncated to the previous grid point and the A/B then tests a
   boundary it did not intend — watch for the `Truncating` line in the new log. A
   value that conflicts with an explicit `cudagraph_capture_sizes` raises
   `customized max_cudagraph_capture_size … should be consistent`.
3. **Boundary stepping** (`--max-num-seqs` N and N+1, same client) only tells you
   where the step is; it does not identify the owner. Keep it for reports that
   refuse the config change.

## If the A/B does not move the cliff

Then the ceiling is not the binding constraint and this becomes an open question,
not a mechanism to keep digging for: the remaining candidate class is a host-side
structure indexed by running-sequence (or padded-token) count — static exchange
buffers sized from `potential_max_tokens` (its consumers *raise* rather than
degrade, which is why they are ranked low), CPU binding / thread pools
(`enable_cpu_binding` is on in the recorded serve line), or per-step Python lists
built over requests. Name one check per candidate and ask for the per-step
`Running batch with cudagraph_mode: …` DEBUG line (it names `NONE` versus `FULL`
explicitly on a collapsing step), the startup capture list, and a `py-spy` sample
on both sides of the cliff — the recorded `py-spy` sample exists only on the
collapsed side, which is why it cannot separate "eager step" from "stalled
collective". Do not present a new code area as the cause in this session —
`../evidence-workspace/delivery-discipline.md` is explicit about that.

## Recorded status

- Ceiling, grid, dispatcher and warning sites: source-verified at v0.23.0rc1
  `f4a08bd` / v0.23.0 `0fc695f`, including the MTP-off truncation
  (`264 → 256`) and the `npugraph_ex`/`cudagraph_mode` direction.
- Reporter measurements, published after the first round of replies: the tp=8
  256 → 258 step, the dp=2/tp=4 healthy run at ≤224 per engine with the
  aggregate ceiling 256 → 448, `--max-model-len 1024` still stepping at
  256 → 258, the flat bare-HCCL microbenchmark, the collapsed-side `py-spy`
  frame, the TTFT / wall-time / acceptance table, and the startup command with
  `enable_dsa_cp`, `multistream_overlap_shared_expert`, `enable_cpu_binding` and
  `block-size 128`.
- Both of their cliff indices reproduce from the serve line by the arithmetic
  above, and their MTP-off row is predicted by the same arithmetic — but the
  `max_cudagraph_capture_size` A/B was **not** run, and no startup log line has
  been shown. Say which of the two you are still waiting for; do not present the
  mechanism as reporter-confirmed.
