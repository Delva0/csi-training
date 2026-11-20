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

The threshold is per **engine**, and it is counted in *decode tokens*, not
requests — so it moves with `num_speculative_tokens` and with anything that
rewrites the capture-size list (`tp` changing the ACL-graph budget, SP). A
re-split such as dp=2/tp=4 multiplies the number of engines; it does not raise
the per-engine threshold.

## Do this first: the arithmetic (one call)

```bash
bash <skill_root>/scripts/seq-cliff-capture-budget.sh --max-num-seqs 264 --num-spec 1 \
    --max-num-batched-tokens 8192 --running 256,258,264
```

It prints the default capture ceiling, the capture-size list around the
boundary, and the sequence count where dispatch flips. For #14440 it prints
`CLIFF: 256 sequences healthy, 257 silently eager` — the reporter's own
256 → 258 observation, from the serve command alone. That is the verdict; the
rest of this page is confirmation and the ablations that do *not* count.

## The mechanism, pinned

| Site | Fact |
|---|---|
| `vllm_ascend/platform.py:207-241` (v0.23.0rc1) | `_get_default_max_cudagraph_capture_size` returns `min(max_num_seqs * decode_query_len, 512)`, `decode_query_len = 1 + num_speculative_tokens`; the docstring calls the removed trailing `* 2` "the only intentional difference from upstream" |
| `vllm_ascend/platform.py:259-261` | that value is injected in `apply_config_platform_defaults`, i.e. **before** upstream derives the list, so it is the value in force |
| `vllm/config/vllm.py:1618-1790` (v0.23.0) | `_set_cudagraph_sizes`: sizes `[1,2,4] + range(8, min(max+1,256), 8) + range(256, max+1, 16)`; the platform-injected max is clipped by `max_num_batched_tokens`; a user-set value that is not the list's max is **truncated** (`:1755-1759`) |
| `vllm/v1/cudagraph_dispatcher.py:276-285` | `num_tokens > max_size` → `return CUDAGraphMode.NONE` — no log, no warning. This is the "silent" half |
| `vllm_ascend/utils.py:1173-1204` | `_compute_potential_max_tokens` compares `max_cudagraph_capture_size` with `min(max_num_batched_tokens, max_num_seqs*dq)` and emits **one** startup `warning_once`: "The max_cudagraph_capture_size (%d) is smaller than the potential max tokens required for decode (%d) … see the issue #8240". At the reported config it prints `512` vs `528` |
| `docs/source/user_guide/release_notes.md:844` | the project's own workaround sentence for the MTP×EP graph-capture family: `--compilation-config '{"max_cudagraph_capture_size": N}'` with `N = max_concurrency * (1 + num_speculative_tokens)` |
| `docs/source/tutorials/models/DeepSeek-V4-Flash.md:166,215` | the validated commands at this ref use `--max-num-seqs 32` / `64`; no in-tree baseline ever runs 256+ sequences, so "your command differs from the tutorial" is not the answer here |
| `vllm_ascend/platform.py:76,113-124,565` | the size list can be pruned (`MAX_CAPTURE_SIZES_FOR_950`) and rewritten by the sequence-parallel path, which also overwrites `max_cudagraph_capture_size` — the reason a `tp`-dependent cliff position is expected |

Three consequences worth stating in the answer:

- Requested concurrency is not what the dispatcher sees. With MTP, a 264-request
  decode step is 528 tokens; the graph ceiling is a **token** budget, so the
  cliff lands at `floor(max_capture / decode_query_len)` requests.
- Under the ceiling the step is a graph replay; over it every step is eager
  Python + per-op dispatch, which is exactly the observed shape (NPUs idle,
  host cores busy, per-request rate collapsing while `Waiting` stays 0).
- The cliff can move with `tp` because the capture budget is a per-rank
  stream/allocation budget, which is why the same reporter saw the step between
  224 and 256 sequences per engine at tp=4 (dp=2) while tp=8 turned over at
  exactly 256. Read the ceiling from **that** run's log instead of assuming 512
  (`<skill_root>/scripts/aclgraph-size-budget.sh`, and the capture list in the
  startup log) — a second, memory-driven ceiling is a real possibility and is
  the one thing this page does not settle.

## Confirmation, cheapest first

1. **One grep of the serve log** for `max_cudagraph_capture_size` — the
   `warning_once` above prints the effective ceiling and the potential decode
   width. Its absence with graphs on means the ceiling already covers
   `max_num_seqs × (1 + num_spec)`; check the printed numbers rather than the
   presence of the line. The capture list itself is in
   `Starting to capture ACL graphs for cases: [...]` and, when pruned,
   `Adjusted ACL graph batch sizes … N -> M sizes`
   (`<skill_root>/scripts/aclgraph-size-budget.sh` reproduces the prune).
2. **The A/B**: relaunch with
   `--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY",
   "max_cudagraph_capture_size": <max_num_seqs × (1 + num_spec)>}'` and repeat
   the *same* N above the cliff. Confirmed only if that N returns to its
   below-cliff rate. Above 256 the grid is 16 wide, so pick a value that lands
   on it (…, 512, 528, 544 …) — an off-grid value is truncated to the previous
   grid point and the A/B then tests a boundary it did not intend. Watch for `Truncating max_cudagraph_capture_size` in the
   new log: a truncated value means the A/B never took effect, and a value that
   conflicts with an explicit `cudagraph_capture_sizes` raises
   `customized max_cudagraph_capture_size … should be consistent`.
3. **Boundary stepping** (`--max-num-seqs` N and N+1, same client) only tells you
   where the step is; it does not identify the owner. Keep it for reports that
   refuse the config change.

## Ablations that cannot refute it — say so

The recorded reporter ran a wide matrix at N=264 and labelled every row
"collapses": MTP off (712 tok/s), `--enforce-eager` (805), `enable_npugraph_ex:
false` (1062), `--async-scheduling` off (795),
`VLLM_ASCEND_ENABLE_FLASHCOMM1` unset + `enable_dsa_cp: false` (1277),
`multistream_overlap_shared_expert: false` (1135). Each of those changes the
per-step cost of *both* sides, and none of them was measured at its own
below-cliff N, so none is a refutation:

- `--enforce-eager` removes the graph entirely — there is no ceiling left to
  cross, and the healthy baseline got slower too. One row that *does* constrain
  the mechanism is `--max-model-len 1024` with 512-token outputs, where the
  reporter saw the same 256 → 258 cliff: the cliff is independent of context
  length and KV occupancy.
- The features toggled (FlashComm1, DSA-CP, npugraph_ex, multistream,
  async-scheduling) all change absolute throughput and none of them changes
  `max_cudagraph_capture_size`, so they cannot move a token-budget boundary
  either way.
- Always ask for the off/on pair at the **same** N; a single number at the
  collapsed N is not an ablation.

## What is not the cause (cheap separations)

- **The HCCL driver.** The same reporter's bare-HCCL
  `dist.all_gather_into_tensor` benchmark (torchrun ×8, bf16, hidden 5120) was
  ~0.3 ms/op at every size from 256 to 640 tokens. One flat microbenchmark
  retires the transport lane for the whole family; propose it instead of reading
  the collective.
- **A py-spy frame inside the attention path.** Their dump showed a worker
  stopped in `dist.all_gather_into_tensor` in
  `_maybe_all_gather_and_maybe_unpad_impl` (`dsa_cp.py:1296`). NPU streams
  execute in order, so that frame marks **where the device stream stalls**, not
  which host code refused to launch work — read it as a sync-point marker
  exactly like a device fault's deepest frame.
- **Capacity / queueing.** `Waiting: 0` and KV well under 100 % at the collapsed
  point (see `index.md` metric rules) make capacity and admission out.

## If the A/B does not move the cliff

Then the ceiling is not the binding constraint and this becomes an open
question, not a mechanism to keep digging for: the remaining candidate class is a
host-side structure indexed by running-sequence (or padded-token) count — static
exchange buffers sized from `potential_max_tokens` (its consumers *raise* rather
than degrade, which is why they are ranked low), CPU-binding thread pools, or
per-step Python lists built over requests. Name one check per candidate and ask
for the per-step `Running batch with cudagraph_mode: …` DEBUG line (it names
`NONE` versus `FULL` explicitly on a collapsing step), the startup capture list,
and a `py-spy` sample on both sides of the cliff. Do not present a new code area
as the cause in this session — `../evidence-workspace/delivery-discipline.md` is
explicit about that.

## Recorded status

- Arithmetic and the four pinned sites: source-verified at v0.23.0rc1 `f4a08bd`
  / v0.23.0 `0fc695f`.
- The cliff position (256 → 258 at tp=8; the same cliff at
  `--max-model-len 1024`; per-engine, dp=2/tp=4 healthy at ≤224 with the
  aggregate ceiling doubling) is the reporter's own measurement, published after
  the first round of replies.
- Not yet confirmed by the reporter: the `max_cudagraph_capture_size` A/B and
  the startup warning text from their log. Say which of the two you are still
  waiting for.
