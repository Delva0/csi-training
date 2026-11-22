# `container_memory_working_set_bytes` climbing for days: host memory, not HBM

Symptom family: the reporter watches a **container** metric in a k8s/cloud
monitor and pastes the curve, not a log — "容器内 `container_memory_working_set_bytes`
持续增加，7 天增加 80G", "container memory keeps growing, the service never
crashed", sometimes with `container_memory_rss` / `container_memory_cache` beside
it. `npu-smi` is flat, `/health` is 200, requests are served at a steady rate, and
there is no traceback, no OOM and no device error. #16563 is the worked case:
0.23.0RC1-era DeepSeek-V4-Flash on one 800I A2, TP8 + EP + MTP + `--async-scheduling`,
20-40 req/s, +80 GB in 7 days.

This is the **host** half of `index.md` part 3 without the crash: the host is not
out of memory yet, one cgroup counter is rising. Do not route it to
`hbm-growth-and-oom.md` (that is `npu-smi` / `NPU out of memory`), and do not
route it to `host-ram-and-per-rank-rss.md` (that is a `top` paste with a dead
box) — the deliverable here is which process class and which term owns the rise.

## Step 1 — take the metric apart before any theory (two asks, both cheap)

`container_memory_working_set_bytes` is the cgroup's `memory.usage_in_bytes`
(cgroup v2: `memory.current`) minus `total_inactive_file`: it sums the
**anonymous memory of every process in the container** (OpenAI frontend +
EngineCore + one worker per TP rank) **plus active file cache**. Definitional,
cAdvisor/cgroup — not a vllm-ascend claim, but it is what closes the "is this
even a leak" question, and it excludes the whole device-side story in the
report's environment block.

Ask for, at two timestamps:

1. the three series side by side — `container_memory_working_set_bytes`,
   `container_memory_rss`, `container_memory_cache`;
2. per-PID RSS: `ps -eo pid,rss,nlwp,cmd --sort=-rss | head -15` (RSS in KiB), so
   the growing process class is named instead of guessed, plus one sample while
   the service is **idle** — growth at idle is not on the request path.

Reading: `cache` rising with `rss` flat = page cache (serve log redirected into
the container's writable layer, checkpoint pages re-faulted), not a leak, and the
cheap fix is where the log goes. `rss` rising with `cache` flat = process
retention. `scripts/working-set-arithmetic.sh --samples <paste|->` prints that
split from a pasted prometheus/cAdvisor block in one call, one block per
container series (read the block for the container the reporter watches). With no
paste, quote the two commands above in the answer and rank on the arithmetic in
step 2.

## Step 2 — the arithmetic fixes the mechanism class before any code read

`scripts/working-set-arithmetic.sh --growth 80GiB --days 7 --rps 20-40` prints
bytes/request and bytes/decode-step. #16563: 80 GiB / 7 d = 8.1 MiB/min (≈139 KiB/s),
so **≈3.5-7 KB per request** at 20-40 req/s — a small host object retained per
request (frontend/engine request lifecycle, a per-request cache entry), or
per-request allocator churn. The same growth is only **7-14 bytes per decode
step** at that rate (10-20 K steps/s at ~500 output tokens per request), which is
smaller than any real per-step allocation — that is what prices out a per-step
host buffer (step 4) without reading any source.

Redo the arithmetic when the window is hours rather than days, or the rate is
tens of requests per minute: per-step then becomes plausible and the ranking
changes. State the assumption (request rate over the whole window) in the answer;
if the reporter cannot supply a rate, the arithmetic is the open question.

## Step 3 — jemalloc before Python-heap theory (this project prescribes it)

The model tutorials export `LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2`
with **no** `MALLOC_CONF` (`docs/source/tutorials/models/DeepSeek-V4-Flash.md:152`;
`docs/source/developer_guide/performance_and_debug/optimization_and_tuning.md:68-78`),
so a reporter who followed the tutorial is running jemalloc on default decay with
`background_thread:false` — freed dirty pages may stay resident. Two one-window
ablations decide the fragmentation-vs-leak question: `MALLOC_CONF=background_thread:true,dirty_decay_ms:0,muzzy_decay_ms:0`,
and a run without `LD_PRELOAD`. Unverified as the cause in this family: it is the
discriminator, never the diagnosis.

## Step 4 — the config diff against the project's own recipe

The reporter's command usually *is* a copied tutorial command with additions, so
diff it against the pinned recipe before reading source:
`docs/source/tutorials/models/DeepSeek-V4-Flash.md:143-186` (A2 single node).
#16563's deviations, each one ablation row:

| Deviation in #16563 | Pinned recipe / code | Reading |
|---|---|---|
| `--enable-prefix-caching` | `--no-enable-prefix-caching` (`:170`, explained `:321`) | documented off for this hybrid model; row 1, and it also changes KV/device behaviour, so it is not evidence about a host metric on its own |
| `VLLM_ASCEND_ENABLE_FLASHCOMM=1` | `additional_config.enable_flashcomm1: true` (`:184`) | at v0.26.0rc1 `enable_sp()` reads `additional_config.enable_flashcomm1` or env `VLLM_ASCEND_ENABLE_FLASHCOMM1` (`vllm_ascend/utils.py:861-889`, `vllm_ascend/envs.py:72`) — the no-`1` name has no consumer, and `enable_dsa_cp: true` without SP raises `ValueError: DSA CP requires SP to be enabled …` (`utils.py:1386-1405`). Their 0.23.0rc1 build started, so the names differ there: check the installed package (`rg -n 'FLASHCOMM' <site-packages>/vllm_ascend/envs.py`) before attributing anything to DSA-CP |
| `VLLM_ASCEND_APPLY_DSWA_PATCH=1` | absent at the pin | same version-correspondence check; a name with no consumer cannot cause or fix growth either way |
| `--async-scheduling` | not in the recipe | row 2 (it changes the host↔device copy path, `model_runner_v1.py:405-411`) |
| `PYTHON_NPU_ALLOC_CONF=expandable_segments:True` | `PYTORCH_NPU_ALLOC_CONF` (`:151`), injected by `vllm_ascend/platform.py:767-785` | the misspelled name has no consumer; more importantly that variable governs **device** allocation and cannot explain a cgroup host metric — say so, it removes the reporter's first suspect |
| `--safetensors-load-strategy prefetch` | absent from the A2 recipe, which sets only `--model-loader-extra-config` with `enable_multithread_load` / `num_threads: 128` | one-time page-cache/peak term from the weight load; a plateau, not a 7-day slope |

## Step 5 — pinned host-allocation sites, with their size class

- `vllm_ascend/worker/model_runner_v1.py:1411-1416`: the MTP spec-decode metadata
  path builds five CPU tensors per step with `torch.from_numpy(...).pin_memory().to(...)`,
  under its own `# TODO: Optimize the CPU -> NPU copy.`, while the same runner
  preallocates its other pinned host buffers once (`:407`, `:452`, `:458`, `:564`).
  Verified at v0.26.0rc1. Cite it as an asymmetry plus the MTP-off ablation row,
  **not** as this report's leak: the shapes are bounded by `--max_num_seqs` and
  `num_speculative_tokens`, and step 2's arithmetic already caps what it can owe.
- The hybrid prefix-cache coordinator patch
  (`vllm_ascend/patch/platform/patch_kv_cache_coordinator.py`, DSV4) keeps its
  block hashes and hits in the engine's block pool, which is bounded by the number
  of KV blocks; the part of that patch read at the pin is block/KV-keyed, with no
  request-id registry — so do not spend the session inside the
  connector/coordinator hunting a host dict. A real frontend retention would be
  upstream vLLM's request lifecycle, and the per-PID RSS sample from step 1 is
  what sends you there.

## Answer shape (truncation-safe)

1. Verdict sentence: this counter is container **host** memory (all processes +
   active page cache), not HBM; the `npu-smi` / `PYTORCH_NPU_ALLOC_CONF` /
   `gpu-memory-utilization` part of their environment cannot move it.
2. The split to run first (three cAdvisor series, per-PID RSS, one idle sample),
   then 3-4 ranked causes, each with its own one check and action directly after
   it — a per-request KB-class host object, page cache/log location, jemalloc
   decay, MTP/prefix-caching single-axis ablation.
3. One ladder of ≤6 lines, cheapest first; precedents in one line
   (`#2650`, `#5847`, `#12345`, `#3002` — see below); 3-5 lines of version scope
   and the artefacts still owed.
4. Keep the body ≈3-4 K characters. Five causes with a sub-paragraph each, a
   precedent table and a ladder is what ran into the output cap in #16563 even
   with the draft file already written.

## Precedent family (class-level only; not a diagnosis of the current host)

| Issue | Version / shape | Takeaway |
|---|---|---|
| #2650 | DeepSeek + torchair, A3 8-node PD, jmeter long run | D node **host** memory +1-2 GB/h, +100 GB in a 3 h mixed run — same metric family and magnitude, mechanism never stated in the body |
| #5847 | v0.13.0.rc1, CPU-memory leak reproduced with vllm-ascend alone | the leak can live entirely in vllm/vllm-ascend, not in the reporter's launcher |
| #12345 | DeepSeek-V4-Pro on A3/A5, PD and co-located | memory growth on both sides, reporter unsure whether device or host — ask which number grows before ranking |
| #3002 | attention-mask family, Qwen3-30B-A3B-Instruct-2507 at a 256K `max_seq_len` | a host-memory **peak** family (≈ 576 GiB peak per rank at 256K, see `attention-mask-host-allocation.md` for the arithmetic and version scope), so a report of that shape is an allocation-size bug, not a slow leak — and when the box dies instead of leaking, that page is the route, not this one |

## What the answer still owes when the paste is only a curve

Two timestamped samples of the three cAdvisor series; per-PID RSS at the same two
times plus one idle sample; whether `vllm serve` stdout is redirected to a file
inside the container and whether the checkpoint is on the container's writable
layer; two `/metrics` samples (`vllm:num_requests_running`, KV usage, prefix-cache
hit rate) to prove traffic really was steady; the request shape (prompt/output
tokens, `stream`, client disconnects); the container memory limit; and whether the
service's own reported ref exists in the snapshot at all — #16563 reports
0.23.0RC1 against a v0.26.0rc1 pin, so every `path:line` needs that scope.

Pins: the `path:line` above are verified at vllm-ascend **v0.26.0rc1**
(`f2f74a16c3c5`) + vllm **v0.26.0** (`568afb3a1380`) — when `SOURCES.json` names
those two, cite them as they stand and spend the turns on the delivery; re-grep only if
the pin moved or a line is marked version-specific (`utils.py:861-889`'s env name, the
`enable_dsa_cp` gate, the recipe block).
