# Long-run OOM and "HBM 不释放": capacity, growth, or a dead flag

## Signature family

- The server was healthy (`/health` 200, requests served), then after minutes to
  hours the process dies or the frontend starts its shutdown sequence; the
  reporter's own words are "显存不释放 / fails to release VRAM".
- Crash tail: `NPU out of memory. NPUWorkspaceAllocator tried to allocate
  216.00 MiB (NPU 0; 61.27 GiB total capacity; 295.32 MiB free)` →
  `torch.OutOfMemoryError … current working operator name is
  SelfAttentionOperation` → `Memory_Allocation_Failure(EL0004)` (#12692,
  reporter-reported). The operator name is the allocation site at failure time,
  not the owner.
- Sometimes the failing allocation names a collective (`HcclAllGather`, #13185) or
  a P/D pair (P node holds KV the D node never consumes, #13240), or HBM simply
  sits far above the documented budget (#13248).

## Step 0 — the stack's own line, before the split

Read `torch_npu==` out of the same `collect_env.py` block: **`2.8.0` has a known
leak and `2.8.0.post1`/`.post2` (+ a vllm-ascend rebuild) fix it** —
reporter-confirmed on #6299, where the growth surfaced in worker host RSS and
`--no-enable-prefix-cache` did not touch it (`host-ram-and-per-rank-rss.md`
§Step 0). One pip line, already fixed once for a reporter, so it ranks above every
growth mechanism below while the paste carries that version.

**When the report carries no env block, the pin's own pin line decides whether the
lane exists.** `rg -n 'torch-npu|torch==' requirements.txt pyproject.toml` at the
case's ref: v0.11.0rc3 pins `torch-npu==2.7.1` (`requirements.txt:13,28`;
`pyproject.toml:16-17`), so nothing about that release *ships* 2.8.0, and the bullet
naming the leak is not in that pin's `release_notes.md` (newest section
`v0.11.0rc0`) — a `leak` grep there returning only the P/D `#2650` bullet and a
v0.9.x CPU-leak fix line is a version gap, not a refutation. Then rank it only on the
reporter's `collect_env.py` line; recorded #4766 ranked it first on an assumption, paid
two greps, and demoted it after reading the pin (it was not the answer there).

## Step 1 — split the class with two samples, not with source

Ask for: `npu-smi info` during the run **twice, timestamped**, the serve log's
startup memory lines, and the crash tail. Then:

| What the samples show | Class | Go to |
|---|---|---|
| HBM flat and near-full from the start; one extra allocation fails | capacity / transient peak | Step 2 |
| HBM rises monotonically while request load is steady | growth (fragmentation or leak) | Step 3 |
| Reporter believes graph mode / scheduler / offload is enabled | dead keys — verify before any memory theory | Step 4 |
| HBM steps **up** during the first high-concurrency loop, then holds flat (idle samples equal, nothing rising) and the *next* loop of the same benchmark is what dies | per-batch high-water retention vs per-request retention — one concurrency sweep decides | Step 3b |

## Step 2 — capacity branch: the launch budget is the cause

- **Documented hardware floor.** The model tutorial states the minimum node: for
  Qwen3.5/3.6-27B, `1 Atlas 800 A3 (64G × 16)` or `1 Atlas 800 A2 (64G × 8)`
  (v0.23.0rc1 `docs/source/tutorials/models/Qwen3.5-27B-Qwen3.6-27B.md:23,28`
  and `:199`). A 4 × 32 GB 910B4 host serving that model at a 64K window is
  **below the documented floor** — say so first, it is the cheapest answer.
- **`--max-num-batched-tokens` sets the profile peak, and the KV pool is what is
  left.** Tutorial explanation of the arithmetic (util × HBM − peak) at `:296`;
  its value for this model is `8096` (`:236`, `:273`) versus `65536` in #16022 —
  an 8× transient activation headroom on 32 GB cards.
- **`--enable-prefix-caching`** is documented off for this model
  (`--no-enable-prefix-caching`, `:239`, `:492`): its hybrid KV layout can inflate
  the effective `block_size` (`:297`).
- **Confirm from the log, not the flags:** `GPU KV cache size: N tokens` and
  `Maximum concurrency for <max-model-len> tokens per request: Mx`
  (`vllm/v1/core/kv_cache_utils.py:2146,2148` at vllm v0.24.0 — re-grep in the
  *installed* vllm, because the snapshot pin may differ from the image, as in
  #16022 where `SOURCES.json` pins vllm v0.24.0 while the image is vllm-ascend
  v0.23.0rc1). `M < 1` means one full-length request cannot
  fit and the server preempts/recomputes forever.

## Step 3 — growth branch: the allocator environment gate comes first

- vllm-ascend fills `PYTORCH_NPU_ALLOC_CONF` with `expandable_segments:True` by
  default and **skips that injection** when the variable already contains
  `max_split_size_mb` or `garbage_collection_threshold`, and always in sleep
  mode (`vllm_ascend/platform.py:761-779` at v0.23.0rc1). The project FAQ points
  fragmented-OOM reports at this variable (`docs/source/faqs.md:141-149`).
- Therefore `-e PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256` in a launch script
  silently turns the anti-fragmentation default **off** — it looks like tuning
  and acts like a regression.
- The same value can also be the *failing site* rather than a tuning choice: on a
  host whose driver rejects the segment unmap, `expandable_segments:True` dies in
  `torch.npu.empty_cache()` with `unmapHandles … 507899` before the server is ready,
  and `expandable_segments:False` starts it (reporter-confirmed #12534) —
  `allocator-expandable-segments-unmap-crash.md`. Decide which lane you are in by
  the frame (an `AclrtUnmapMem` inside `empty_cache` = that page; a request-path OOM
  = this page).
- The effective value is logged: `Set PYTORCH_NPU_ALLOC_CONF=%s`
  (`platform.py:779`). One grep of the reporter's serve log
  (`grep -m1 'Set PYTORCH_NPU_ALLOC_CONF' <log>`) settles the branch; no source
  walk and no request replay needed.
- If the env is already correct: distinguish reserved growth from requested
  growth. Reserved-only growth is allocator/planner behaviour (try the env
  change, `--gpu-memory-utilization` headroom, and dropping graph capture);
  monotonically rising *requested* memory points at a request-path leak — KV
  connector/offload, prefix cache, or a captured-graph path that keeps tensors.
  That distinction is what decides between "config" and "bug", so make the
  reporter state which number grows.

## Step 3b — the amount that tracks concurrency, not request count

A load generator produces a shape the two-sample rule reads wrong: usage jumps
during the **first** loop, then holds flat, no allocation fails until the *next*
identical loop, and the reporter's words are "测完不释放" rather than "持续增长".
One experiment separates the two owners and it is cheaper than any source read:
from a fresh start, replay a **fixed slice of the same dataset** twice — once at
concurrency 1, once at the benchmark's concurrency — and read the number after each
loop has finished and the server is idle (`npu-smi info`, or
`torch_npu.npu.memory_reserved()` against `torch_npu.npu.memory_allocated()`, or the
engine's own KV usage metric).

- tracks **concurrency** and plateaus when the loop stops → per-batch high-water
  retention: activations/workspace for the in-flight sequences (or a captured graph
  pool) kept by the allocator as reserved bytes. Launch parameters bound it —
  `--max-num-seqs`, `--max-num-batched-tokens` (Step 2's profile-peak arithmetic) —
  plus the allocator env gate in Step 3.
- tracks the number of **requests** at fixed concurrency → per-request retention
  (encoder cache, prefix cache, KV connector/offload, a captured path holding
  tensors). No launch parameter bounds it; the reserved-vs-requested split in
  Step 3 then names which one it is.

Recorded, and still open: #3198's reporter followed up on the same
"显存占满，长期不释放，再压测就崩溃" thread (2025-10-28) with `vllm 0.11.0rc0` (their
spelling — vllm vs vllm-ascend is not distinguished), **Qwen2.5-VL-32B, TP2,
TextVQA, `concurrency: 128`, total prompt 512**: roughly **21 GiB held after the run
and released by nothing**, and the next TextVQA loop drives the card to OOM. Three
readings, none of which needs the source: the retained amount is reached while the
total prompt is small, so it is not prompt-length KV; it survived a **release and a
model change** from the original 0.10.2rc1 / 7B report, so a pin-specific finding (a
dead `additional_config` key, one `--max-num-batched-tokens` value) is a secondary
lever and belongs below the class split; and the reporter supplied neither the
reserved/allocated split nor a `--max-num-seqs` sweep, so Step 3b *is* the missing
measurement — ask for it rather than ranking a mechanism.

## Step 4 — `--additional-config` keys are silently ignored

- At v0.23.0rc1 only `xlite_graph_config` is parsed
  (`vllm_ascend/ascend_config.py:37-38`, fields at `:651-656`).
  `torchair_graph_config` / `use_cached_graph` were **removed**
  (`docs/source/user_guide/release_notes.md:1276`: "Torchair graph mode is
  removed. `--additional-config {"torchair_graph_config": {"enabled": true}}`
  doesn't work anymore"), and `ascend_scheduler_config` no longer has any
  consumer in `vllm_ascend/` (only stale references under `tests/`).
- Nothing validates unknown keys — every reader is `additional_config.get(...)` —
  so a stale key starts the server with the feature **off and no warning**. A
  launch script carrying `torchair_graph_config` + `ascend_scheduler_config`
  therefore never enabled graph mode or the Ascend scheduler: do not blame them
  for the OOM, and do tell the reporter the features were never active (that
  also explains a performance expectation gap).
- Generalise: before attributing any behaviour to a flag in `additional_config`,
  run `rg -n '<key>' vllm_ascend/` at the pinned commit and check
  `release_notes.md` for a removal entry. Scope the "silently ignored" reading to
  pre-migration refs: once `AscendConfig` became a pydantic dataclass with
  `extra="forbid"` (v0.27.1rc1 era) the same stale key is a fatal
  `unexpected_keyword_argument` before any worker starts, so it is a startup
  question, not a silent no-op — `../startup-failures/additional-config-schema.md`.

## Step 4b — a multimodal (VL) memory report: the MM cache is not the HBM

Verified at vllm-ascend v0.11.0rc3 (`b6d63bb`) + vllm v0.11.0 (`b8b302c`); re-grep
per ref, because the same switches move between releases.

- `--disable-mm-preprocessor-cache` / `--mm-processor-cache-gb 0` size a **host** shm
  ring buffer, not HBM: `SingleWriterShmRingBuffer(data_buffer_size =
  mm_processor_cache_gb * GiB)` (`vllm/multimodal/cache.py:402-403`, again at
  `:620-621`). At vllm v0.11.0 the first switch is already DEPRECATED (removal
  announced for v0.13, `vllm/engine/arg_utils.py:387,969-975`) while still zeroing the
  value. So "I disabled the MM cache and HBM still grows" is *expected* — say it and
  drop the thread, do not investigate the preprocessor.
- `--mm-processor-cache-type shm` is the only setting that routes MM features through the
  **object store** (a named segment, `VLLM_OBJECT_STORAGE_SHM_BUFFER`); it is not the
  default (`lru`), and it can abort the engine with a stale-slot
  `ValueError: Data for address:id …` on a request — if the paste carries that line the
  report is a cache/flag question, not memory: `../multimodal-inputs/shm-mm-cache-invalid-object.md`.
- The VL side's HBM cache is the **encoder** cache: `encoder_cache` is keyed by mm
  hash, filled per image and popped on `free_encoder_mm_hashes`
  (`vllm/v1/worker/gpu_model_runner.py:273,550,1601,3402`;
  `vllm_ascend/worker/model_runner_v1.py:283,617,1005,2600`), bounded by
  `encoder_cache_size` and released at request end.
- The effective default `--max-num-batched-tokens` on a < 70 GiB card is **2048 even
  for a VL model**: `_set_default_args` fills the field first
  (`vllm/engine/arg_utils.py:1595,1617-1626`, called at `:1162`) and `SchedulerConfig` is
  only constructed afterwards (`:1360`), so its `is_multimodal_model` → 5120 floor
  (`vllm/config/scheduler.py:187-202`) never fires; `encoder_cache_size =
  max_num_encoder_input_tokens = max_num_batched_tokens` follows it (`:201-202`).
  Confirm from the report's own log line
  (`Chunked prefill is enabled with max_num_batched_tokens=N`), never from the flags.
- A user `--block-size` on a **non-hybrid** model at this ref is rewritten to 128
  whenever prefix caching or chunked prefill is on (`vllm_ascend/platform.py:269-285`,
  warning `If chunked prefill or prefix caching is enabled, block size must be set to
  128.`, `qwen3_next` exempted) — so `--block-size 16` is inert and is not the cause
  either way. `refresh_block_size`, the site newer refs use, does not exist yet; see
  `../accuracy-benchmarks/hybrid-block-size-corruption.md` for the hybrid form of the
  same rule.
- **First ask which tail the crash carries; if it is not an OOM, do no memory
  arithmetic.** At the v0.11.0 line an image-dataset benchmark (TextVQA, and
  InfoVQA/DocVQA in the same report) killed `EngineCore` with `Program terminated
  with signal SIGBUS, Bus error` — the frame in CPython's memoryview copy
  (`copy_single` / `memory_ass_sub`, `Objects/memoryobject.c`) — for **both**
  Qwen3-VL-30B-A3B and Qwen2.5-VL-7B, on a different test case each replay, with
  `EngineDeadError` at the API server and `There is no error in any plog files`
  (#3513, vllm-ascend 0.11.0rc0 + vllm 0.11.0rc3, `torch_npu 2.7.1.dev…`; left
  unresolved in that corpus). A write past a mapped buffer's end is a host-side
  buffer-length fault, so it classifies away from HBM growth, and two different VL
  models dying the same way is what rules out one model's weights — send the tail
  line rather than a memory ladder (mechanism open: that report was never carried to
  a resolution in its corpus).

## Step 4c — older refs: the allocator gate may not exist at all

At v0.11.0rc3 nothing in `vllm_ascend/platform.py` touches `PYTORCH_NPU_ALLOC_CONF`:
`rg -n PYTORCH_NPU_ALLOC_CONF vllm_ascend/` hits only
`device_allocator/camem.py:147`, and `tests/e2e/*` set `max_split_size_mb:256` for
itself. The Step 3 injection and its `Set PYTORCH_NPU_ALLOC_CONF=%s` log line are
**later-release** behaviour, so on an older tree their absence proves nothing about a
shadowing env var — the anti-fragmentation default is simply missing, and the
reporter has to export `expandable_segments:True` themselves. State which release the
injection exists in before asking for that log line. One build-level exception:
the CaMem memory pool asserts `expandable_segments:True` away
(`vllm_ascend/device_allocator/camem.py:153` at v0.26.0rc1), so on a pooled/KV-pool
deployment that value raises instead of helping —
`../hccl-and-hardware/aiv-mc2-env-ablation.md`.

## Step 5 — deliverable order (truncation-safe)

1. One-line verdict: capacity or growth, plus the single cheapest parameter
   change (`--max-num-batched-tokens`, drop `max_split_size_mb`, or the hardware
   floor), with the version scope.
2. Ranked causes, each carrying **its own action** immediately after its
   evidence — a closing "action table" is the first thing lost if the message is
   cut.
3. Then 3-5 lines of scope / open questions / what to send back: two timestamped
   `npu-smi` samples, the `Set PYTORCH_NPU_ALLOC_CONF` line, the KV-cache and
   preemption lines, the full crash tail, and both serve commands if P/D.

## Precedent family (class-level only; not a diagnosis of the current host)

| Issue | Version / shape | Takeaway |
|---|---|---|
| #12692 | v0.20.2rc1, 397B-A17B w8a8c8, diverse long + multimodal, OOM at ~37 min | `NPUWorkspaceAllocator` + "current working operator name is …" is a late allocation site; reporter's profile-peak / KV / weights numbers are usable arithmetic |
| #13185 | v0.13.0, Qwen3-VL-32B long run | a collective name in a late OOM (`HcclAllGather` alloc failure) is still the alloc site, not the collective |
| #13248 | v0.23.0rc1, P/D w8a8c8 + MooncakeLayerwise | HBM abnormally high then P node exhausts; check connector/offload paths per side |
| #13240 | P/D disaggregation | D node errors and the P node never frees KV → P node HBM exhausted while waiting; a peer-side lifecycle bug, not fragmentation |
| #3198 | v0.10.2rc1, Qwen2.5-VL-7B, benchmark loops; reporter follow-up on `vllm 0.11.0rc0`, Qwen2.5-VL-32B TP2, TextVQA `concurrency: 128` | the literal "显存占满，长期不释放，再压测就崩溃" phrasing that recurs in this family; ~21 GiB held after a 128-way loop, then OOM on the next loop — the class stayed open through a release and a model change (Step 3b) |
| #2650 | v0.9.1, DeepSeek + torchair | long-run leak under graph mode; graph paths are a legitimate growth owner |

These are title/body-level precedents: quote them as `#<id>: <one fact>`, never as
confirmation that the current host has the same defect.
