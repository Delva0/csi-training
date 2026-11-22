# Memory: growth/OOM after a healthy period, and allocation failure before ready

Scope, part 1: the service came up, served requests, and later died or degraded on
memory — "跑了 X 时间后显存不释放，最后崩溃", "fails to release VRAM, eventually
crashes", `NPU out of memory`, `torch.OutOfMemoryError` /
`Memory_Allocation_Failure(EL0004)` on a long-running server, HBM pinned at
100 % in `npu-smi`, a P node that keeps KV it should have released.

Scope, part 2: the engine never became ready because an allocation failed —
`Memory_Allocation_Failure(EL0004)` / `alloc device memory failed, runtime result
= 207001` while the KV cache or the model weights are being put on the device.
Same decision (capacity vs growth vs dead key), different phase and different
cheapest test (a smaller `--max-model-len` instead of two timestamped samples).

Scope, part 3: the **host** ran out, not the card — the reporter says the machine
crashed (ssh lost, box unusable) or pastes a `top`/`free` with every python3 rank
holding tens of GiB RES, and the NPU log carries no device or collective error.
Host RAM is invisible to `npu-smi`, so it is the first thing a "machine crashed"
report needs a number for (`host-ram-and-per-rank-rss.md`). When there is no
`top` to sample because the box died **during engine start** — the log's last
lines are the per-rank `rank N in world size M is assigned ...` set and no
`Starting to load model` follows — the suspect is one eager host allocation, the
`max_model_len²` attention mask every worker builds
(`attention-mask-host-allocation.md`, `scripts/attn-mask-budget.sh`).

Scope, part 4: the host did **not** run out — a container-scoped counter climbed
for days while the service stayed healthy and `npu-smi` stayed flat:
`container_memory_working_set_bytes` / `container_memory_rss` /
`container_memory_cache` "持续增加，7 天增加 80G". That is host memory in a cgroup,
so the device-side knobs in the report's environment block are irrelevant and the
deliverable is which process class and which term (anonymous RSS vs page cache vs
allocator retention) owns the rise — `container-working-set-growth.md`.

Scope, part 5: **sleep mode** — `enable_sleep_mode=True` / `llm.sleep(level=…)`, an RL
sleep/wake loop, or "sleep 掉模型 A 再起模型 B" — where the second engine does not die of
OOM but of a **communicator creation** refusal (`sleep-mode-multi-instance.md`). Read it
as an offload-capability question: what `sleep()` returns to the allocator, and what it
leaves resident on the device.

Scope, part 6: the host had RAM to spare and no device error at all — a worker that
served for **days** dies inside a Triton-Ascend kernel compile with
`OSError: /tmp/tmpXXXX/libkernel.so: failed to map segment from shared object` (often
followed by `MLIRCompilationError: [ConvertLinalgRToBinary] encounters error` carrying the
same text). That is `mmap` returning ENOMEM because the **per-process mapping count**
(`vm.max_map_count`, default 65 530) is exhausted by leaked `dlopen` handles, not because
memory, disk or `/tmp` permissions are short — `dlopen-map-count-exhaustion.md`.

Scope, part 7: the process is **gone** and the card still reads occupied — the serve
died (OOM, crash, `pkill`), `npu-smi` keeps showing tens of GB, and the next serve cannot
take the card; often with the ask "register SIGTERM/SIGINT handlers so an abnormal exit
releases ACL resources". That is a process-attribution question before it is a memory
one: a row in the process table is a live context, and at v0.18.0 there is no exit-time
device release in vllm-ascend to add
(`../deploy-sizing/node-processes-and-device-attribution.md`, its "after the engine is
gone" section). Do not route it to the growth page: nothing here grew.

Scope, part 8: nothing failed and nothing grew — a **fixed device allocation lands
after `determine_available_memory` has closed**, so the KV budget is computed without
it and the instance quietly uses `gpu_memory_utilization` **plus** that delta: the ATB
`TilingBufferPool` (~96 MiB pool / ~98 MiB observed per rank at v0.18.0, scaling with
TP, created by `_warm_up_atb()` inside `compile_or_warm_up_model`), or an ACL-graph
capture transient in `capture_model()`. The entry form is often an issue that *names
the allocation and offers a patch* ("allocated after memory profiling and not
accounted for, should I open the PR"), which is a code-change review judged against the
accounting ledger. Recognise the delta by its shape: constant across `max_num_seqs`,
graph buckets and model size, per worker process, invisible to
`memory_allocated()`/`memory_reserved()` — `device-allocations-outside-profiling-window.md`.

Not an accuracy or corruption report, and not a collective error even when the
crash text names one (`HcclAllGather` in a late OOM is the allocation site); go
to `../hccl-and-hardware/index.md` first only when a real collective failure
accompanies the OOM. The one HCCL form that *is* this topic: `EI0007 … Failed to
allocate resource[qp] … Memory resources are exhausted` at graph capture with a
`[Create][DestLink]` line for every peer — budget, not link
(`../hccl-and-hardware/qp-resource-exhausted-at-graph-load.md`), and note the KV
cache is allocated before capture, so a start-up that fails there already holds
its `--gpu-memory-utilization` share.

Inherit the rules in SKILL.md. Rules that decide success here:

1. **Split the class before investigating**: capacity (pool or transient peak was
   wrong from the start), growth (reserved or requested HBM rises while the
   workload is steady), or a dead configuration key (the feature the reporter
   thinks is on never engaged). Each has a different owner and a different
   one-command check; the report's narrative rarely says which it is. Inside
   growth, a benchmark-loop report needs the second axis too — **per-batch
   high-water vs per-request retention** — and one concurrency sweep decides it
   (`hbm-growth-and-oom.md` Step 3b).
2. **The crash text is an allocation site, not a cause.** An async OOM names the
   *current* operator (`current working operator name is SelfAttentionOperation`,
   `HcclAllGather` failure) — the same sync-point trap as SKILL.md rule 11. The
   cause is whatever grew; say that explicitly before the reporter patches an op.
   The startup counterpart is a `npu_format_cast` frame in KV-cache
   initialisation: it is where a second buffer was requested, not an operator
   defect (`kv-init-and-format-cast-alloc.md`); an *unmap* failure is the same
   class of statement about a release site
   (`allocator-expandable-segments-unmap-crash.md`).
3. **Read the stack's `torch_npu` (pta) line in the same paste before any growth
   arithmetic.** pta leaks are real, the fix is one install, and the release's own
   pin can *be* the leaking version: at v0.13.0rc1 `requirements.txt:31` pins
   `torch-npu==2.8.0` — the leaking line the tree's own release notes name
   (`docs/source/user_guide/release_notes.md:197`, "a bug from torch-npu") and
   which `2.8.0.post1` / `2.8.0.post2` fix (reporter-confirmed on #6299: with
   `--no-enable-prefix-cache` the growth continued unchanged; installing `.post2`
   **and rebuilding vllm-ascend** ended it). So an env dump whose every number
   matches the release is exactly the paste that carries it, and a vllm-ascend
   source walk cannot refute it — the allocator layer it calls is not in the
   snapshot. `host-ram-and-per-rank-rss.md` §"Step 0" has the recorded case, the
   matching rule and the line that closes the lane (`torch_npu` already at
   `.post1`/`.post2` or a later pta line). **Scope the lane to the ref before you
   rank it, the more so when the paste carries no env block**: one
   `rg -n 'torch-npu|torch==' requirements.txt pyproject.toml` at the pin says
   whether the leaking line ships at all — v0.11.0rc3 pins `torch-npu==2.7.1`
   (`requirements.txt:13,28`; `pyproject.toml:16-17`), so a report that names that
   release is not the 2.8.0 lane unless the reporter's own dump shows a hand-installed
   pta. And that pin's `release_notes.md` does not carry the bullet (its newest
   section is `v0.11.0rc0`), so grepping the old pin's notes for `leak` and finding
   only the P/D `#2650` bullet and a v0.9.x CPU-leak fix line (`:537`) refutes
   nothing. Recorded #4766 ranked the leak first,
   spent two calls on those greps, then read `requirements.txt` and demoted it; the
   paste-side rung is `collect_env.py`, the pin-side one is that grep, in that order.
4. **Ask for two timestamped samples**, not one `npu-smi` dump: monotonic growth
   and a flat near-full pool look identical in a single sample.
5. **Check flag liveness before attributing behaviour to a flag.** On pre-migration
   refs vllm-ascend silently ignores unknown `--additional-config` keys (since the
   `@config` migration a pydantic `ValidationError` rejects them before any worker:
   `../startup-failures/additional-config-schema.md`), and graph/scheduler keys
   were removed or renamed across releases; `rg -n '<key>' vllm_ascend/` at the
   pinned commit is one call and prevents a whole wrong hypothesis. A multimodal
   model's `--mm-processor-cache-gb` / `--disable-mm-preprocessor-cache` size a host
   shm buffer, so their state says nothing about HBM (`hbm-growth-and-oom.md` 4b).
   Worked instance: **`--kv-cache-memory` / `--kv-cache-memory-bytes` is accepted by
   the CLI and inert on the NPU worker at v0.13.0rc1 (#6299) and v0.18.0rc1
   (#9372)** — upstream honours it only in `vllm/v1/worker/gpu_worker.py:362-380`,
   which the Ascend `determine_available_memory` override replaces, and
   `rg -n 'kv_cache_memory' vllm_ascend/` returns only its own profiling variable.
   It is live by v0.22.1rc1 — the generic worker gains the fast path at
   `worker.py:477-490` there (`kv-budget-per-instance-vs-device-global.md` 3), which
   is the same version family as the v0.21+ log text that *recommends* a
   `--kv-cache-memory=<bytes>` value. So re-grep per ref and never quote the flag on
   a v0.18-or-older serve.
6. Everything here is about the reporter's host: local `npu-smi` output is not
   evidence about their memory behaviour.
7. **A KV budget that comes out at ~0 or negative is a bookkeeping error before it
   is a small card.** When a device with free HBM hands the engine
   `Available KV cache memory: 0.09 GiB (halved for workspace)` (310P) or
   `Available memory: 0` / `-1.33GB` (the second of two co-tenant engines), the
   worker has charged usage that is not this instance's. One subtraction of the
   logged `Total non KV cache memory` against the requested budget names the
   offending term; `gpu_memory_utilization` is a **per-instance** limit
   (`vllm/config/cache.py:66-74` at v0.22.1), so "raise the ratio, it includes the
   other instance" is not the fix — `kv-budget-per-instance-vs-device-global.md`.
   **A single instance on a quiet device with `Available memory: 0` has the other
   owner**: the profiling dummy forward is sized by `max_model_len` there, so the
   peak term — not bookkeeping — is what exceeds the budget, and raising the ratio
   changes nothing (recorded #3843: 0.5 → 0.9, same error). Read `Using max model
   len N` from the log first; the mechanism and the context-preserving levers are
   `../startup-failures/max-model-len-budget.md`.
   On a *negative* budget the log line is not the whole user-visible surface: the
   scheduler path reaches upstream's `_check_enough_kv_cache_memory`, which already
   raises `ValueError("No available memory for the cache blocks. Try increasing
   `gpu_memory_utilization` …")` with an estimated max length, so a request to
   "add actionable suggestions" is measured against that text before any patch
   proposal (#9372: the recorded run quoted both the vllm-ascend line and that
   ValueError, and kept the deliverable a diagnosability finding rather than a
   crash).
8. **The exception *class* is a text predicate, so `torch.OutOfMemoryError` is not a
   measurement of HBM.** torch_npu raises that class only inside `NPU_CHECK_ERROR`'s
   generic `else` branch, gated on `c10_npu::isCannOOM(retmsg)`
   (`torch_npu/csrc/core/npu/NPUException.h:199-204` at
   `v26.0.1-pytorch2.10.0`; the same gate in `HCCLUtils.hpp:38`), over a `retmsg` that
   ends with `c10_npu::c10_npu_get_error_message()` — ACL's *recent*-error buffer
   (`:196-198`, declared `:337`). A stale `Memory_Allocation_Failure(EL0004)` /
   `alloc device memory failed, runtime result = 207001` from earlier in the same log
   therefore rides into a later failure's text and relabels it OOM. It cuts both ways:
   neither vLLM nor vllm-ascend mints or branches on the class (vllm v0.23.0 mentions
   `OutOfMemoryError` in only `v1/worker/gpu_model_runner.py` and a p2p NCCL engine), so
   "vLLM's OOM path logs the `--gpu-memory-utilization` hints" is unsupported at the
   pin. When the class is OOM but the text is a sync / stream / timeout code, read
   `oom-class-is-not-a-cause.md` before any capacity arithmetic — and correct the code
   names in the report (`107020` is `ACL_ERROR_RT_TASK_TIMEOUT`, not a synchronize
   timeout).
9. **A loader failure is a mapping question, never a memory question until the mapping
   count is printed.** `failed to map segment from shared object` with no errno suffix is
   glibc's `dlopen` reporting ENOMEM from `mmap`; `free -g`, `df -h /tmp` and a `noexec`
   check cannot confirm or refute it, and the recorded #10551 host had 979 GB free with
   171 GB free in `/tmp`. Two numbers decide it: `sysctl vm.max_map_count` and
   `wc -l /proc/<worker-pid>/maps`. The same reading applies to the `unmap` direction
   (`507899`), and to `MLIRCompilationError` text whose *detail* is an `OSError` — the
   compiler banner is the wrapper, the loader line is the cause
   (`dlopen-map-count-exhaustion.md`).

## Reading a v0.21+ startup memory block

Since v0.21.0 the workers profile the ACL graph pool, so a startup log carries new
lines beside `Available KV cache memory` (#10700, 2× A2, DP2 × TP8, `gmu=0.92`):

| Line | Read it as |
|---|---|
| `ACL graph memory profiling is enabled (default since v0.21.0). The current --gpu-memory-utilization=0.9200 is equivalent to --gpu-memory-utilization=0.9103 without ACL graph memory profiling. … increase … to 0.9297. To disable, set VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0.` | `--gpu-memory-utilization` no longer maps 1:1 onto KV: ~1 % of the budget goes to the graph pool. On that log the equivalent ranged `0.9101`-`0.9104` across TP ranks. |
| `ACL graph pool memory: 0.44 GiB (actual), 0.59 GiB (estimated), difference: 0.15 GiB (33.8%).` | estimate error of the pool (30-40 % of a sub-GiB number is normal here); it is the delta any KV arithmetic has to absorb, not a defect. |
| `Free memory on device (60.6/60.96 GiB) … Desired GPU memory utilization is (0.92, 56.08 GiB). Actual usage: 30.16 GiB for weights, 2.0 GiB for peak activation, 2.07 GiB for non-torch memory, 0.44 GiB for NPU graph memory. Replace gpu_memory_utilization with --kv-cache-memory=22829425254 (21.26 GiB) to fit into requested memory, or … (25.78 GiB) to fully utilize NPU free memory.` | a complete budget split plus a byte-exact alternative to gmu. Quote it instead of re-deriving weights/activation/KV with `scripts/oom-budget-split.sh`. A log printing it comes from a ref that reads the key, but on refs where the override does not (v0.18 and older, rule 5) the flag is inert — check the ref before passing the advice on. |

These lines are a **phase** test as much as a presence test, and the three do not
share an era or an emit site. The first two (`ACL graph memory profiling is enabled …`,
`ACL graph pool memory: … (actual), … (estimated)`) are the v0.21.0rc1 / v0.22.1rc1
*pre-capture estimate* pair: upstream still says `CUDA graph …` and gates
`profile_cudagraph_memory()` on `current_platform.is_cuda()`, which is false on NPU, and
neither pinned v0.23.0rc1 tree carries the ACL-worded pair at all. The third
(`Free memory on device … GiB for NPU graph memory … --kv-cache-memory=…`) **does**
exist at v0.23.0rc1 — `vllm_ascend/worker/worker.py:719` measures `capture_model()`,
`:729-757` builds the message, `:758` logs it, and the whole block is gated on
`kv_cache_memory_bytes is None` (`:728`) — so it is a *post-capture measurement plus
next-launch suggestion*, not a reservation,
and a run that dies inside capture never prints it (`f4a08bd`). So: the third line
present ⇒ capture finished and the pool number is in hand; absent in a failed-capture
paste ⇒ nothing about the reservation. Whether the pool was reserved *before* capture
is decided by code at that pin (`rg -c profile_cudagraph_memory vllm_ascend/` = 0, and
the activation peak is the pre-capture one — next paragraph), never by the missing line
(`../startup-failures/acl-graph-capture.md`, the v0.21–v0.23 section).

Caveat for cross-version work: the same `--gpu-memory-utilization` yields a ~1 %
smaller KV on v0.21+ than on an older release, and the per-rank `21.26` / `25.78`
GiB numbers differ by ~0.02 GiB between ranks. Neither is growth.

**The two KV figures in that line are an identity, not a warning.** `Current KV cache
memory` exceeds the `--kv-cache-memory=<N> … to fit into requested memory` suggestion by
exactly `NPU graph memory + 150 MiB` on every graph run, because
`available_kv_cache_memory_bytes = requested_memory − non_kv_cache_memory` comes from the
profile that runs **before** capture (`vllm_ascend/worker/worker.py:563-590` at v0.23.0,
`5cb98caaadef`; the comment says "Record torch peak INSIDE the context and BEFORE graph
capture"), while the suggestion subtracts `npugraph_memory_bytes` too (`:732`, `:743-750`,
`logger.info` at `:771`). `requested_memory` is the `gpu_memory_utilization` *policy*
target, never the card: physical headroom is `free_memory − (weights + peak activation +
non-torch + graph + KV)`. Recorded #16649 (`gmu=0.92` on a `60.61/60.96 GiB` device:
23.1 + 4.01 + 1.59 + 0.9 + 27.38 = 56.98 GiB accounted) leaves **~3.6 GiB unused** on the
card while a 20 MiB physical segment fails — but the same line read against `requested
56.08` looks like 0.3 GiB of headroom, and the recorded answer published "the config
overshot by ~1 GiB, 0.3-0.4 GiB left" and ranked `capacity` first on it. Read the line as
three numbers — policy target 56.08 / accounting 56.98 / physical free 3.63 — and treat a
shortfall against `requested` alone as expected on any CUDAGraph run; the growth sample
(`hbm-growth-and-oom.md`) and the allocator's segment map
(`allocator-expandable-segments-unmap-crash.md` §*The map direction*) are what the
physical free figure sends you to.

Conditional routes:

- Any memory-growth report, or "VRAM not released then crash" where the process is
  still running → `hbm-growth-and-oom.md` (the split, the concurrency sweep for a
  benchmark loop, the allocator environment gate, the KV / profile-peak arithmetic,
  the dead-key check, the precedent family).
- The service is **already dead** and `npu-smi` still shows HBM, so the next serve cannot
  start (or the reporter proposes a SIGTERM handler to "release ACL resources") →
  `../deploy-sizing/node-processes-and-device-attribution.md` §"After the engine is gone
  the card still shows HBM" first: attribute the PID, kill the process group, and note
  that vllm-ascend installs no exit-time device release (its only `atexit` unsubscribes
  print streams) while the upstream parent-death pipe only fires at a child's read point.
- The crash instead happens during `profile_run`/`determine_available_memory`
  before the server is ready → `../startup-failures/index.md`
  (`max-model-len-budget.md`) — that is a configuration budget, not a leak. If the
  model is a **VL/omni** checkpoint and the innermost frame is in the vision tower,
  the number is the dummy profiling batch's (`max_num_batched_tokens` × the
  per-modality defaults) and TP does not divide it:
  `../deploy-sizing/vl-profile-run-multimodal-budget.md` (recorded #1617).
- The report is **「推理卡死」 + a plog OOM** on a MoE model whose launch carries
  `--data-parallel-size` > 1 and `additional_config={'expert_tensor_parallel_size': N}`
  with **no** `--enable-expert-parallel`, and there is no traceback → leave this
  topic: `ep_size` is 1, the MoE's DP-wide output reduction is the owner, and the
  per-card bytes will keep coming out "it should fit"
  (`../feature-enable-recipes/dp-moe-padding-and-owner-slice.md`, *The DP output
  reduction*, recorded #2069 reporter-confirmed — a `reduce_scatter_tensor` call form
  that `get_dp_group().reduce_scatter` replaces).
- The traceback is a **loader/mapping** failure on a worker that ran for days —
  `OSError: /tmp/tmpXXXX/libkernel.so: failed to map segment from shared object` from
  `ctypes.CDLL(callback_path)` in triton-ascend's
  `linalg_to_bin_enable_npu_compile_A2_A3`, wrapped by
  `MLIRCompilationError: [ConvertLinalgRToBinary]` and surfaced as
  `RuntimeError: Worker failed with error '…'` on EngineCore →
  `dlopen-map-count-exhaustion.md`: the leak is 7 VMAs per JIT compile
  (`ctypes.CDLL`, no `dlclose`), the ceiling is `vm.max_map_count`, and the remedy is
  `sysctl -w vm.max_map_count=1048576` (live) plus a persisted `TRITON_CACHE_DIR` and the
  triton-ascend fix — **not** free RAM, `/tmp` space, `noexec`, or the compiler.
- The failure is a **free/unmap**, not an allocation: `unmapHandles …
  AclrtUnmapMem(… getHcclComm()) error code is 507899` with `ERR00100` /
  `EL0003`, raised in `torch.npu.empty_cache()` inside the `memory_profiling`
  window on every rank, right after graph compile → the expandable-segments
  allocator lane, `allocator-expandable-segments-unmap-crash.md` (reporter-confirmed
  #12534: `expandable_segments:False` starts; note vllm-ascend injects `True`
  unless the variable already carries another of the three keys, so "unset" is
  not "off"; if the reporter returns **after** that change with a stall instead of
  the crash, the page's §*If the same reporter returns …* holds the three checks —
  TP against the container's exposed `/dev/davinciN` set, the `107001` rank-death
  fingerprint vs the `shm_broadcast` watchdog, and the launch **export** block
  classified name by name, `scripts/env-knob-liveness.py`). The **map** direction of
  the same allocator —
  `torch.OutOfMemoryError: map:…NPUCachingAllocator.cpp:457 … aclrtMallocPhysical,
  error code is 507899` beside `[drv api]halMemCreate failed drvRetCode=17 …
  [FUNC:MallocPhysical]`, on an engine that has been serving for hours — is that
  page's §*The map direction*: there an allocation **did** fail, so the headroom
  arithmetic above comes first and `expandable_segments:False` is an ablation, not
  the cure.
- Start-up dies **at KV sizing, before any allocation fails**: `Available KV cache
  memory: … (halved for workspace)` then `ValueError: To serve at least one request
  with the model's max seq len …`, or `Available memory: 0` / `-1.33GB` on the
  **second** engine sharing one card, while `npu-smi` shows free HBM →
  `kv-budget-per-instance-vs-device-global.md`: device-global HBM charged to one
  instance, the `#7308` → `#7427` → `#7372`/`#14274` propagation chain, the
  per-instance `gpu_memory_utilization` contract, the identity that makes the fix
  one line of arithmetic, and the one-line patch. It is also the page for a paste
  where **`PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` was tried and did not
  help** (`#5707`, reporter-confirmed): that variable cannot move a device-global
  free/used reading, so do not re-route such a paste to the allocator page.
- The failure is an allocation error inside KV-cache initialisation, with the
  deepest frame in `_convert_torch_format` / `torch_npu.npu_format_cast` and
  `507013` / SDMA / SMMU text around it → `kv-init-and-format-cast-alloc.md`
  first. The format cast is the allocation site, not the defect; a 310P host adds
  the `FRACTAL_NZ` copy and the 300I allocator prescription
  (`../soc-platform-paths/atlas-300i-310p.md`).
- The traceback class is `torch.OutOfMemoryError` but the text is a synchronize /
  stream / timeout code (`AclrtSynchronizeStreamWithTimeout … 107020`,
  `107027`, `event wait timeout … task_timeout`) → `oom-class-is-not-a-cause.md`
  first: the class comes from torch_npu's `isCannOOM` predicate over a buffer that still
  holds an earlier `207001`, so the lead is the *earlier* allocation line, not the HBM
  budget.
- Any "memory leak" / "内存持续增长" report whose env dump carries
  `torch_npu==2.8.0` (or an unqualified `2.8.0`) → `host-ram-and-per-rank-rss.md`
  §"Step 0" first: the known pta leak, `2.8.0.post1`/`.post2` plus a vllm-ascend
  rebuild as the cheapest action, and the line that closes the lane.
- "The machine crashed" / ssh lost / a `top` paste whose python3 ranks each hold
  tens of GiB RES on one host, with no NPU-side error → `host-ram-and-per-rank-rss.md`
  and `bash <skill_root>/scripts/host-ram-check.sh <paste|->` (worked: #3902,
  16 ranks × 88-100 GiB = 1472 of 2014 GiB on a 2 TB host, swap 0).
- The host died with **nothing** to sample: the log stops at the per-rank
  `rank N in world size M is assigned ...` lines, there is no
  `Starting to load model`, and `--max-model-len` is 128K+ → the eager
  `max_model_len²` host attention mask, `attention-mask-host-allocation.md` and
  `bash <skill_root>/scripts/attn-mask-budget.sh --log <serve.log> --host-gib N`
  (#2873 reporter-confirmed: 262144 at TP=8, 0.10.1rc1, 0.10.0rc1 works).
- The report is a **sleep-mode** one (`enable_sleep_mode`, `llm.sleep`, "sleep A then
  start B"): `sleep()` returns only the pluggable allocator's `weights`/`kv_cache`
  blocks; the process, its HCCL communicators and the devices' host-side HCCP process
  survive, so the *new* instance dies creating a communicator (`hcclCommInitRootInfoConfig
  … error code is 7` + `EJ0001 … HCCP process`) → `sleep-mode-multi-instance.md` (verdict,
  the four greps, the disjoint-card discriminator, the vendor's own 10 s remedy).
- The report is about score/accuracy dropping under memory pressure, or about
  garbled output after long runs → `../accuracy-benchmarks/index.md`.
- The paste is a **container metric** (`container_memory_working_set_bytes`,
  k8s/cAdvisor dashboard, "容器内内存持续增加") on a service that never crashed,
  with or without `npu-smi` in the body → `container-working-set-growth.md` first
  (`scripts/working-set-arithmetic.sh` splits rss from cache and converts the
  slope into bytes/request). Contrast part 3: there the box is dead and `top` is
  the artifact.
