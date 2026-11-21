# `Event resources are insufficient` / `207007` / `rtEventRecord … no event resource` mid-serving

## Scope

Sibling code, different table: a `207005` refusal inside
`AclmdlRICaptureBegin` (`[SqCqManage]Alloc sq cq fail, stream_id=1984`) is the
**stream** table at startup capture, not this event table —
`../startup-failures/capture-stream-table-exhaustion.md`.

The engine started, served traffic, and then one worker dies on a runtime
**event-table** refusal — reported as a resource leak (泄漏) because it appears
after a while:

```text
[E..] operator():…/DivKernelNpuOpApi.cpp:190 NPU function error: call aclnnInplaceDiv failed, error code is 207007
[ERROR] … (PID:304, Device:0, RankID:-1) ERR00100 PTA call acl api failed
[Error]: Event resources are insufficient.
        Check whether the number of events meets the requirements by referring to the description of the acl.rt.create_event interface.
EE9999: Inner Error!
EE9999[PID: 304] … Event id alloc error, deviceId=6, tsId=0, retCode=0x7020019![FUNC:Record][FILE:event.cc][LINE:427]
       TraceBack (most recent call last):
      rtEventRecord execution failed, reason=driver error:no event resource[FUNC:FuncErrorReason][FILE:error_message_manage.cc][LINE:65]
      record event failed, runtime result = 207007[FUNC:ReportCallError][FILE:log_inner.cpp][LINE:148]
      Kernel Run failed. opType: 7, RealDiv
```

Recorded #13007: vllm-ascend v0.23.0rc1 (`f4a08bddd0cc`) + vllm 0.23.0, single
300I Duo (310P3) `TP=2` on devices 6/7, CANN 9.1.0-beta.1, Qwen3-14B fp16,
`--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [4,8,16]}'`,
`--max-num-seqs 16`, `--no-enable-prefix-caching`, under `AIS_BENCH` with 2 running
requests and `GPU KV cache usage: 11.9%`.

Not this lane:

- `Memory_Allocation_Failure(EL0004)` / `alloc device memory failed … 207001` /
  `NPU out of memory` → `../runtime-memory/index.md`. This page's code is not an
  HBM error at all, and the last throughput line in the paste usually proves HBM
  was not near its limit;
- `EI0007 … Failed to allocate resource[qp] … Memory resources are exhausted` →
  `qp-resource-exhausted-at-graph-load.md` (a *comm* resource refusal, different
  pool);
- `507018` + `libpt_kernels.so` / CPU-fallback text → `../soc-platform-paths/310p-cpu-fallback-op-fault.md`;
- `507013` + SMMU text → `../runtime-memory/kv-init-and-format-cast-alloc.md`;
- the same `Insufficient_Event_Resources(EL0008)` at **capture** time
  (`Create capture event failed, stream_id=[12->44], event_id=[-1->1018],
  error=117571609`, `[FUNC:GetCaptureEvent][FILE:api_impl.cc][LINE:1507]`) is the
  same pool consumed by a different phase — precedent #10521 (300I Duo, dev 26.0.0
  POC image), read it with this page and rank capture before the per-step churn.

## Read it in this order

1. **The code is the answer, the op is the messenger.** `207007` is
   `ACL_ERROR_RT_NO_EVENT_RESOURCE` (`cann/runtime/include/external/acl/error_codes/rt_error_codes.h:68`,
   doc row `cann/runtime/docs/api_docs/aclError.md:142`: “Event资源不足 … 检查Event数量”).
   `Kernel Run failed. opType: 7, RealDiv` / `current working operator name is
   aclnn<Op>` name the **last queued kernel** at the sync point, not the owner:
   in #13007 `RealDiv` is `probs.div_(q)`, one statement after the reported frame.
   So quote the enum name and say the run ran out of device events.
2. **The Python frame is one sync point off.** The reported deepest frame is
   `vllm_ascend/_310p/sample/sampler.py:89 q = torch.empty_like(probs).cpu()` — the
   blocking copy that *waits* for work already queued. The event request itself is
   the `torch.npu.current_stream().wait_stream(global_stream())` on `:92` and the
   stream change at `:88`. Do not attribute the refusal to the `.cpu()` line.
3. **What asks for events, per decode step.** `Stream.wait_stream` is
   `self.wait_event(stream.record_event())`, and `record_event(None)` allocates a
   fresh `Event` (`Ascend/pytorch` export `torch_npu/npu/streams.py:42-66`) — one
   new event object per pass, taken from the device table when recorded. Note the
   **generic** sampler has the same shape (`vllm_ascend/sample/sampler.py:32,41`,
   `npu_stream_switch` at `utils.py:1080`, `global_stream` at `:531`), so 310P is
   where the frame landed, not proof of a 310P-only defect.
4. **The table is per Device and finite.** ACL documents that an `Event` takes no
   resource until it is recorded, and that the per-Device maximum is fixed
   (`runtime/docs/api_docs/aclrtCreateEvent.md`, A2/A3 row: 65536; the release's
   product table omits 310P, so its cap is undocumented). The error's own
   `deviceId=6, tsId=0` says the refusal is **device-scoped**, and EL0008's canned
   `Solution: Close applications not in use.` says the same — a co-tenant process
   on that die competes for the pool.
5. **Two phases consume it.** Graph capture holds events for the life of the
   process (a `FULL_DECODE_ONLY` serve reuses them every replay), while decode
   churn requests more each step. #10521 exhausted the pool at capture with
   `event_id` running to 1018, so "the departure point" is usually the phase
   combination, not one leaking line.

## Answer rule

Deliver, in this order: not an HBM/host leak (quote the last
`GPU KV cache usage` line), the enum name and the pooled resource, then the check
that splits the two phases, then the reporter's next action. Cheapest first:

1. **`--enforce-eager` rerun of the same benchmark** (or
   `--compilation-config '{"cudagraph_mode": "NONE"}'`). Survives → capture
   reservations plus per-step events share one table, and trimming
   `cudagraph_capture_sizes` to the shapes actually served is the fix direction.
   Still dies → the per-step request rate (or a co-tenant) owns it, and go to 2/3.
2. **Sample the remaining events on the device.** `aclrtGetEventAvailNum(uint32_t*)`
   is exactly this counter (`runtime/include/external/acl/acl_rt.h:3311`, doc page
   `runtime/docs/api_docs/aclrtGetEventAvailNum.md`), so one `ctypes` probe against
   the image's `libascendcl.so` after `import torch_npu` (which brings the ACL
   runtime up in-process) reads it — sketch, since the library name/path is the
   image's: `python -c "import ctypes, torch_npu; L=ctypes.CDLL('libascendcl.so');
   n=ctypes.c_uint32(); L.aclrtGetEventAvailNum(ctypes.byref(n)); print(n.value)"`.
   Sample at start, after N decode steps, and at failure: a monotone fall to ~0 is
   the engine's own consumption; a low value from the first sample is a co-tenant.
   Pair it with `npu-smi info`'s process table for that `deviceId`.
3. **On 310P, take the host-copy route out for one run**: `VLLM_BATCH_INVARIANT=1`
   sends `AscendTopKTopPSampler310.forward_native` to the parent implementation
   (`_310p/sample/sampler.py:98-99` at v0.23.0rc1 — no stream switch, no `.cpu()`),
   which removes that per-step event request. Localization only: it changes
   numerics and throughput, so say so.
4. If the reporter attaches the op-level frame, ask for
   `ASCEND_LAUNCH_BLOCKING=1` **together with** graph mode off — vllm-ascend
   refuses blocking mode while `cudagraph_mode != NONE`
   (`index.md` rule 1) — and for the plog
   (`~/ascend/log/debug/plog/`) and the step/iteration count at failure. The
   208-line log in #13007 carried no plog, so the device-side owner stayed open.

## Do not claim

- that this is a host or HBM memory leak, whatever the title says — read the KV
  usage and the allocator lines first;
- **which** statement allocated the missing event. Dozens of
  `wait_stream`/`record_event` sites exist across the attention, MoE and sampler
  streams, and a driver-side id refusal cannot be attributed to a Python line from
  the paste;
- that `1018` (#10521) is a documented 310P limit — it is one recorded capture
  allocation on one image, and the ACL doc's own table row is A2/A3-only;
- that the ACL doc's "reaching the cap makes the system wait" describes this
  failure: the observed record path returns `no event resource` instead of waiting,
  which is itself worth reporting upstream;
- that eager mode is the fix rather than the discriminator.

## Recheck

`rg -n "NO_EVENT_RESOURCE" <cann>/runtime/include/external/acl/error_codes/rt_error_codes.h`
and the row in `runtime/docs/api_docs/aclError.md`; the per-Device maximum table in
`runtime/docs/api_docs/aclrtCreateEvent.md` (a 310P row there would replace the
"undocumented" caveat); `rg -n "def wait_stream" -A3 <Ascend/pytorch>/torch_npu/npu/streams.py`
(a pooled/cached event implementation would retire step 3's arithmetic); and
`rg -n "_random_sample_310p" -A12 <vllm-ascend>/vllm_ascend/_310p/sample/sampler.py`
for the line numbers, which move between releases.
