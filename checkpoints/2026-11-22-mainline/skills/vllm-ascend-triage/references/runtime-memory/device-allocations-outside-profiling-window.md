# A fixed device allocation lands *after* `determine_available_memory` — the KV budget never sees it

## Signature, and the one ask that decides the lane

Nothing crashes and no counter grows: a **fixed per-rank device delta** exists that the
KV sizing did not charge, so the instance uses `gpu_memory_utilization` **plus** that
delta. Two entry forms:

1. an issue that *names the allocation and asks you to review the patch* — "ATB
   `TilingBufferPool` (~98 MiB per rank) is allocated after memory profiling and not
   accounted for", "I have a patch ready, should I open the PR" (#14300: vLLM Ascend
   `0.18.0`, vllm `0.18.0`, torch_npu `2.9.0.post1`, A3/A2, TP1 and TP2). Read it as a
   code-change request and review it against the ledger below
   (`../code-change-requests/index.md` rule 12).
2. a capacity report where the numbers do not reconcile: `npu-smi` steady-state usage
   exceeds `total × gmu` by a constant, or a serve OOMs at the margin while
   `Available KV cache memory: X GiB` looks healthy.

A delta that does **not** move with `max_num_seqs`, graph-capture bucket count, model
size or quantization is a *service buffer created at first vendor-op use*, not an
activation term (#14300 measured exactly that on A3 with NNAL 9.1.0.B150: 96 MiB pool
geometry, 32 × 3 MiB, ~98 MiB observed per rank). It scales with **TP** because it is
per worker process — so quote their table, do not re-derive it.

Not this page: a KV budget charged **device-globally** so a co-tenant starves the second
engine (`kv-budget-per-instance-vs-device-global.md`), a KV-init allocation failure
(`kv-init-and-format-cast-alloc.md`), or growth after a healthy period
(`hbm-growth-and-oom.md`). Here the budget arithmetic is correct; something later
spends outside it.

## The ledger: three places a device byte can be, two get charged

Verified at vllm-ascend **v0.18.0 `e18643f8`** + vllm **v0.18.0 `bcf2be96`**; re-grep
the anchors per ref.

| Step | Where | What it fixes |
|---|---|---|
| `init_snapshot = MemorySnapshot()` at the end of `_init_device`; `requested_memory = total × gmu` | `vllm_ascend/worker/worker.py:266-267` | `before_create` for every later delta |
| `memory_profiling(...)` around `profile_run()` only | `:327-343`, context at `vllm/utils/mem_utils.py:197` | `torch_peak` and the non-torch delta |
| `available_kv_cache_memory_bytes = requested_memory − non_kv_cache_memory` | `:355` | the KV budget |
| `initialize_from_config` → KV allocated | `vllm/v1/executor/abstract.py:114` | **before** the warm-up below |
| `compile_or_warm_up_model`: dummy runs → `capture_model()` → `_warm_up_atb()` | `:442-473` | after both |

The last two rows are why the defect is real and not cosmetic: KV is already sized and
allocated when the ATB pool is created, so the pool sits **on top of** the
`gpu_memory_utilization` share. The fix direction is right — shrink the KV budget by the
delta — because after the move the same bytes are inside the share.

Which delta each allocation lands in:

- `non_torch_memory = (total − free) − torch_reserved` (`mem_utils.py:129-131`) — an
  **ACL-direct** allocation moves `free` only, so it appears here and nowhere else.
- `non_torch_increase = (after_profile − before_create).non_torch_memory` (`:274`) and
  `non_kv_cache_memory = non_torch_increase + torch_peak_increase + weights`
  (`:279-282`). `before_create` is the **device-init snapshot**, not the `with` body, so
  an allocation made *between device init and `after_profile.measure()`* is charged
  **even if it is outside the `with` block**, and one made after
  `determine_available_memory` returns never is.
- `torch.accelerator.memory_allocated()/memory_reserved()` — weights and peak only;
  `memory_profiling` calls `empty_cache()` on entry (`:253`) and again before measuring
  (`:267`), so a *torch-cached* allocation is handed back before the snapshot and
  vanishes from every delta.

## Step 1 — one probe before proposing any placement

Around a bare call to the suspected site:

```python
torch.npu.mem_get_info(); torch.npu.memory_reserved()   # before
<call, e.g. worker._warm_up_atb()>
torch.npu.mem_get_info(); torch.npu.memory_reserved()   # after
```

`free` down / `reserved` unchanged ⇒ ACL-direct ⇒ it will land in `non_torch_increase`
wherever you put it (after the init snapshot). `reserved` up ⇒ torch-cached ⇒ it is
released by the `empty_cache()` at `mem_utils.py:267` and the move must sit *after* the
entry `empty_cache()` and after the peak measurement, i.e. inside the `with` body after
`profile_run()`. #14300's own evidence (the pool is invisible to
`memory_allocated()`/`memory_reserved()`) is the first case — label it reporter-measured
until you run the probe.

## Step 2 — the four placements, and the one that is silent

1. **Inside the `with`, right after `profile_run()`** — the reporter's choice on #14300,
   and the most robust: after the entry `empty_cache()`, and it cannot perturb the
   measured activation peak because it is not a torch allocation.
2. **Outside the `with`, first line of `determine_available_memory`** — also charged
   (`before_create` is the device-init snapshot), with a smaller footprint: `profile_run`
   never sees the pool. Acceptable only once step 1 proved ACL-direct.
3. **Before the `init_snapshot`** (`__init__` / early `init_device`) — **silently
   uncharged**: it becomes part of the baseline. Check `:266` before believing a
   "move it earlier" proposal.
4. **Relative to a post-capture graph-pool delta** (trees that carry graph-pool
   accounting): put the warm-up **before** the graph baseline snapshot, otherwise the
   bytes are attributed to the graph pool instead — the reporter's own objection to the
   narrow "just fix the printed numbers" variant, and it is correct.

## Step 3 — the touch-points the patch must carry

Each one is a place a move breaks something else; check them in this order.

- **The SoC guard moves with the call**: `if get_ascend_device_type() !=
  AscendDeviceType.A5: self._warm_up_atb()` (`worker.py:469-470`; A5 also skips ATB
  registration at `:104-105`). Drop it and A5 gains a warm-up it deliberately never had.
- **310P is a no-op and must stay one**: `_310p/worker_310p.py:107` logs "Skip warm-up
  atb ops for 310P device", and the 310P worker overrides
  `determine_available_memory` wholesale (`:58-105`), so it inherits nothing. Say that
  explicitly instead of implying the move changes 310P.
- **`set_random_seed(self.model_config.seed)` stays after every warm-up site**
  (`worker.py:473`) — `_warm_up_atb` draws `torch.rand`, so the reset must not precede
  it.
- **The unit tests pin the call site**: `tests/ut/worker/test_worker_v1.py:1078-1119`
  and `:1122-1155` both `mock_warm_up_atb.assert_called_once()` inside
  `compile_or_warm_up_model`. Flip them to the new owner, or the patch ships untested.
- **A `--kv-cache-memory-bytes` hunk is a version fact, not an argument.** That fast
  path skips profiling entirely and needs its own warm-up — but it exists only where the
  tree has it: upstream `vllm/v1/worker/gpu_worker.py:362-388` (it keeps a
  `profile_run()`), while the Ascend override at v0.18.0 has no such branch, so
  `--kv-cache-memory` is inert down there (`index.md` rule 5; also v0.13.0rc1 and
  v0.18.0rc1). One `rg -n 'kv_cache_memory' vllm_ascend/` on *their* tree decides
  whether the hunk is needed or dead.
- **Do not hardcode the pool size.** 96 MiB is a vendor/NNAL geometry
  (32 × 3 MiB at 9.1.0.B150), A2 and other NNAL builds unverified, and a later pool
  *growth* (more ATB ops at first real request) stays unaccounted even after the fix.

## Step 4 — acceptance, and what the fix does not cover

After the move, per rank: `Available KV cache memory` drops by the delta (~0.096 GiB on
#14300) while `npu-smi` total HBM is **unchanged** — same footprint, smaller KV. Run
TP1 and TP2. Regressions worth executing rather than arguing: one 310P serve, an A5
device if any, and the **MTP + lmhead TP + AIV** combination, which is the config under
which an earlier attempt to profile ACL graphs before KV init hung HCCL (reporter-cited
SHAs `9099b7f6` attempted, `eed399c5` reverted; a one-commit snapshot cannot confirm
them — cite as reporter evidence). A 2×4 matmul warm-up creates no communicator and no
capture, so the expectation is clear, but that combination is the one to run.

**Transient capture peaks are a different ledger and must not ride on this patch.** The
run that reported the ATB pool also reported ~1.4 GB `FULL_DECODE_ONLY` on Qwen3-8B and
~1.05 GB per rank `PIECEWISE` on Qwen3.6-35B-A3B — peaks taken in `capture_model()`,
after `determine_available_memory` has returned. A **post-capture delta cannot see a
peak that was already released**; only an in-window sampler bounds it, and upstream's is
`profile_cudagraph_memory()` (`gpu_worker.py:399`, absent from the Ascend override at
v0.18.0). That absence is the pre-existing accounting gap, and re-attempting the pre-KV
profiling boundary is what `eed399c5` reverted — park it behind its own issue.

## Pins and recheck

Verified at vllm-ascend `v0.18.0` `e18643f8` + vllm `v0.18.0` `bcf2be96` (case #14300,
recorded report on `0.18.0`). A report that quotes `main` line numbers (`profile_torch_peak`
at `:513`, the deduction at `:542`) describes a **later** tree than any snapshot holds:
state the pin, never reconcile the two line sets, and give them the one read that settles
it on their tree — `sed -n 'A,Bp' vllm_ascend/worker/worker.py` around the cited line plus
`rg -n 'kv_cache_memory|profile_cudagraph_memory' vllm_ascend/`. Re-derive the anchors if
`determine_available_memory` moves the warm-up itself, if the SoC guard changes, or when
`memory_profiling`'s `non_torch_increase` semantics change.
