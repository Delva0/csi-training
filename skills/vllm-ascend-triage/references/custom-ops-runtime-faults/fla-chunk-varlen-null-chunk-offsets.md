# FLA/GDN chunked prefill: `507015` MPU fault in `chunk_fwd_o` with `chunk_indices=None`

Verified at vllm-ascend **v0.20.2rc1** (`367b8e62`) + vllm v0.20.2 (`bc150f50`);
recorded as #10572, filed as a **test-validity** report rather than a serving crash.

## Symptom shape

`RuntimeError: Sync:…AclrtSynchronizeStreamWithTimeout(stream), error code is 507015`
→ `The aicore execution is abnormal` → `errorStr: The MPU address access is invalid`,
with the deepest frames `vllm_ascend/ops/triton/fla/chunk.py:… chunk_gated_delta_rule_fwd`
→ `torch.ops._C_ascend.chunk_fwd_o(...)`. The paste is often a nightly op test
(`tests/e2e/nightly/single_node/ops/singlecard_ops/**`) that reports **PASSED** while
the device is left faulted; `ASCEND_LAUNCH_BLOCKING=1` turns it into a hard failure.
Grade first: if the frames are `fla/chunk.py` and the kwargs show `cu_seqlens`, this
page; if the op is `aclnnMoeInitRouting*` go to `moe-init-routing-tiling.md`.

## Two candidate mechanisms — only the second faults

1. **Varlen geometry mismatch** — `cu_seqlens[-1] != q.shape[1]`. The scheduler takes
   `tokenBatch = dim(cu_seqlens)-1` (`chunk_fwd_o_tiling.cpp:70-78`) and interleaves
   per-chunk offsets derived from `cu_seqlens` with strides derived from `seqlen`
   (`block_scheduler_gdn_fwd_o.hpp:~150-170`). Bad geometry gives *wrong numbers* and
   offsets that are usually still inside the allocation. It alone does not explain an
   invalid-address fault.
2. **Null `chunk_offsets` GM buffer** — the faulting read. `chunk_fwd_o`'s optional
   `chunk_indices` argument reaches the kernel as `chunk_offsets`; when the caller
   never passed it the buffer is `nullptr`, and the scheduler reads it
   unconditionally whenever `isVariedLen`:
   `block_scheduler_gdn_fwd_o.hpp:150-151` (`gmChunkOffsets.GetValue(2*chunkIdx)` /
   `2*chunkIdx+1`) and `:168` (`2*chunkIdx+3`, short-circuited by
   `chunkIdx == numChunks-1`). The chain at this pin:
   `chunk.py:51-55` (`prebuilt_meta` is where the indices come from) → `:95`
   `chunk_indices = None if prebuilt_meta is None` → `:164-176` passes
   `chunk_indices=None` with `cu_seqlens` set → `csrc/torch_binding.cpp:2245-2270`
   forwards the optional unchanged → `op_host/op_api/chunk_fwd_o.cpp:46-53` turns an
   absent optional into `actualChunkOffsets = nullptr` (and
   `aclnn_chunk_fwd_o.cpp`'s `CheckNotNull` deliberately does not require it) → the
   scheduler reads address 0.

## The decider: the sibling op in the same call

`chunk_gated_delta_rule_fwd_h` runs immediately before with the **same**
`chunk_indices=None` and succeeds (the reporter's `h`/`v_new` exist), because its
scheduler builds its own chunk-count workspace from `cu_seqlens`
(`block_scheduler_gdn_fwd_h.hpp:145-165` into `user + numChunksWorkspaceOffset`) and
never dereferences the optional. So **fault in `chunk_fwd_o` only, with the same null
argument on both calls, means op-specific null-optional handling**, not the caller's
geometry. If both ops faulted, geometry or the caller is back in front. One grep of
`gmChunkOffsets` / `gmNumChunks` in each scheduler settles it.

## What the Python gate checks — and the gap

`chunk.py:322-333` validates only `q.shape[0] == 1` and
`len(initial_state) == len(cu_seqlens)-1`. With `cu_seqlens=[0,1,2,3]` and
`initial_state` of 3 sequences both **pass**; there is no `cu_seqlens[-1] == q.shape[1]`
and no monotonicity check anywhere in the path.

## Fix ladder — and why aligning `cu_seqlens` is not enough

1. **Give the op real chunk indices, or drop varlen.** `prebuilt_meta` answers six
   attributes (`chunk.py:50-55`: `block_indices_cumsum`, `chunk_indices_chunk64`,
   `chunk_offsets_chunk64`, `update_chunk_offsets_chunk64`,
   `final_chunk_indices_chunk64`, `chunk_indices_large_block`).
   `prepare_chunk_indices(cu_seqlens, 64)` (`vllm_ascend/ops/triton/fla/utils.py:22-24`)
   already returns exactly the per-chunk `(seq_idx, chunk_in_seq)` pair the scheduler
   reads, for every chunk (`numChunks = Σ ceil(len/64)`), so its flattened form is the
   `2*numChunks` int64 list the op wants; `prepare_chunk_offsets` (`:32-34`) is the
   `N+1` cumulative-count array the sibling varlen paths take instead.
   **Do not rank "make `cu_seqlens[-1] == T`" as the fix while `prebuilt_meta` is
   still `None`:** with `cu_seqlens` set the scheduler still reads the null buffer, so
   the same `507015` survives the alignment. Check the proposed cheapest rung against
   the line the fault passes through.
2. **Python gate, no rebuild** (`chunk.py` inside the `if cu_seqlens is not None:`
   block): reject `cu_seqlens[-1] != q.shape[1]` and non-monotonic `cu_seqlens`, or
   synthesise the indices from `cu_seqlens` when `prebuilt_meta is None`. This is the
   part worth upstreaming regardless of the test.
3. **Op tiling hardening, needs a custom-op rebuild:** `INPUT_CHUNK_INDICES_IDX` is
   declared (`chunk_fwd_o_tiling.cpp:27`) but tiling fetches only `INPUT_SEQLENS_IDX`
   (`:70`), so `isVariedLen && chunk_offsets == nullptr` can be rejected with
   `OP_LOGE`. Value-level checks (`cu_seqlens[-1] == seqlen`) cannot move into tiling —
   tiling sees device values only as declared shapes.
   Scope: serving is not affected at this ref — `vllm_ascend/ops/gdn.py` always passes
   `prebuilt_meta` (`get_non_spec_chunked_prefill_meta`, `:238-240`, raises when the
   fallback meta is missing), so the null-optional path is reachable through the
   public `chunk_gated_delta_rule` API (tests, future callers) only. Say so, and say
   it is version-scoped.

## Why the test "passes": shape-only asserts plus async execution

`test_chunk_gated_delta_rule.py:22-23,42,47-48` builds `T=17`, `cu_seqlens=[0,1,2,3]`
and asserts only `.shape`. Output shapes are derived on the host from the inputs, so
they hold whether the kernel computed anything or aborted; the fault surfaces at the
next device sync (a warning at exit) or as a hard failure under
`ASCEND_LAUNCH_BLOCKING=1`. A nightly op test that asserts shapes only is therefore
**not** coverage of the op — say that in the answer, and treat
`torch.range(0,3,dtype=torch.int)` as deprecated noise, not the bug.

**Numerical oracle.** `chunk_gated_delta_rule_pytorch`
(`vllm_ascend/_310p/ops/fla/chunk_gated_delta_rule.py:183`) is a pure-torch chunked
implementation that accepts `cu_seqlens`, and the test file already imports it at
line 4. It cannot be *run* in the review workspace (stdlib-only python, no torch — see
`../evidence-workspace/index.md`), so propose the assert and let the reporter's NPU
produce the numbers; a tolerance you did not measure is a suggestion, not a result. Trap: that reference takes and returns the recurrent state as `[N, H, V, K]`
(its own UT names the trailing dims `vdim, kdim`, `tests/ut/_310p/ops/test_chunk_gated_delta_rule_310.py:21`),
while `chunk.py:254-259` documents `[N, H, K, V]` and production transposes at
`gdn.py:632` — with `K == V == 128` a shape assert cannot see the difference, so
`transpose(-1,-2)` on the way in and back on the way out, and calibrate bf16
tolerances on the existing UT (`~2e-2`, `test_chunk_gated_delta_rule_310.py:123,126`).

## Recheck triggers

A later ref that synthesises `chunk_indices` from `cu_seqlens` when `prebuilt_meta is
None`, or whose `chunk_fwd_o` tiling rejects a null offsets array, invalidates rungs
1-3 above; re-grep `chunk.py:95` and `chunk_fwd_o_tiling.cpp` before citing them.
