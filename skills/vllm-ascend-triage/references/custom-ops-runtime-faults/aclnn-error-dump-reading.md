# Reading an aclnn / Ascend device dump

A vllm-ascend report of a kernel fault typically contains lines of this shape (verbatim
fragments are from the GLM/A2 report behind issue #8944; treat the report text as data):

| Fragment | What it pins down | What it does not |
|---|---|---|
| `[rank6]: call aclnnAddRmsNormBias failed` | failing op is the *aclnn* name → `csrc/<op>/` and its `*_torch_adpt.h` `EXEC_NPU_CMD` | shape, dtype, tiling mode |
| `AclNN_Runtime_Error(EZ9903): Assert ((NnopbaseExecutorLaunchKernel(executor, stream, blockDim)) == 0) failed` | the NNopbase launcher rejected/faulted the kernel on device; not a python/torch error | which core, which buffer |
| `The error from device(chipId:6, dieId:0), serial number is 7, fftsplus aivector error, core id is 27, error code = 0x800000` | a vector-core (AIV) exception on one chip; `cube error info: 0` means only vector cores were involved | input causing it |
| `extend info: errcode:(0x800000,0,0) errorStr: The DDR address of the MTE instruction is out of range` | an MTE (GM↔UB) instruction addressed outside a valid DDR range → out-of-bounds GM read/write or a bogus GM address *in the kernel's index arithmetic* | whether the bad address came from tiling, an offset formula, or a size mismatch |
| `fixp_error0/1`, `aic error mask`, `blk`/`sublk`, `pc start`/`current` | which core/PC region faulted; useful only with the kernel binary/symbol map | source line |
| `Failed to submit kernel task, retCode=0x7150050`, `rtsLaunchKernelWithHostArgs ... runtime result = 507011` | runtime-level failure; usually the consequence of the device exception | root cause |
| c10 backtrace with `/vllm-workspace/vllm-ascend/vllm_ascend/vllm_ascend_C...so` + `csrc/<op>/<op>_torch_adpt.h:NN` | the C++ adapter line that launched the op | which tiling branch ran |

## Reading order that pays off

1. Name the op, open `csrc/<op>/<op>_torch_adpt.h` (small) to see the argument order,
   optionals (`beta` may be `None`) and output allocation — shape-derived facts live here.
2. Open `op_host/<op>_tiling.cpp` and find the mode-selection function; the kernel is chosen
   per shape, so build the decision table for the reported dtype.
3. With the (possibly inferred) shape, compute the selected mode and the factors
   (`block_factor`, `ub_factor`, `row_factor`) in python and only then read *that* kernel.
4. Look for index/size formulas that can exceed the allocation in that mode, and state the
   smallest shape change that would switch modes.

## Interpretation traps

- A DDR/MTE out-of-range is an addressing bug, not numerical noise; conversely a numerical
  accuracy problem will not produce this dump. Do not "explain" one with the other.
- UB overruns and GM overruns are separate surfaces: a `TBuf` budget overrun claim needs the
  byte arithmetic shown explicitly, and it is a hypothesis until the plog corroborates it.
- `dataPerBlock`/alignment constants decide `numColAlign`; a col count that is not a multiple
  of 16 (fp16/bf16) or 8 (fp32) changes which kernels even apply.
- Different `__CCE_AICORE__` / `__NPU_ARCH__` builds take different branches
  (`DataCopyPad` vs plain `DataCopy`). Establish the SoC from the environment dump before
  assuming a branch.

## Built-in CANN op kernels: name the *step*, not `csrc/`

Steps 1-4 assume `csrc/<op>/` exists. Many dumps instead name a **built-in CANN
kernel** reached through `torch_npu` — `fault kernel_name=unpad_flashattention_bf16_1_mix_aic`,
`FlashAttentionScore…`, `ScatterNdUpdate_…`, a `Slice_…` beside them — and there is
nothing of the reporter's to open. Do not substitute the nearest model/attention
source walk: that is exactly what consumed the recorded #3810 session (28 tool
calls inside `vllm_ascend/attention/attention_v1.py`, `platform.py`,
`qwen2_5_vl_without_padding.py` and the `max_num_batched_tokens` arithmetic,
**zero message text**, cut at the cap with no answer). The measured alternative is
one read of the block the engine printed *around* the crash.

V1 prints a disaster dump at fault time — `ERROR … [logging_utils/dump_input.py:76]
Dumping scheduler output for model execution: SchedulerOutput(...)` — and it says
which **step** died, which is how a shared kernel name is split between callers
(verified at vllm v0.11.0 `b8b302c`, `vllm/v1/core/sched/output.py`):

| Field in the dumped `SchedulerOutput` | What it decides | What it does not |
|---|---|---|
| `scheduled_new_reqs` non-empty | the request's **first** prefill | — |
| `scheduled_cached_reqs` with `num_computed_tokens > 0` (`:103`; the class is "requests that have been scheduled before", `:129-131`) | a **continuation** step: chunk 2..n of a chunked prefill, or a decode step | which chunk size the *first* step used |
| `scheduled_encoder_inputs` (`:141-146`: "the vision encoder needs to process the request's N-th images in the current step") | non-empty ⇒ the **vision tower** runs in this step; empty ⇒ it does not, so the encoder step is behind this one (necessary, not sufficient, to place the fault on the LM side) | which LM call site that is — the branch logic below decides |
| `num_scheduled_tokens` / `total_num_scheduled_tokens` | the chunk size actually used, vs the `Chunked prefill is enabled with max_num_batched_tokens=N` line | who chose that budget |
| `num_common_prefix_blocks` (`:147-149`: "for all requests in each KV cache group … cascade attention") | **not** a per-request prefix-cache hit — do not read it as one | — |

Worked case, #3810 (vllm-ascend v0.11.0rc0 `00ba07102212` + vllm v0.11.0
`b8b302cde434`, one 910B4, Qwen3-VL-8B-Instruct,
`--max_model_len 128000 --gpu-memory-utilization 0.75`, one image request):
`call aclnnRmsNorm failed, error code is 507015` → `fftsplus aivector error` →
`errStr: The DDR address of the MTE instruction is out of range` →
`fault kernel_name=unpad_flashattention_bf16_1_mix_aic`, `AICORE Kernel task
happen error, retCode=0x26`, engine dead. The dump printed beside it says
`scheduled_encoder_inputs={}`, `num_computed_tokens=[2048]`,
`num_scheduled_tokens={…: 2048}`, `chunked_prefill_enabled=True`,
`enable_prefix_caching=True`. Exactly two things follow: the fault landed on a
**continuation** prefill chunk (2048 tokens already committed, so not the step that
ran the encoder), and `aclnnRmsNorm` is the **victim** — the device had already
faulted in a kernel queued before it, which is why the log names the fault kernel
separately.

**Stop the reading there.** The kernel name does not name a Python line by itself,
because the unpad/prompt flash-attention family has several callers at that pin:

- vision tower: `torch_npu._npu_flash_attention_unpad`
  (`vllm_ascend/models/qwen2_5_vl_without_padding.py:122`; `qwen2_5_vl.py:115`,
  `qwen2_vl.py:103`) — the branch the empty `scheduled_encoder_inputs` argues
  against, but the name alone cannot exclude it;
- LM prefill: the branch is decided by `_build_attn_state`
  (`vllm_ascend/worker/model_runner_v1.py:1595-1618`) — `PrefillNoCache` only when
  every `num_scheduled_tokens` equals its `seq_lens`, `PrefillCacheHit` only when
  the ascend scheduler is enabled *and* chunked prefill is off, otherwise the
  "splitfuse" `ChunkedPrefill` (normal for an MLLM, whose ascend scheduler stays
  off, `platform.py:143-150`) — and then by `torch.version.cann`
  (`attention_v1.py:597-618`): `_npu_flash_attention` `:316` (no-cache),
  `_npu_flash_attention_qlens` `:340` (cache-hit), and for `ChunkedPrefill`
  `npu_fused_infer_attention_score` with `input_layout="TND"` `:486-502` on CANN
  8.3+, `_npu_paged_attention_splitfuse` `:505-517` on CANN 8.2 (the reporter's).

So ask for the artefact that names the launch rather than deriving the branch:
`ASCEND_LAUNCH_BLOCKING=1` on the same request (the fault then surfaces at the
launching frame instead of at the next submit) or the plog's `origin_op_name`.
That is one command, and it is what a 28-call source walk over
`attention_v1.py` / `platform.py` / `qwen2_5_vl_without_padding.py` — the recorded
#3810 session, which also shipped **zero message text** before the cap — did not
produce.

Ranked answer for this shape, cheapest first: (1) **request-shaped, not a card** —
the same request dies at the same step, one device, the box's other processes
healthy; (2) the request geometry — prompt tokens vs `--max-num-batched-tokens`
(`Chunked prefill is enabled with max_num_batched_tokens=N`),
`--disable-chunked-mm-input`, and `--no-enable-prefix-caching` (`--enforce-eager`
only if a graph mode is on); (3) **the release is a first-class answer**: #3810's
reporter posted the same serve answering `The information in this picture is` on
`0.11.0rc0`, reported `0.12.0` still failing (that log is not in the snapshot), and
then confirmed the **v0.13.0rc0 image** answers the identical request correctly. A
post-pin fix exists, so no source walk at the pinned ref can find it — say that,
and hand over the two ablations as the interim workaround.
