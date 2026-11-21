# 310P: an op with no device kernel — `507018` "aicpu execution is abnormal", `RunCpuKernel`, `<Op>Operation setup failed!`

## Signature

`/v1/models` answers, the first `/v1/chat/completions` returns HTTP 500, and the paste
carries an ACL code plus a **fault block naming an ordinary tensor op**:

```text
kernelName=IndexPut, soName=libpt_kernels.so, funcName=RunCpuKernel, errorCode=0x2a
[Error]: The aicpu execution is abnormal
RuntimeError: aclrtSynchronizeStreamWithTimeout(copy_stream), error code is 507018
File: gdn_linear_attn.py (forward_cuda / gdn_attention_core)
```

Recorded `#11188`: vllm-ascend **v0.21.0rc1** (`80610e4438db`),
`quay.io/ascend/vllm-ascend:v0.21.0rc1-310p`, Atlas 300I Duo 310P3 ×2 (TP2), Qwen3.6-35B-A3B
`w8a8` (hybrid GDN linear attention), `--enforce-eager --max-model-len 8192`, CANN
9.1.0-beta.1 in the container, host HDK 25.2.0 / firmware 7.7.0.1.231. The host dmesg at
the same second carries, once per device:

```text
[ascend][devmm][ERROR] Reserve addr is invalid, should call halMemMap
(hostpid=5073; devid=0; va=0x12c3c0200000; ret=-22)   (… devid=1 …)
```

The sibling surface is `<Op>Operation setup failed!` out of a `torch_npu._npu_*` op with
an op-plugin frame in the traceback (`ReshapeAndCacheOperation`, `SelfAttentionOperation`);
the op-plugin/ATB-owned variant has its own page,
`../spec-decode-mtp/atb-op-setup-failure-at-proposer.md`, because there the owner is a
different library. What both have in common, and what this page is about: **the failing
thing is one op, and the report can tell you where it was launched from.**

## Read the fault block, not the code number

1. `funcName=RunCpuKernel` + `soName=libpt_kernels.so` say the op ran in the **CPU/AICPU
   kernel library** — no AI-Core kernel was selected for it with this SoC / dtype /
   layout. That is a dispatch fact from the reporter's own log, and it reframes the
   question: *which Python statement launches that op*, not "is the card or the driver
   broken".
2. `507018` is what the **copy-stream sync** reports afterwards, exactly as `507013` +
   `sdma copy error` relates to an allocation failure on
   `../runtime-memory/kv-init-and-format-cast-alloc.md`. The loudest code is not the
   owner; the op name in the `[AIC_INFO] fault kernel_name=` block is the informative
   field.
3. `devmm … Reserve addr is invalid, should call halMemMap (ret=-22)` is the driver
   refusing to map a VA the fallback kernel passed — a consequence, printed once per
   `devid`. Quote the recorded ablations as the reason not to chase the driver lane: on
   `#11188` a motherboard replacement + BIOS 3101 + `pci=realloc`, a firmware upgrade to
   7.7.0.1.231 (`npu-smi` `Compatibility: OK`), and a clean single-path HDK 25.2.0
   reinstall with dkms boot fixed left the 507018 **byte-identical**. That is the
   behaviour this reading predicts.

## Map the op name to the launching statement

- `IndexPut` is `aten::index_put_`. It names indexed write syntax — `t[idx] = v`,
  `index_copy_`, `index_fill_`, `scatter`-style stores — never a vllm-ascend
  kernel. Grep the **assignment spelling**, not the word `index_put`; on a 310P report
  also check that the tree's own 310P substitutions were actually reached.
- The 310P lane knows this family is hostile, and says so in two places at this pin:
  `patch/worker/__init__.py:44-45` imports `patch_idex_310` **only** in the `else`
  branch of `if not is_310p():` (the file name is misspelled — `rg --files vllm_ascend |
  rg index` will not find it), and that file rebinds `fla.ops.index.prepare_chunk_indices`
  / `prepare_chunk_offsets` and carries "310P: skip NPU `index_fill_` when there are no
  discarded requests".
- **The decisive local check: the file whose docstring claims to avoid the op still uses
  it.** `_310p/ops/fla/gdn_310.py:215-216` — *"Avoid NPU `index_copy_` (IndexPutV2)
  which fails on some layouts; use direct indexing instead"* — and then `:227-228` writes
  `out[spec_token_indx] = spec_out` / `out[non_spec_token_indx] = non_spec_out`, while
  `:443` writes `ssm_state[non_spec_state_indices_tensor] = last_recurrent_state.to(…)`.
  Direct indexing **is** `index_put_`. So "this 310P path already avoids IndexPut" is a
  false negative one grep away: `rg -n '\[[^]]+\] *=' vllm_ascend/_310p/ vllm_ascend/patch/`.
- Keep the mapping a **hypothesis**: the plog names the op, never the line. The check
  that closes it is one re-run with `ASCEND_LAUNCH_BLOCKING=1` (legal here — the recorded
  config is already `enforce_eager=True`, so the ACL-graph refusal does not apply; see
  `../startup-failures/graph-mode-not-applied.md`) plus the plog
  `[AIC_INFO] fault kernel_name=` block from the same request.

## When the paste names two ops, and the model has no `_310p/ops/fla/` path

Recorded #11450: vllm-ascend v0.21.0rc1 (`80610e4`) + vllm v0.21.0 (`ad7125a4`),
`v0.21.0rc1-310p`, **PaddleOCR-VL-1.6** on a 300I Duo (`--max-model-len 1024 --dtype
float16 --enforce-eager --no-enable-prefix-caching`). `/v1/models` answers, the first
chat request kills EngineCore, and the paste carries **two** op names:

```text
08:53:33.404  AclNN_Runtime_Error(EZ9903): aclrtLaunchKernelWithHostArgs failed, return: 507018
              Aicpu kernel execute failed, device_id=0, stream_id=9, task_id=4507,
              soName=libpt_kernels.so, funcName=RunCpuKernel, kernelName=IndexPut, errorCode=0x2a
08:53:33.856  SubKernelNpuOpApi.cpp:57 NPU function error: call aclnnInplaceSub failed, error code is 507018
              KernelLaunch failed: …/ascend310p/ops_legacy/sub/Sub_…_high_performance.o
              Kernel Run failed. opType: 34, Sub / launch failed for Sub, errno:361001
```

Read them in **timestamp order**, not in the order the wrapper screams: the aicpu task
is the fault, and the later `aclnnInplaceSub` is the next launch on the poisoned stream
(`errno:361001`, and its `.o` is an ordinary AI-Core kernel). "Which op does
`aclnn<Op> failed` name" is not the question — `kernelName=` is. The op **name** in
the fault block is aten's, not a CANN/ATB op's (`libpt_kernels.so` / `RunCpuKernel` is
the CPU-dispatch library; its ownership is in *Limits* below), so `IndexPut` =
`aten::index_put_`, i.e. tensor indexed assignment (`t[idx] = v`, `index_copy_`,
`index_fill_`, `scatter`-style stores).

**The `_310p/ops/fla/` hits are unreachable on a VL/OCR checkpoint.** Every indexed
write this page lists lives in the GDN/FLA path, which PaddleOCR-VL, GLM-OCR and MinerU
never enter — move the grep to the generic path (`vllm_ascend/worker/`,
`_310p/model_runner_310p.py`'s mrope position building, the vision tower, `patch/**`)
instead of re-reading `gdn_310.py`.

**One candidate that looks right and is not: the sampler's Triton-absent penalty
fallback.** The first sampling step prints `[sample/sampler] Triton not available,
falling back to vLLM default penalty implementation` (`vllm_ascend/sample/sampler.py:53-58`,
a `warning_once`), and `_310p/sample/sampler.py`'s `AscendSampler310` does **not**
override `apply_penalties`, so upstream's torch body runs
(`vllm/model_executor/layers/utils.py:79-88` → `vllm/_custom_ops.py:472-485`, chosen
because `logits.is_cuda` is False). It explains the `aclnnInplaceSub` in the paste
exactly — `logits -= frequency_penalties… * output_bin_counts` at `:87-88` — but it
contains **no indexed write**: `get_token_bin_counts_and_mask` uses `scatter_add_`
and the repetition op is `where`/`div`/`mul`. It explains the Sub, not the `IndexPut`;
one grep of that body saves the turn, and it must not be delivered as the IndexPut
owner without the launch-blocking traceback. Two traps around the line: it is
`warning_once`, so it lands in the same second as *any* first-request fault; and the
`no_penalties` early return (`vllm/v1/sample/sampler.py:414`) makes the torch body
unreachable for a default-sampling request.

**Ladder for this shape, cheapest first:**

1. `ASCEND_LAUNCH_BLOCKING=1` with the same request — the statement, which is the only
   thing that names the indexed write.
2. The same request body with default sampling (`repetition_penalty` /
   `presence_penalty` / `frequency_penalty` removed) — kills the penalty lane above.
3. A text-only prompt on the same serve — splits the vision tower from the text path.
4. The pinned tutorial's own Atlas-300 tab as the config control: for PaddleOCR-VL
   `docs/source/tutorials/models/PaddleOCR-VL.md:51` ("supported from version
   0.15.0rc1") and `:93-115` (`--max_model_len 16384`, whose `:115` note says the option
   exists to avoid attention-mask errors; the reporter used 1024). Follow the tab, but
   say it does not explain this fault — the prompt was 168 tokens. Note the contrast
   with the Qwen3.6 row below: here the pinned docs *do* validate the model on Atlas
   300, so "unsupported" is not the answer.
5. The plog `[AIC_INFO] fault kernel_name=` block, `npu-smi info`, the full log.

## Then the capability verdict, in the same message

The reported model is outside the pin's validated set on three axes, and the same document
says so: `docs/source/tutorials/hardwares/310p.md:405`, the closing note —
*"For early access to Qwen3-MoE, Qwen3-VL, and **preview support for Qwen3.5 and Qwen3.6**
with performance acceleration, follow #7394"* — while the table at `:98-107` validates only
`Eco-Tech/Qwen3-{8B,14B,32B}-w8a8sc-310` (dense, 310-suffixed conversions). A generic
`-w8a8` 35B-A3B MoE on a GDN family is therefore *documented preview*, and `#7394` is
usable only for what the doc says about it. `atlas-300i-310p.md` holds the per-release
blocks and the triton-less hybrid trap; the quantised-checkpoint axis is
`310p-quant-checkpoint-soc-match.md`.

## Ranked actions for this shape

1. `ASCEND_LAUNCH_BLOCKING=1` on the same request, same flags → the deepest `vllm_ascend`
   frame names the statement that launched the op. Nothing else in this page is worth a
   turn before that frame exists.
2. The plog bundle for the same request: `[AIC_INFO] fault kernel_name=` block,
   `dmesg-ascend.txt`, `driver/firmware version.info`, `npu-smi info`. When the issue
   points at a bundle by local path (`reports/<model>-issue/<name>.tar.gz`), ask for the
   files, not the path — on `#11188` the reporter eventually posted exactly that tarball
   as an attachment, and it is the only artefact that can name the statement.
3. The doc's validated row on the same box (a `w8a8sc-310` dense model): one launch that
   separates *this model on this SoC* from *this host*, and cheaper than any driver
   theory after the ablations above.
4. Name the artifact axis (generic `-w8a8` vs a `-310`-suffixed conversion) before
   proposing any code change.
5. A workaround is a **code** substitution in `_310p/ops/fla/gdn_310.py` (rewrite the
   indexed writes as `index_select` + `torch.cat` / `scatter`, or pre-build the merged
   layout), not a flag: no env knob at this pin moves `index_put_` off the CPU fallback.
   Label it unverified, and only after 1-3.

## Limits

- Which of `:227-228` / `:443` faulted on `#11188` is **not** established: the pasted
  fault block names the op only, and the Python frames it quotes beside it
  (`gdn_linear_attn.py` `forward_cuda` / `gdn_attention_core`) carry no line number.
- `libpt_kernels.so` / `libcpu_kernels.so` belong to torch_npu/CANN and are not in this
  snapshot — one grep bounds the question, then stop (the sibling run spent three empty
  greps on an op-library message string).
- Whether the CPU dispatch is a deliberate "no 310P kernel for this pair" or a
  registration/packaging gap is not decidable from the tree; the probe is the same op in
  a single-device `torch_npu` snippet on that box.
- Line numbers and the `_310p/` layout are version-specific (`atlas-300i-310p.md` records
  how they moved from v0.11 to v0.22); re-grep at the reporter's pin.
