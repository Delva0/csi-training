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
instead of re-reading `gdn_310.py`. Then go **upstream of the plugin**: on an image request the
first indexed write the generic forward makes is vLLM's own `_merge_multimodal_embeddings`
(next section), which `vllm_ascend/` never overrides.

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

## A VL model on 310P: the owner is upstream's multimodal-merge write

Recorded **#12086**: image `quay.io/ascend/vllm-ascend:v0.22.1rc1-310p` (pin vllm-ascend
**v0.22.1rc1** `5f6faa0` ↔ vllm **v0.22.1** `0decac0`), 2×310P1, `Qwen3-VL-4B-Instruct-
w8a8s-310` and `Qwen3-VL-8B`, both `--tensor-parallel-size 2 --dtype float16
--max-model-len 8192 --enforce-eager --no-enable-prefix-caching`. The serve comes up, a
**text-only** chat answers normally, and every request that carries an image kills
EngineCore. The two pastes look unrelated — one headline is `QuantBatchMatmulV3 … x1
format should be ND`, the other `kernelName=IndexPut` — and they are **one fatal site, in
upstream vLLM, not in `vllm_ascend/`**:

```text
File "/vllm-workspace/vllm/vllm/model_executor/models/utils.py", line 477, in _merge_multimodal_embeddings
    inputs_embeds[is_multimodal] = mm_embeds_flat.to(dtype=input_dtype)
File ".../utils.py", line 490, in _merge_multimodal_embeddings
    raise ValueError("Error during index put operation") from e
```

`inputs_embeds[is_multimodal] = …` is a **boolean-mask** indexed write, i.e.
`aten::index_put_` — the same failure family as the rest of this page — but the statement is
upstream (`_merge_multimodal_embeddings`, `utils.py:456`, write `:477` at vllm v0.22.1)
and `vllm_ascend/` carries **no** substitution for it (`rg -n '_merge_multimodal_embeddings'
vllm_ascend/` → release-note prose only), so no plugin patch and no env knob moves it off
the CPU/AICPU library. Only image requests reach the write: a text-only prompt returns
before the merge, which is exactly why the reporter's text test passed. Consequences:

- **Grep the upstream tree before the plugin's "generic path".** One command at any pin
  (`rg -n 'inputs_embeds\[is_multimodal\]|_merge_multimodal_embeddings'
  <vllm>/vllm/model_executor/models/utils.py`). The `except RuntimeError` four lines below
  re-raises `ValueError("Error during index put operation")` (`:490`), or
  `Attempted to assign <expr> = N multimodal tokens to M placeholders` (`:485`) when the
  counts differ. So that `ValueError` beside an aicpu / `507018` block **is this fallback,
  not a placeholder-count bug** — read the frames above it before routing the case to
  `../multimodal-inputs/index.md`.
- **When the paste already carries upstream frames with line numbers, the statement is
  named** and rung 1 of *Ranked actions* is moot. Here the 8B log names it on both ranks
  (`utils.py:477` → `:490` → `embed_input_ids`) beside the same-second
  `Aicpu kernel execute failed … soName=libpt_kernels.so, funcName=RunCpuKernel,
  kernelName=IndexPut, errorCode=0x2a`.
- **The line numbers move with the pin, and the `-310p` image is often a 0.21 one.**
  At vllm **v0.21.0** (`ad7125a4`, upstream behind `vllm-ascend v0.21.0rc1`
  `80610e4` and its `v0.21.0rc1-310p` image) the def is `utils.py:458`, the write
  `:479`, the re-raise `:492` (verified for #11450); grep the expression before
  quoting a number borrowed from v0.22.1. Recorded **#11450** is this same write on a
  **PaddleOCR-VL-1.6** serve, one 300I Duo die, `--max-model-len 1024 --dtype float16
  --enforce-eager`: `/v1/models` 200, then the first chat request kills EngineCore —
  `aclrtLaunchKernelWithHostArgs failed, return: 507018` stamped `08:53:33.404`
  wrapping `Aicpu kernel execute failed … soName=libpt_kernels.so,
  funcName=RunCpuKernel, kernelName=IndexPut, errorCode=0x2a` (device 0, stream 9,
  task 4507), and at `.856` the *next* launch `aclnnInplaceSub failed` +
  `launch failed for Sub, errno:361001` on the poisoned stream. The same-second
  `Triton not available, falling back to vLLM default penalty implementation` warning
  is not the owner — that body has no indexed write (§*two ops* above). Its support
  verdict is not here either: `310p-vl-ocr-garbage-output.md` §1 holds the
  PaddleOCR-VL pair (tutorial Atlas 300 from 0.15.0rc1 with an own
  `--max_model_len 16384` tab against `supported_models.md:107` `🔵 A2/A3`), and a
  matrix column is not the refusal.

### The quantised model's `QuantBatchMatmulV3` block is a start-phase latch

The 4B-w8a8s half of #12086 leads with
`E69999 op[QuantBatchMatmulV3], Input x1 format should be ND, actual format is FRACTAL_NZ
… [FUNC:CheckDtypeOnOnlyL0c2ub][FILE:quant_batch_matmul_v3_tiling.cpp][LINE:173]`, and the
report reads it as that model's own failure. Two tells say otherwise, and both are in the
paste:

1. **That log has zero `kernelName=` lines** (`grep -c 'kernelName=' 4b-.txt` → 0), so no op
   is attributed to the failing request at all — while the second log has two.
2. **The block's own stamp is from the init window.** `2026-07-15-06:38:12` sits between
   `Encoder cache will be initialized with a budget of 16384 tokens, and profiled with 1
   image items …` (06:36:56) and `init engine (profile, create kv cache, warmup model) took
   92.46 s` (06:38:20), i.e. inside the encoder `profile_run`; it is *printed* at 06:39:27
   by `CopyKernelOpApi.cpp:58 Warning: E69999`, beside a request fault whose traceback is
   the same merge line. Read `[W… <now> …] Warning: E<code>` against the block's own
   `time=` before calling the block the request's owner; the device-side form of the same
   masking is in `../hccl-and-hardware/latched-runtime-error-masking.md`.

The int8→NZ half is real but is not that request's owner, and it is not switchable off:
`_should_trans_nz` returns `True` at `if is_310p():` (`vllm_ascend/utils.py:219-221` at
v0.22.1rc1) **before** it reads `config.weight_nz_mode`, so `VLLM_ASCEND_ENABLE_NZ=0` cannot
stop the int8→FRACTAL_NZ cast on 310P. Grep `is_310p()` in that function before offering the
env lever.

So the answer to "why do the two models differ" is: one fatal site, and the difference is
only **which op the log names** — the fp16 model's fault block names `IndexPut`, the
quantised one names nothing and re-prints a profile-stage tiling refusal.

### Capability verdict and the sizing question

At v0.22.1rc1 `docs/source/user_guide/support_matrix/supported_models.md:87` lists
`Qwen3-VL` with **A2/A3** in the `Supported Hardware` column,
`docs/source/tutorials/hardwares/310p.md:405` is the closing early-access note (Qwen3-VL →
#7394), the only validated 310P quant rows are the three dense `Eco-Tech/Qwen3-{8B,14B,32B}-w8a8sc-310`
dense conversions, and `docs/source/tutorials/models/Qwen-VL-Dense.md` has **no** 300I/310P
section (`rg -n -i '300i|310p|duo'` → 0 hits). So a "which Qwen3-VL size fits 2×310P1"
question has **no validated row at this pin**, and memory is not the binding constraint —
the 8B log reports `Loading model weights took 9.6286 GB` per rank at TP2 and 13.4 GiB KV
against 43 GiB/die, so one die holds it. Say that in one sentence, name the operator
coverage as the limit, and do not invent a supported size.

The only workaround is a **code** substitution in *upstream* vLLM, applied inside the image
(rewrite the merge without a bool-mask write: `index_select` + `copy_`, `scatter`, or
pre-built indices). Label it unverified; nothing in `_310p/` is involved.

## The GDN family on 310P: prefill-only indexed writes, and the CI row that decides the verdict

Recorded #13449: vllm-ascend **v0.20.2rc1** (`367b8e62da79`, image
`v0.20.2rc1-310p-openeuler`), **one** Atlas 300I Duo 310P1 die (44 GB), CANN 9.1.0, driver
25.5.t7, Qwen3.5 **0.8B and 9B** (hybrid GDN), `--dtype float16 --max-model-len 4096
--enable-chunked-prefill --enable-prefix-caching --mamba-cache-mode align
--mamba-ssm-cache-dtype float16 --max-num-seqs 8 --gpu-memory-utilization 0.6`,
`FULL_DECODE_ONLY` with capture sizes `[1,4]`. Weights load, "Graph capturing finished in
1 secs", KV cache sized, `/v1/models` is fine — the **first** chat request kills
EngineCore with the same three lines as #11188 (`507018`, `RunCpuKernel`,
`kernelName=IndexPut`) plus a frame that names the reporting point:

```text
RuntimeError: operator(): ../torch_npu/csrc/aten/common/LocalScalarDenseNpu.cpp:23
  NPU function error: AclrtSynchronizeStreamWithTimeout(copy_stream), error code is 507018
```

- **`LocalScalarDense` on `copy_stream` is a `.item()` on a device scalar, and at this pin
  the 310P path has exactly two of them:** `_310p/ops/causal_conv1d.py:137`
  `int(cache_indices[i].item())` and `:142` `bool(has_initial_state[i].item())` — both
  inside the per-request Python `for` over `splits`. So the loud frame is the first host
  sync *after* the fault, and the owner is the launch before it. Same reading as
  `../hccl-and-hardware/latched-runtime-error-masking.md`, one layer up in Python.
- **The path is reachable for a GDN checkpoint on 310P, and the reachability read is
  positive here:** `vllm_ascend/utils.py:793-794` imports `AscendGatedDeltaNetAttention310`
  under `is_310p()`, so Qwen3.5 goes through `_310p/ops/fla/gdn_310.py` (the opposite of
  the VL/OCR row above — that model never enters it).
- **Both indexed writes are prefill-only.** At v0.20.2rc1 they sit inside
  `if attn_metadata.num_prefills > 0:` (`gdn_310.py:249`): `:251`
  `initial_state[~has_initial_state, ...] = 0` (boolean-mask write = `nonzero` +
  `index_put_`) and `:269` `ssm_state[non_spec_state_indices_tensor] =
  last_recurrent_state.to(ssm_state.dtype)`; the decode branch (`:270+`) calls the
  dedicated `npu_recurrent_gated_delta_rule_310`. That is *why* every GDN report in this
  family looks like "starts cleanly, dies on the first request" with `FULL_DECODE_ONLY`
  capture succeeding — and why **`--enforce-eager` is not an ablation for it**: the
  suspected writes are outside the captured decode region, so removing graph mode removes
  nothing they run in.
- **`--mamba-cache-mode align` + `--enable-prefix-caching` is the ranked sub-cause, not a
  hypothesis-free guess:** under align mode the mamba state indices are built per block
  with padding slots, and both statements index `ssm_state` with
  `non_spec_state_indices_tensor`, and the prefill-side `npu_causal_conv1d_310` call
  (`:178`, same tensor as `cache_indices` at `:184`, `initial_state_mode=has_initial_state
  .to(torch.int64)` at `:185`; the spec-decode twin is `:161-173`) consumes it too, so one
  bad slot can fault in either op. A padded/invalid slot is exactly the shape of an
  out-of-range `index_put_`. The check is one launch, not a code walk.
- **The capability verdict needs the CI row beside the doc line.** `310p.md`'s closing
  note is only half the evidence at this pin: `tests/e2e/310p/singlecard/
  test_dense_model_singlecard.py:75,91` runs `Qwen/Qwen3.5-4B` **on 310P single card**,
  `tensor_parallel_size=1`, `dtype="float16"`, `max_model_len=16384` — one eager case and
  one ACL-graph case (`cudagraph_mode: FULL_DECODE_ONLY`, capture sizes
  `[1,2,4,8,16]`, `max_num_seqs=16`, `gpu_memory_utilization=0.80`,
  `mamba_ssm_cache_dtype="float16"`, `fuse_norm_quant: False`) — with **no** prefix
  caching, **no** chunked prefill and **no** `mamba-cache-mode align`. So Qwen3.5 on 310P
  is CI-covered for one size, the verdict is **per-size** ("0.8B/9B are outside the
  covered row, the family is not"), and that CI command *is* the config control the
  ladder below should step back to.

Ladder for this sub-shape: 1) `ASCEND_LAUNCH_BLOCKING=1` on the same request; 2)
same box, the CI command as written (drop `--enable-prefix-caching` **and**
`--mamba-cache-mode align` together — they are one pair), then drop
`--enable-chunked-prefill`, `--max-num-seqs 16`, `--gpu-memory-utilization 0.8`; 3) the
covered `Qwen/Qwen3.5-4B` on the same box (4B passes, 0.8B/9B fail ⇒ checkpoint/size;
all fail ⇒ this flag set); 4) the plog `[AIC_INFO] fault kernel_name=` block. Do not
spend a rung on `--enforce-eager` (see above).

## Then the capability verdict, in the same message

The reported model is outside the pin's validated set on three axes, and the same document
says so: `docs/source/tutorials/hardwares/310p.md:405`, the closing note —
*"For early access to Qwen3-MoE, Qwen3-VL, and **preview support for Qwen3.5 and Qwen3.6**
with performance acceleration, follow #7394"* — while the table at `:98-107` validates only
`Eco-Tech/Qwen3-{8B,14B,32B}-w8a8sc-310` (dense, 310-suffixed conversions). A generic
`-w8a8` 35B-A3B MoE on a GDN family is therefore *documented preview*, and `#7394` is
usable only for what the doc says about it. Check it against the CI row in the section
above before writing "unsupported": from v0.20.2rc1 the tree also runs
`Qwen/Qwen3.5-4B` in `tests/e2e/310p/singlecard/`, so *preview* is the doc's statement
about the sizes it does not list, never "no 310P coverage" — name which of the two you
are extrapolating from. `atlas-300i-310p.md` holds the per-release
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
