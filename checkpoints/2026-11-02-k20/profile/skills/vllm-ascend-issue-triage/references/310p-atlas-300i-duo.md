# 310P (Atlas 300I DUO) triage notes

Verified in the clones at tags `v0.21.0rc1` (2026-06-12) and `v0.22.1rc1` (2026-06-30); upstream
comparisons in `/repos/vllm-project/vllm-ascend`. The early-release section below was re-verified at
`v0.9.1rc1`, `v0.9.2rc1`, `v0.10.0rc1`, `v0.11.0rc0` and through `v0.27.1rc1` in the same clone.

## Platform shape

- 310P = Atlas 300I DUO, Ascend310P3 chips, fp16-oriented. `vllm_ascend.utils.is_310p()` selects it.
- No Triton: `docs/source/tutorials/hardwares/310p.md` says Atlas 300I DUO does not support `triton` /
  `triton-ascend` and recommends `pip uninstall -y triton-ascend triton`. A 310P log line "Triton not
  installed or not compatible" is therefore expected, not a cause.
- Platform code: `vllm_ascend/_310p/{attention,ops/fla,fused_moe,quantization,spec_decode,sample,
  kv_block_zeroer.py,block_table.py,model_runner_310p.py,npu_input_batch.py,sharded_state_loader_310p.py}`;
  images `Dockerfile.310p` / `Dockerfile.310p.openEuler`; e2e coverage under
  `tests/e2e/**/_310p/` and `tests/e2e/nightly/310p/`. That whole directory arrives with
  `1ffca8673` ("[Feature]: Support 310P device run qwen2.5/3 dense and qwen2.5vl models", #5776,
  2026-01-17), **first tag `v0.14.0rc1`**; `v0.10.0rc1`-`v0.13.0` contain **no** `_310p/` file at all
  (`git ls-tree -r --name-only <tag> | grep -c '_310p/'` = 0, 12 files at `v0.14.0rc1`), so a
  pre-0.14 310P report must be traced through the generic classes and must not be explained with a
  `_310p/...` path that does not exist at its tag.
- `Ascend310PDeviceAdaptor` in `vllm_ascend/device/device_op.py` exists only from `v0.23.0`; do not cite
  it for a 0.21/0.22 report.

## Wrong image variant: `socversion ... does not match the current device socversion`

A 310P box started from the generic (A2) image dies at the first inference with:

```text
[ERROR] the socversion Ascend910B1 of bin package does not match the current device socverison
Ascend310P3. Please modify default socversion in run.sh or execute run.sh with socversion parameter.
```

- The string is **not** in the `vllm-ascend` clone (checked); it is the CANN custom-op kernel package
  refusing to load. Do not grep the plugin for it, and do not treat the model as the fault.
- The in-tree half is verified at `v0.10.0rc1`: `vllm_ascend/envs.py:56-57` defaults `SOC_VERSION`
  to `ASCEND910B1`; `setup.py:85-98` bakes that value into `vllm_ascend/_build_info.py` as
  `__soc_version__`; `vllm_ascend/utils.py:62-66` derives `is_310p()` from
  `_build_info.__soc_version__` — **build-time, never the runtime device**. The generic `Dockerfile`
  sets no `SOC_VERSION` at all (zero hits), so it bakes `ASCEND910B1`, while `Dockerfile.310p:53`
  exports `SOC_VERSION=ASCEND310P3`. A generic image on 310P hardware therefore loads a 910B1 op
  package *and* silently takes the A2 code paths.
- `setup.py:90-92` also rejects a 310 build without `COMPILE_CUSTOM_KERNELS=1` ("SOC version 310 only
  supports custom kernels"), so a source build for 310P needs `SOC_VERSION=ascend310p1` (or
  `ASCEND310P3`) **and** that variable set.
- The build-time rule survives the later refactor, so it is not a 0.10 artefact: on `main`
  `vllm_ascend/device/device_config.py:28-42` prefers `_build_info.__device_type__`, falls back to
  `__soc_version__` (default `"ASCEND910B1"`), `:57-58 is_310p()` compares it against
  `AscendDeviceType._310P`, and the generic `Dockerfile:53` is explicit about it
  (`ARG SOC_VERSION="ascend910b1"`) where `Dockerfile.310p:51` is `ascend310p1`. From `v0.12.0rc1`
  the plugin also detects the mismatch itself and names it, so the CANN text above stops appearing:
  `worker_v1.py:96` calls `check_ascend_device_type`, which reports "Current device type: {runtime}
  does not match the installed version's device type: {built}, please check your installation
  package." (`v0.12.0rc1` as an `assert` in `utils.py:707`; `v0.18.0` `utils.py:720,738`; `main`
  `device/device_config.py:65-78` as a `RuntimeError`, called from `worker/worker.py:207`). Either
  message means **wrong image variant**, not a broken NPU: re-read the pair at the reported tag
  instead of quoting these line numbers.
- The workaround is the documented variant `quay.io/ascend/vllm-ascend:<ver>-310p`
  (`docs/source/tutorials/single_node_300i.md:15` at `v0.10.0rc1`-`v0.11.0rc0`; the same page is
  `310p.md` from `v0.12.0rc1` and `hardwares/310p.md` from `v0.15.0rc1` — see "The tutorial moved"
  below), whose recipe is not the A2 one: Qwen2.5-7B-Instruct
  is TP=2 with `--enforce-eager --dtype float16 --compilation-config
  '{"custom_ops":["none", "+rms_norm", "+rotary_embedding"]}'` (`:87-99`), and the page's note says the
  300I series is experimental, eager-only and float16. A reporter running TP=1 on the generic image is
  two documented deltas off that path.
- A **different** start-up failure, `AclSetCompileopt ... 500001` / `init runtime platform info failed`
  with `SocVersion = Ascend310P3` in the block, is not the image-suffix mismatch above: it
  names the hardware, not a `-310p` image tag. Treat it as container device access
  ([image-and-build-reports.md](image-and-build-reports.md)); a report can carry both that and a
  310P model-support symptom, and they are two layers.
- Boundary: read from tag-pinned source; nothing was executed on 310P. The device identity rests on
  the error text plus whatever `npu-smi` the thread supplies — at `v0.10.0rc1` nothing compares the
  built type to the runtime SoC, so the variant can pass image pull and container start before the op
  package rejects it.

## Attention mask memory — the dominant 310P startup failure mode

The A2/A3 shape of the same class (v0.11/v0.12 line, `AttentionMaskBuilder` in
`vllm_ascend/attention/attention_mask.py`, no NZ conversion) is in
[attention-mask-oom.md](attention-mask-oom.md); the two share the `N^2 * dtype_bytes` size check
but not the code, the gate or the fix boundary.

Documented, in-tree, identically at v0.20.2rc1, v0.21.0rc1 and v0.22.1rc1
(`docs/source/tutorials/hardwares/310p.md`,
"Online Inference on NPU" warning): do not rely on `max-model-len` auto detection; the 310P attention
path passes `model_config.max_model_len` to `AttentionMaskBuilder310`, which builds a full causal mask
`[max_model_len, max_model_len]` in float16 and converts it to FRACTAL_NZ; the prefill/chunked-prefill
path consumes that explicit mask and "there is currently no compressed-mask path"; `O(max_model_len^2)`
can exceed NPU memory and OOM. Recommendation: always pass a conservative explicit value such as
`--max-model-len 16384`.

Code behind it:

- `vllm_ascend/_310p/attention/metadata_builder.py`: `AttentionMaskBuilder310(self.device,
  vllm_config.model_config.max_model_len)`.
- `vllm_ascend/_310p/attention/attention_mask.py`: `gen_causal_additive_mask` allocates
  `[max_seq_len, max_seq_len]` float16; `_get_causal_mask` caches `npu_format_cast(nd_to_nz_2d(mask),
  ACL_FORMAT_FRACTAL_NZ)`; `get_splitfuse_mask` selects rows from that cached full mask and converts
  with `nd_to_nz_spec`. `ACL_FORMAT_FRACTAL_NZ = 29` in `vllm_ascend/utils.py`.
- Compressed path: `COMPRESSED_MASK_SEQ_LEN = 2048`, `is_compressed_mask_supported()` =
  `hasattr(torch_npu, "_npu_flash_attention_v3") and hasattr(torch_npu, "_npu_paged_attention_splitfuse_v2")`.
  When supported the mask is a fixed `[2048, 2048]`; otherwise `max_seqlen = max_model_len`.
- Arithmetic: 65536^2 x 2 B ≈ 8.6 GB for the cached mask, plus the NZ conversion copy.

Corroborating issue-thread statement (vllm-ascend#10872, maintainer reply 2026-06-30): the image's
CANN 9.1.0.BETA1 attention operators still do not support a compressed mask, so 64K produces a mask of
about 8 GB and the ND2NZ conversion needs "at least 16 GB more"; the advice was to reduce
`--max-model-len` to about 20K and lower `--gpu-memory-utilization`, and earlier to drop
`--max-num-batched-tokens` set equal to `--max-model-len`.

Boundary: docs, code and maintainer agree on the mechanism and on the workaround. The report's symptom
was "stuck" (no explicit OOM text captured), so present "mask allocation exhausts NPU memory" as the
documented mechanism and the hang as consistent with it, not as an observed OOM. The decisive runtime
check is the `hasattr` result inside the container (i.e. whether the compressed path is even available)
plus device memory during mask build.

## MoE on 310P: dispatcher OOM at element counts a 16-sequence batch cannot produce (#8248)

This is the allocation class, not the routing-accuracy class below (#10892): here the service
fails during `arange`/`bincount` (and one mask) allocation, while #10892 serves and answers badly.

Report shape: `vllm-ascend:v0.18.0rc1-310p-openeuler`, Qwen3-30B-A3B-Instruct-2507 (MoE), body saying
only "model runs normally, cannot output" plus a `user-attachments/files/.../3.txt` link — no serve
line, no log. The 521 KB attachment is the real report; its script sets
`PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:128` and serves with `--tensor_parallel_size 4
--enable-expert-parallel --dtype float16 --kv-cache-dtype fp8 --gpu-memory-utilization 0.95
--max-num-seqs 16 --enforce-eager --tool-call-parser hermes` (among other flags), then repeats
first-request tracebacks on a 42.67-43.24 GiB card with 39-40 GiB already allocated.

Three failing allocations appear, in this order (timestamps from the log, same attachment):

| failing frame (frame lines match `v0.18.0rc1` exactly) | attempt | when |
|---|---|---|
| `vllm_ascend/_310p/fused_moe/token_dispatcher.py:113` `torch.arange(actual_expert_total_num, dtype=torch.int32)` in `moe_init_routing` | 20.21 GiB | 17:06:43, and again 17:11:40 / 17:14:53 |
| `vllm_ascend/_310p/attention/attention_mask.py` `_get_causal_mask` -> `nd_to_nz_2d(mask)` | 764.00 MiB | 17:09:33-17:09:40 |
| `vllm_ascend/_310p/fused_moe/token_dispatcher.py:116` `torch.bincount(sorted_expert_idx[:n] - expert_start, ...)`, same function | 8.41 GiB | 17:14:51 |

Read the failing frame before advising, because the two families pull opposite ways:

- The mask row is the section above: ~764 MiB is the `[max_model_len, max_model_len]` fp16 scale
  (800 MB ~ 20 000² x 2 B), i.e. the 20 000 run. It was requested with 1017 MiB reported free
  (952/691/398 MiB on later ranks), so the mask was not plainly larger than the free budget — on a
  card at `--gpu-memory-utilization 0.95` with `max_split_size_mb:128` set, the plausible extra
  factor is a fragmented/large-pool block (candidate). Advise lowering the utilization and
  reconsidering that allocator setting — a launcher `max_split_size_mb` suppresses the plugin's
  `expandable_segments` default ([env-vars-and-additional-config.md](env-vars-and-additional-config.md))
  — not only `--max-model-len`.
- Lowering `--max-model-len` from 20 000 to 15 000 did **not** clear the dispatcher failure — it
  recurs at 17:11 and 17:14 after that edit — so do not answer this report with the mask advice.
- Neither dispatcher row is a budget problem. At `v0.18.0rc1` `moe_init_routing` sets
  `actual_expert_total_num = mask.sum().item()` with `mask` over `expert_idx.flatten()`
  (`token_dispatcher.py:82-118`), which bounds both the `arange` length (`int32`, 4 B/element) and the
  `bincount` slice (whose int64 output plus int32->int64 cast is 8 B/element) by the same
  `num_rows * top_k` — yet 20.21 GiB (5.4e9 elements at 4 B) and 8.41 GiB (1.1e9 at 8 B) do not
  correspond to one shared `n`. Either way `n` is 1e9-scale, 3-4 orders above what
  `--max-num-seqs 16` with a <=20k prompt can hand the dispatcher, so the candidate is the
  shape/`top_k`/expert-range passed into `token_dispatch` (`ops/fused_moe/moe_comm_method.py:136
  fused_experts` -> `_310p/fused_moe/token_dispatcher.py:62 token_dispatch`) or an intermediate cast
  inside the function — not the device headroom. The traceback never shows the caller's shapes: one
  log line of `hidden_states.shape`/`topk_ids.shape` at dispatch entry, or the same run with
  `--enable-expert-parallel` off, is the next check.
- The thread is closed with zero comments and no root cause, so the arithmetic above is derived from
  the tag and the "which value is wrong" half stays a candidate.

Boundary: the reported frames (`:62`, `:113`, `:116`) pin to `v0.18.0rc1`, and the serve flags and
attempt sizes are read from the attachment; nothing was reproduced on hardware here, and whether
`--enable-expert-parallel` on 310P MoE is a supported shape is not documented at this tag.

## W8A8 startup crash: `QuantBatchMatmulV3` `507015` L0C conflict

Fault signature (identical in both recorded reports):

```text
ERROR ... [multiproc_executor.py:962] Kernel task happen error, retCode=0x26, [aicore exception]
  errorStr: L0C read/write conflict(vec read operation or cube write operation)
  [DFX_INFO]Aicore kernel execute failed, ... fault kernel_name=
  QuantBatchMatmulV3_NZ_NZ_int8_int8_fp16_high_performance_21, hash=5247287448945562503
  record event failed, runtime result = 507015
```

- Raised inside `NPUWorker310.determine_available_memory` -> `model_runner.profile_run()`, i.e. after
  all weights loaded and before the server starts — not an OOM and not a checkpoint-format error.
- The Python frame that raises is only the next synchronisation point and differs between runs:
  #10122 shows `_dummy_sampler_run` -> `hidden_states[logit_indices]`, #17082 shows
  `aclnnDynamicQuantV2` / `aclnnRmsNorm`. Take the faulting kernel from the device DFX line.
- The kernel is issued by the 310P W8A8 dynamic linear scheme:
  `vllm_ascend/_310p/quantization/methods/w8a8_dynamic.py`,
  `AscendW8A8DynamicLinearMethod310.apply` -> `torch_npu.npu_dynamic_quant` then
  `torch_npu.npu_quant_matmul` (v0.23.0rc1: `apply` line 184, `npu_quant_matmul` line 202). The
  CANN-side body for the 310P (arch20) kernel is `/repos/cann/ops-nn` ->
  `matmul/quant_batch_matmul_v3/op_kernel/arch20/`; that repo's tags (`v9.1.0-beta.2.pre`, `v9.1.1`,
  `v9.2.0`, ...) are independent of the plugin tags and are not what the image ships.
- The same `apply` carries two in-code 310P caveats at v0.23.0rc1: "There is an accuracy issue
  currently, which is expected to be fixed in the next version", and "Currently, W8A8 dynamic
  quantization supports only symmetric quantization." Reach for these when the complaint is 310P
  W8A8 accuracy rather than startup.
- Recurrence across models and topologies means "same operator defect", not "bad model": #10122
  (2026-06-08, 310P Qwen3.6-27B-w8a8, TP=2, `vllm-ascend` 0.20.2rc1 image) and #17082 (2026-09-21,
  Qwen3.5-9B-W8A8, TP=1, `v0.23.0rc1-310p`) carry the same kernel name, same hash and 507015.

Maintainer disposition (thread statements, not code): #10122, Tflowers-0129 2026-06-09 — the quant
operator is broken in that CANN version and the fix is merged into the CANN `9.1.0.beta2` branch,
awaiting a CANN community release; #17082, same maintainer 2026-09-23 — upgrade CANN, `9.1.0`
GA/commercial has the fix. Issue #17082 was closed on that statement.

Workaround direction (image metadata verified, runtime unverified): use an image whose declared CANN is
the fixed `9.1.0`. `Dockerfile.310p` `FROM quay.io/ascend/cann:` gives the map —
v0.23.0rc1, v0.24.0rc1 and v0.25.1rc1 build on `9.1.0-beta.1-310p`; v0.23.0.post1 builds on
`9.1.0-310p`; from v0.26.0rc1 the file uses `ARG CANN_VERSION="9.1.0"`. Registry-confirmed
(`linux/arm64`): `v0.23.0rc1-310p` (manifest `sha256:2f201dff…`, config `sha256:c91472dc…` — the
Image ID in #17082) declares `ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-9.1.0-beta.1`,
`SOC_VERSION=ascend310p1`, `VLLM_TAG=v0.23.0`; `v0.23.0.post1-310p`
(manifest `sha256:9f6434d6…`, config `sha256:4e604c58…`) declares `/usr/local/Ascend/cann-9.1.0`.
So `v0.23.0.post1-310p` keeps upstream vLLM `v0.23.0`, the V1 runner and the same 310P quant scheme
while moving the CANN base from `9.1.0-beta.1` to `9.1.0`; it is the cheapest way to separate the
CANN hypothesis from the MRv2 one (the plugin patch level still differs between rc1 and post1, so
this is a near-single-variable retest, not a controlled experiment).

Boundary: the reporter's success retest (#17082, 2026-09-21, `nightly-main-310p`, CANN `9.1.0`,
`NPUModelRunner310V2`) changed CANN *and* the model runner and says so itself, so it does not isolate
the fix; the V1 path on `9.1.0-beta.1` stays untested, and nothing here was executed on NPU. The
reporter's own observation stands: mocked W8A8 unit tests cannot catch a real device kernel fault, so
only a single-card 310P hardware run discriminates.

## Early-release 310P start-up failures (v0.9-v0.11, 2025)

Three 310P start-up crashes from the pre-0.14 line have their own mechanisms; none is the attention
mask nor `QuantBatchMatmulV3` above. Verified in the clone at `v0.9.1rc1`, `v0.9.2rc1`,
`v0.10.0rc1`, `v0.11.0rc0`, `v0.11.0`, `v0.12.0rc1`, `v0.13.0rc1` and `v0.14.0rc1`. #1383 has no
comments at all, and #3316 closes with the maintainer's platform-support statement quoted below;
#5170's only fixes are a maintainer "resolved in the latest version" and a reporter report upstream.

### `torch_npu._npu_rotary_embedding` -> `TransData` unsupported (v0.9.1rc1, #1383)

```text
Optype [TransData] of Ops kernel [AIcoreEngine] is unsupported. Reason: [tbe-custom]:op type TransData
  is not found in this op store. ... [Static shape check]:The format and dtype is not precisely
  equivalent to format and dtype in op information library
Optype [TransData] of Ops kernel [aicpu_ascend_kernel] is unsupported. Reason: Transdata op, groups
  should be greater than 1, but now is 1.
No supported Ops kernel and engine are found for [trans_TransData_6], optype [TransData].
```

- Raised in the dummy `profile_run` -> `rope_forward_oot` -> `torch_npu._npu_rotary_embedding`, i.e.
  the first forward, before any request. The failing node is a format-conversion `TransData` in the
  single-op GE graph; the device text itself says the op store has no matching kernel, and the
  operator-name line — not the Python frame — names what failed.
- Why 310P reaches `torch_npu` rather than the plugin's own kernel: at `v0.9.1rc1`
  `vllm_ascend/ops/rotary_embedding.py:51-52` guards the custom path with
  `custom_rotary_embedding_enabled(...) and not is_310p()`, and the fallback call is
  `torch_npu._npu_rotary_embedding(...)` at `:69` — exactly the reported line. So on 310P the plugin
  deliberately does not use `_C.rotary_embedding`, and its own code is not the raising statement.
- The `torch_npu` side: in `Ascend/op-plugin`, `op_plugin/python/atb/_atb_ops.py` lists
  `_npu_rotary_embedding` in `API_LIST`, loads `lib/libop_plugin_atb.so` (needs NNAL `libatb.so`), and
  `_patch_atb_ops()` does `setattr(torch_npu, api_name, getattr(torch.ops.atb, api_name))`. The
  reported traceback itself contains `torch_npu/op_plugin/atb/_atb_ops.py:72/80` frames, so that
  rebinding was active in the reporter's install — that part is evidence, not inference.
- Candidate mechanism (not verified): ATB's RotaryEmbedding op converts its inputs through a
  grouped `TransData` that the image's 310P op store does not provide, so the failure is CANN/ATB-side;
  the report's own log says "Submit an issue ... at gitee.com/ascend". Keep it a candidate — no
  maintainer comment, the issue closed without one, and the local `op-plugin` clone is `master`
  (26.x/6.0 branches), not the reporter's torch_npu 2.5.1 line.
- Lifecycle worth knowing before proposing an upgrade: the plugin calls `_npu_rotary_embedding` at
  every tag checked from `v0.9.1rc1` to `v0.26.0rc1`; it is replaced by `torch_npu.npu_mrope` in
  `dde6c59cb` ("Replace _npu_rotary_embedding with public npu_mrope API", #13631, 2026-08-07), first
  tag `v0.27.1rc1`. `vllm_ascend/_310p/ops/rotary_embedding.py` (from `1ffca8673`, first tag `v0.14.0rc1`)
  does not replace the plain-rope call, but by `v0.21.0rc1` it is a full MRoPE bridge that passes
  `is_neox_style` into `ApplyRotaryEmb.forward_static`; see the rope-override section below.
- Discriminator to ask for: the `_atb_ops.py` frames are the ATB detour; when they are absent the
  same `TransData` text means something else (a plugin-issued format cast). A run on
  `v0.27.1rc1`+ with the same model is the cheap applicability test.

### KV cache NZ cast -> `507013` SDMA fault + `EL0004` alloc failure (v0.11.0rc0, #3316)

```text
RuntimeError: npuSynchronizeDevice:... AclrtSynchronizeDeviceWithTimeout, error code is 507013
[Error]: System Direct Memory Access (DMA) hardware execution error.
EL0004: ... Failed to allocate memory. ... alloc device memory failed, runtime result = 207001
there is a sdma error, sdma channel is 0, ... The SMMU returns a Terminate error during page table
  translation ... Memory async copy failed ... length=2855272448
```

- Different device fault from the `507015` matmul and from an OOM: the DMA/SMMU terminate plus a
  failed allocation, on a multi-GB async copy during `initialize_kv_cache`.
- Raising statement at `v0.11.0rc0`: `worker/model_runner_v1.py:3043/3045`
  `k_cache/v_cache = self._convert_torch_format(...)`, and `_convert_torch_format` at `:2672-2674` is
  `torch_npu.npu_format_cast(tensor, ACL_FORMAT)`. `ACL_FORMAT` is module-level and 310P-selected:
  `:142 ACL_FORMAT = ACL_FORMAT_FRACTAL_NZ` under `is_310p()` (which also sets
  `torch_npu.npu.set_compile_mode(jit_compile=False)`), else `:144 ..._FRACTAL_ND`;
  `ACL_FORMAT_FRACTAL_NZ = 29` in `vllm_ascend/utils.py`. The reporter's frame numbers are 2 lines
  higher, so state which you pinned.
- Do **not** claim the conversion is new in 0.11: at `v0.10.0rc1` line 2309 already does
  `kv_cache = torch_npu.npu_format_cast(kv_cache, acl_format)` on the freshly zeroed cache. The
  observed difference "v0.11.0rc0 fails, v0.10.0rc1 runs" therefore has no established cause; the
  candidate is that the NZ copy needs a second full-size allocation on top of a KV pool sized to fill
  the device (the reported copies are 2.7-3.1 GiB per call), which the allocator cannot place.
- `Cannot create tensor with NZ format while dim < 2, tensor will be created with ND format` is the
  torch_npu warning that precedes the casts; it is not the failure and appears on working 310P runs.
- Maintainer disposition (Yikun, closing comment on #3316, 2025-12-23): because of community
  bandwidth, 310P has not been adapted to more models or to the latest vLLM versions; use the
  `v0.10.0rc1-310p` guide, "currently the most stable and fastest version for 310P"
  (`docs.vllm.ai/projects/ascend/.../tutorials/310p.html`). So pinning that image is the practical
  workaround, and the situation is platform-wide rather than model-specific: other reporters in the
  same thread hit it on 300I Pro (310P3) and on dual-chip 310p1 (which additionally needed an HCCL
  patch on that image), and one reports Qwen2.5-32B running single-card on 0.10.0rc1 at ~2.5
  tokens/s.
- Lowering `--gpu-memory-utilization` / `--max-model-len` to shrink the pool so the NZ copy fits is an
  untested candidate; the report's own `npu-smi` shows one chip in `Critical` health, which is a
  separate hardware question and would invalidate any single-chip measurement taken on that card.

### `_warm_up_atb` -> `RuntimeError: LinearOperation CreateOperation failed!` (v0.11.0, #5170)

```text
File ".../vllm_ascend/worker/worker_v1.py", line 341, in _warm_up_atb
    torch_npu._npu_matmul_add_fp32(x, weight, c)
RuntimeError: LinearOperation CreateOperation failed!
```

- Raised in the start-up warm-up, after the compile/graph step and before any request: at `v0.11.0`
  `vllm_ascend/worker/worker_v1.py` ends `compile_or_warm_up_model` with an unconditional
  `self._warm_up_atb()`, whose body allocates three small NPU tensors and calls
  `torch_npu._npu_matmul_add_fp32`. Nothing gates it: `envs.py` at `v0.11.0`, `v0.12.0rc1`,
  `v0.13.0rc1` and `v0.14.0rc1` has no ATB/warm-up switch and the method has no try/except, so it
  runs on every start-up that reaches warm-up, not only HTTP serving.
- Why it is a 310P matter at all: `_npu_matmul_add_fp32` is one of the ATB ops op-plugin patches into
  `torch_npu` (`op_plugin/python/atb/_atb_ops.py`, the same `API_LIST`/`_patch_atb_ops()` mechanism as
  the `_npu_rotary_embedding` entry above), and it is a *sample* op — the reporter says upstream
  `Ascend/pytorch` #94 answered that it is not intended for external use and is supported by no
  device model. Do not go looking for its kernel in `Ascend/pytorch` or `Ascend/op-plugin`: the
  recorded forward spent its remaining budget there after both decisive facts were already in hand.
- Call-site lifecycle (a moved-file trap, not a removal): `_npu_matmul_add_fp32` is at
  `worker/worker_v1.py:341` at `v0.11.0`, `:371` at `v0.12.0rc1`, then the file is renamed and the
  call sits at `worker/worker.py:384` at `v0.13.0rc1` and `:408` at `v0.14.0rc1`. `git show
  <tag>:vllm_ascend/worker/worker_v1.py | grep ...` returns nothing from `v0.13.0rc1` on while the
  call is still there — `git grep -n '<pat>' <tag> -- vllm_ascend` instead.
- Fix and boundary: `vllm_ascend/_310p/worker_310p.py` (`class NPUWorker310(NPUWorker)`) overrides
  `_warm_up_atb()` with a log line only — "310p device do not support
  torch_npu._npu_matmul_add_fp32 atb ops", "Skip warm-up atb ops for 310P device". That file arrives
  with `1ffca8673` (#5776), **first tag `v0.14.0rc1`**, the same PR that adds the `_310p/` package
  (11 paths). Before it, 310P runs the generic worker — 310P was *partially* supported at `v0.11.0`
  (`Dockerfile.310p`, `tests/e2e/310p/test_offline_inference_310p.py` and the 310P image workflows
  all exist) — and nothing in the tree skips this call. So the workarounds are a release
  `>= v0.14.0rc1` or a local no-op override of `_warm_up_atb` in the installed worker; there is no
  flag. The thread adds no mechanism: maintainer "resolved in the latest version", issue closed, and
  the reporter's own upstream report is the only statement about the op's supported-device scope.
- Not established: whether *every* 310P box on `v0.11.0` fails, or only ones whose ATB/NNAL install
  reaches `CreateOperation` (the repo's own 310P offline e2e test at that tag would take the same
  path). Read the reporter's `ATB Home` line and NNAL version before generalising; nothing was
  executed on 310P here.
- Cheap confirmation for a reporter stuck on an older tag: `git ls-tree -r --name-only <tag> | grep -c
  '_310p/'` (0 before `v0.14.0rc1`) plus the `_warm_up_atb` frame in their traceback.

## VL/OCR models that serve but answer wrongly: read the rope overrides first

A report of the form "GLM-OCR / MinerU2.5-2509-1.2B pulls up on the official `-310p` image but every
image-recognition answer is wrong, PaddleOCR-VL seems fine" (issue #11936, `v0.21.0rc1-310p`, no log)
has no traceback to pin. Read the tag's rope code before the model: the plugin substitutes its own
fused rotary module for every model, and that replacement ignored the pair style through the 0.29 line.

- **Registration.** `vllm_ascend/utils.py:671`/`:705` at `v0.21.0rc1` maps
  `"ApplyRotaryEmb": AscendApplyRotaryEmb` (still `main:851`/`:886`), so any model constructing
  `ApplyRotaryEmb` runs the plugin class. At vLLM `v0.21.0` that includes
  `glm_ocr.py:145 ApplyRotaryEmb(enforce_enable=True)`, `paddleocr_vl.py`, `dots_ocr.py`, `keye.py`,
  `mimo_v2_omni.py`, `step_vl.py`, `qwen2_vl.py` and `qwen2_5_vl.py`.
- **The defect.** At `v0.21.0rc1` `vllm_ascend/ops/rotary_embedding.py:570-590` (call at `:586`)
  duplicated `cos`/`sin` with `torch.cat((cos, cos))` and called
  `torch_npu.npu_rotary_mul(x, cos, sin)` with no `rotary_mode`: Neox "half" pairing for every
  caller, `self.is_neox_style` ignored, `rotary_dim < head_size` unhandled too. The same ignoring call
  is present at `v0.22.1rc1` through `v0.29.0.rc1` (by then with a `rotary_dim` split, still no
  `rotary_mode`). `0bb59db16` "[BugFix][Model] Honor interleaved RoPE in AscendApplyRotaryEmb
  (#17825)", 2026-10-08, first tags `v0.30.0rc1`/`v0.30.0rc2` and `v0.30.0`, branches on
  `self.is_neox_style`, pairs GPT-J with `cos.repeat_interleave(2)` and
  `rotary_mode="interleave"` (`main:765-782`). So `v0.30.0rc1`+ is the first upgrade that can carry
  this fix; below it the workarounds are a local patch or a non-310P/A2 comparison run.
- **310P has its own bridge, so name the class that ran.** `vllm_ascend/_310p/ops/rotary_embedding.py`
  (added by `1ffca8673`, first tag `v0.14.0rc1`) is not only an `MRotaryEmbedding` subclass at
  `v0.21.0rc1`: it carries `_apply_rotary_mrope_torch` (`:35-48`), which delegates to
  `ApplyRotaryEmb.forward_static(..., is_neox_style)` and so *does* honour the pair style, and
  `merge_mrope_cos_sin_for_apply` (`:51-65`), which handles `mrope_interleaved`; a module-level cos/sin
  slice is filled once per forward by `NPUModelRunner310`. An MRoPE model on 310P can therefore be
  correct where the generic fused module is wrong, and vice versa.
- **Keep the `mrope` `rotary_mode="half"` question open.** `AscendMRotaryEmbedding.forward_oot` ends
  in `torch_npu.npu_mrope(..., mrope_section=..., rotary_mode="half")` after its gates
  (`v0.21.0rc1:523-554`: Triton branch only when `HAS_TRITON and positions.ndim == 2 and
  mrope_interleaved`, `super()` when `mrope_section != [16, 24, 24]`; unchanged on `main:716-724`).
  Whether that is wrong for an interleaved model that falls through is argued in PR #12820 (open, no
  merged fix, no runtime evidence) — a candidate, not a finding.
- **Discriminator.** Greedy single-image OCR on the same checkpoint at another backend, plus the
  construction site in the model file at the reported vLLM tag
  (`vllm/model_executor/models/<model>.py`: `ApplyRotaryEmb(...)` or `get_rope(... is_neox_style=...)`),
  says whether the pair style ever differed for that model. `glm_ocr.py:145` builds the module with the
  default `is_neox_style=True` and feeds cos/sin from its own 2-D `rot_pos_emb`, so for that model the
  flag alone is not the mechanism — compare the custom layout, and treat the class as the hypothesis
  the OCR report has to be tested against. Recorded evidence does not establish that #11936 is this
  defect; it establishes that this is the first place to look.

## MoE routing: `npu_moe_gating_top_k_softmax` and the >1024-token chunk guard (#10892)

This is the accuracy class, distinct from the dispatcher OOM above (#8248), which has no output
at all.

A 310P MoE report that **serves but answers badly** — "过度思考"/overthinking, degraded or malformed
JSON, no traceback — can be the platform's fused routing op, not the sampler or the model.
Maintainer-named cause in #10892 (Qwen3.6-35B-A3B-w8a8, `v0.21.0rc1`, closed 2026-07-23, labels
`310p`/`qwen3.6`): `MoEGatingTopkSoftmax` returns invalid routing (INF) once the token count in one
call is large.

- The call at the reported tag is unchunked: `vllm_ascend/_310p/fused_moe/experts_selector.py:60-62`
  at `v0.21.0rc1` — under `scoring_func == "softmax" and not use_grouped_topk and
  custom_routing_function is None`, `torch_npu.npu_moe_gating_top_k_softmax(router_logits, k=top_k)`
  receives the whole `router_logits`. Reached by the 310P MoE schemes, e.g. W8A8
  `_310p/quantization/methods/w8a8_dynamic.py:110` and `_310p/fused_moe/fused_moe.py:85`; the file
  exists `v0.21.0rc1`-`v0.26.0rc1`.
- Fix `0dc58f8e8` (#11391, 2026-07-04) chunks it when `router_logits.shape[0] > 1024`:
  `router_logits.split(1024, dim=0)` then concatenates the per-chunk weights/ids; **first containing
  tag `v0.23.0rc1`**. Quote the threshold from the code, not the PR body: the body says "input token
  dimension ... is 2048", the implemented chunk size is 1024. So a `v0.21.0rc1`/`v0.22.1rc1` run
  whose prefill batch passes 1024 tokens is inside this defect; the PR says the operator-level fix
  "may not be externally available", so the reachable fix is the plugin tag, not an operator
  reinstall — move to `>= v0.23.0rc1`.
- The code moved rather than regressed. `40348c31d` (#13621) deletes the legacy helper modules at
  `v0.27.1rc1`, and the same chunking survives in `_310p/fused_moe/grouped_topk_router.py`
  (`MAX_TOKENS_PER_GATING_CALL = 1024`, guard at `:49-57`; from `26f350ead`, #13417) with an added
  `router_logits.to(torch.float16)` ("`npu_moe_gating_top_k_softmax` requires FP16 ND input"). Check
  that file on `main` before calling a missing `experts_selector.py` a regression.
- The reporter's deployment is off-recipe: the tag's 310P page documents recipes only for dense
  Qwen3-8B/14B/32B W8A8SC and at `310p.md:405` (`v0.21.0rc1`) points Qwen3-MoE, Qwen3-VL and
  "preview support for Qwen3.5 and Qwen3.6" at #7394. Say the deployment is preview, then give the
  op answer; do not transplant a dense recipe's flags.
- Untested avoidance on a pre-`v0.23.0rc1` tag: keep one forward's token count at or under 1024
  (`--max-num-batched-tokens 1024`), the range the guard restores. Derived from the guard; no run
  behind it — label it a candidate next to the upgrade.

## Model runner V1 vs V2 on 310P (which releases can run MRv2)

`VLLM_USE_V2_MODEL_RUNNER=1` sets `vllm_config.use_v2_model_runner`; `vllm_ascend/worker/worker.py`
applies a version guard, then `NPUWorker310._create_model_runner` selects `NPUModelRunner310V2` or
`NPUModelRunner310` and logs `Using NPUWorker310 and NPUModelRunner310[V2].`

| ref | guard in `worker.py` | 310P V2 class present |
| --- | --- | --- |
| v0.23.0rc1 / v0.23.0 / v0.23.0.post1 | `vllm_version_is("0.23.0")` -> warning, forced back to V1 | no (`_310p/model_runner_310p.py` only) |
| v0.24.0rc1 | `vllm_version_is("0.24.0")` -> "supported only on the verified vLLM main commit", forced back | no |
| v0.25.1rc1, v0.26.0rc1 | none | no |
| v0.27.1rc1 and later | none | yes, `_310p/worker/v2/model_runner.py` |

The 310P V2 runner arrives with commit `59a36f29` (`[Feature][310P] Adapting MRV2 for the 310P`,
#14761, 2026-08-26); `git tag --contains 59a36f29` starts at `v0.27.1rc1` and does not include
v0.23.x/v0.24/v0.25/v0.26. So on any 0.23–0.26 310P image the flag cannot select a V2 runner — answer
"not available on that release" instead of searching for a V2 log line, and never offer it as the fix
for a v0.23.0rc1-310p report.

A nightly/main build is identified by its commit, not its version string: `c173a64a4` (2026-09-21, after
`59a36f29`) reports `0.19.1rc2.dev2294+gc173a64a4`, because the package version comes from
`setuptools_scm` in `setup.py` on a shallow clone.

## The tutorial moved, and the graph-mode answer is the doc first

A `310p.md` path only exists from `v0.12.0rc1`. Read the right file or the fetch returns nothing.
(A startup that dies with `socversion ... does not match the current device` before any request is
the wrong image variant, not a doc-path question — see "Wrong image variant" above.)

| ref | tutorial file |
| --- | --- |
| `v0.10.0rc1`, `v0.10.1rc1`, `v0.11.0rc0` | `docs/source/tutorials/single_node_300i.md` |
| `v0.12.0rc1` .. `v0.14.0rc1` | `docs/source/tutorials/310p.md` |
| `v0.15.0rc1`+ (restructure `7d4833bce`, #6501, 2026-02-10) | `docs/source/tutorials/hardwares/310p.md` |

"When will the 310 series support graph mode / quantization" is answered by that file's note, not by
a code mechanism, and the note changes over time:

- `v0.10.0rc1` .. `v0.18.0rc1`: "Currently, the 310I series only supports eager mode and the data
  type is float16", plus "This Atlas 300I series is currently experimental". Every serve line in the
  0.10.0rc1 file is `--enforce-eager --dtype float16`, so at that tag the reporter's command being
  identical to the tutorial is the whole answer: the perf gap is a documented experimental-platform
  limit. The note survives the doc moves: `grep -c 'only supports eager mode'` is 1 at
  `v0.10.0rc1`, `v0.12.0rc1`, `v0.14.0rc1`, `v0.15.0rc1`, `v0.16.0rc1`, `v0.17.0rc1` and
  `v0.18.0rc1`, 0 at `v0.18.0`. So a `v0.17.0rc1` or `v0.18.0rc1` reporter still gets "eager-only is
  the documented platform state", not "graph mode arrived".
- `v0.18.0`+ adds the `FULL_DECODE_ONLY` graph-mode note (5 hits in `hardwares/310p.md` at `v0.18.0`,
  0 at `v0.18.0rc1`, `v0.15.0rc1` and `v0.16.0rc1`) — the *release*, not its rc, so "no graph mode
  on 310P" becomes wrong at exactly `v0.18.0`. An rc that lags its own release doc is normal here.
- No `v0.17.1rc1` tag exists in the clone (`git tag --list 'v0.17*'` -> `v0.17.0rc1` only). Never close
  one of these boundaries on a tag you did not check exists: `git show <missing-tag>:<doc>` fails and
  `grep -c` still prints `0`, indistinguishable from "the line was removed" — that is how an earlier
  pass recorded the eager note as ending one release early. Assert the ref first
  (`git rev-parse -q --verify refs/tags/<t>`) when a per-tag loop prints a 0.
- The `--compilation-config '{"custom_ops":["none","+rms_norm","+rotary_embedding"]}'` in those
  recipes is not a user perf knob: `vllm_ascend/platform.py:185-187` at `v0.10.0rc1` reads
  "Activate custom ops for v1, except on 310P" -> `if not is_310p(): compilation_config.custom_ops =
  ["all"]`, so the recipe restates the platform default after the gate. And `+rotary_embedding` is
  inert on 310P there: `vllm_ascend/ops/rotary_embedding.py:60-61` selects the custom kernel only
  when `custom_rotary_embedding_enabled(...) and not is_310p()`. Do not promise a speedup from either
  entry; do not read their presence as a misconfiguration either.
- The quantization half of the same question is code-present but doc-silent at `v0.10.0rc1`:
  `platform.py:53 supported_quantization = [ASCEND_QUATIZATION_METHOD]` (= `"ascend"`, `utils.py:49-50`),
  `quantization/quant_config.py:44 @register_quantization_config(...)`, and
  `quantization/{w8a8,w8a8_dynamic,w4a8_dynamic}.py` all exist, while `SOC_VERSION_INFERENCE_SERIES
  = ["Ascend310P3"]` shows the platform was already special-cased. The `single_node_300i.md` file
  never mentions `--quantization`, so at that tag "quantization is implemented but has no 310P
  recipe" is the defensible statement — promise no recipe, and ask for the quantized checkpoint's
  provenance before discussing a kernel.
- At `v0.10.0rc1` `check_ascend_config` (`ascend_config.py:148-182`) is the graph-mode gate, and it
  is model- and mode-specific: the eager+graph abort ("Can't enable graph mode and eager mode at the
  same time. Please set `enforce_eager=False` if you attempt to enable NPU graph mode.", `:152-157`)
  fires only for **torchair** graph config plus `--enforce-eager`, not for a `cudagraph_mode`; in
  graph mode ACL graph is refused for `deepseek*` model types (`:174-178`, "ACL Graph does not
  support deepseek ... Or set `enforce_eager=True`") and only warns for types other than qwen
  (`:179-182`, "ACL Graph is currently experimental"). Read these three branches before telling a
  reporter that a flag combination is rejected.

## Graph mode limits

`docs/.../hardwares/310p.md` (the post-`v0.15.0rc1` path): `FULL_DECODE_ONLY` is supported, but with
TP > 1 the number of capturable graphs is limited by model depth (example given: Qwen3-32B
captures/replays 2); TP = 1 has no such limit.

That note's device-side form, observed in #12989 (`v0.23.0rc1-310p`, Qwen3-14B W8A8SC, TP2,
`cudagraph_capture_sizes [1,2,4,8,16]`): capture dies partway through the list and the driver
reports event-resource exhaustion, not a kernel fault.

```text
Capturing CUDA graphs (decode, FULL): 4/5 | ...        (dies on the 5th)
Insufficient_Event_Resources(EL0008): The event resources are insufficient.
  Create capture event failed, stream_id=[27->36], event_id=[-1->640] ...
  rtEventRecord execution failed, reason=driver error:no event resource ... runtime result = 207007
RuntimeError: NPUModelRunner failed, error is capture_end:../torch_npu/csrc/core/npu/NPUGraph.cpp:247
  ... AclmdlRICaptureEnd(...), error code is 507903   (stream capture invalidated by the above)
```

- Read it as the documented limit, not as `acl_graph`/kernel behaviour. The underlying
  `EL0008` + `207007` is event-id exhaustion (`event_id=[-1->640]` is the driver probe); the
  plugin-side frames are only `vllm_ascend/worker/model_runner_v1.py capture_model` wrapping
  `GPUModelRunner.capture_model`, and the `torch_npu/dynamo/npugraph_ex/_acl_concrete_graph/acl_graph.py`
  frames live in the torch_npu wheel, which is not in these clones — do not pin or explain them
  ([revision-pinning.md](revision-pinning.md)). The 310P recipes disable that backend
  (`tutorials/models/Qwen3.5-27B-Qwen3.6-27B.md:386` at `v0.23.0rc1`: `enable_npugraph_ex: false` is
  "required on Atlas inference products because `enable_npugraph_ex` is not supported on this
  platform"), yet #12989's traceback still contains `npugraph_ex` frames — do not spend rounds on
  whether the flag was honoured; the `EL0008` underneath decides the outcome either way.
- Workaround, reporter-verified in the same thread: the identical serve line plus `--enforce-eager`
  (no capture) started and served TP2 on the two cards. Fewer capture sizes is the middle option and
  is what the docs' TP>1 recipes do (Qwen3-32B TP4 keeps `[16,32]`; TP=1 has no limit). The
  maintainer answer is the doc pointer plus "needs a future driver release", and no 310P
  event-resource fix appears in the commit history searched through `main` — do not promise one.
- Availability boundary: the note exists in that page through `v0.23.0rc1` only (`grep -c
  'event-id resource constraints'` = 1 at `v0.21.0rc1`, `v0.22.1rc1`, `v0.23.0rc1`). From
  `v0.24.0rc1` the English page is absent from the tree — only the stale
  `docs/source/locale/zh_CN/LC_MESSAGES/tutorials/hardwares/310p.po` still carries the text — so the
  note's disappearance is page removal, not evidence the limit was fixed.
- The same tag's `tutorials/models/Qwen3-Dense.md` 310P tab prescribes Qwen3-8B at TP1
  `[1,2,4,8,16,32]`, Qwen3-14B at TP1 `[1,2,4,8,16]` and Qwen3-32B at TP4 `[16,32]`: there is no TP2
  14B recipe, so a TP2 run with five sizes is off the documented tab before it fails.

## Documented 310P model recipes

Same doc, per model: 310P-converted, quantized ModelScope weights (`Eco-Tech/*-310-vllm`, e.g.
`Eco-Tech/Qwen3-14B-w8a8sc-310-vllm/TP1/...`) served with `--quantization ascend`, `--dtype float16`,
`--load_format sharded_state`, `--no-enable-prefix-caching`, `--gpu-memory-utilization 0.90`, often
`--additional-config '{"ascend_compilation_config": {"fuse_norm_quant": false}}'`, with
`--max_model_len 16384` or `20480` and `FULL_DECODE_ONLY` plus explicit `cudagraph_capture_sizes`.
Reports deviating from this (plain HF checkpoint, no `--quantization`, no `--load_format
sharded_state`, plus `--enforce-eager` alongside a graph config) are outside the documented path — say
so and separate that from the crash analysis.

- **A `TP<N>` subdirectory checkpoint served without `--load_format sharded_state` dies in the
  vocabulary-parallel loader, not in quantization or the model.** #11406 served
  `.../Qwen3-32B-w8a8sc-310-vllm/TP4/...-tp4` at `v0.21.0rc1-310p` with every documented flag except
  that one; all four ranks failed after `Loading safetensors checkpoint shards: 0/4` with
  `AssertionError` at `vllm/model_executor/layers/vocab_parallel_embedding.py:463`
  (`assert loaded_weight.shape[output_dim] == self.org_vocab_size`, vLLM `v0.21.0`). That
  `weight_loader` expects the full unsharded vocab dim and narrows it per rank, so a rank-local shard
  (`org_vocab_size/TP` rows) trips the assert; `ShardedStateLoader` reads only the
  `model-rank-{rank}-part-{part}.safetensors` files and copies straight into the already-sharded
  parameter, bypassing the loader. The `v0.21.0rc1` recipe for this exact path carries `--load_format
  sharded_state` (`310p.md:171`); a launcher that generates the command without it (GPUStack did)
  reproduces this every time, and the fix is that flag, not a new checkpoint. The
  `vllm_ascend/patch/worker/patch_gqa_c8.py` frames in the traceback are the plugin's C8 cache-scale
  wrapper (installed for `Qwen3ForCausalLM.load_weights`, `_patched_causal_lm_load_weights` ->
  `_intercept_c8_scales`), which only diverts names that `quant_config.get_cache_scale` maps to a
  scale and forwards the rest — not the cause.

**"Can/can't 310P run model X" is answered by that page's model list at the reported tag, before any
mechanism.** The list is a tab set (`:::::{tab-item}`) whose heading carries the card count, e.g.
`single_node_300i.md` at `v0.10.0rc1` and `310p.md` at `v0.12.0rc1` list only Qwen3-0.6B (1 card),
Qwen2.5-7B-Instruct (2), Qwen2.5-VL-3B-Instruct and Pangu-Pro-MoE-72B (8). Quote the tab headings
rather than a maintainer's categorical "Qwen3 has not been supported in 310P" (issue #5905,
2026-01-15): Qwen3-0.6B is listed at every one of those tags. Dense **Qwen3-32B** is documented
only from **`v0.18.0`** (`00ddacf4e`, #8639; 0 hits at `v0.18.0rc1` and `v0.18.0rc2`), and there
only as **W8A8SC** weights (`Eco-Tech/Qwen3-32B-w8a8s-310`, `--load_format sharded_state
--quantization ascend`, TP4 example). A plain fp16 TP1 32B serve on `v0.10.0rc1-310p` is therefore
outside every documented path, and the reported silent stall after the `Initializing a V1 LLM
engine ... with config:` line (no traceback, no memory figures in the thread) is *consistent with* an
unsupported dense 32B load — label that mechanism a candidate, since the thread shows no OOM text. Code support for 310P qwen dense arrives
with #5776 at `v0.14.0rc1` (previous bullet), so the upgrade boundary for that report is 0.14
(code) / `v0.18.0` (documented recipe), not a flag change. A reporter's "I deployed it, see my
blog" is not a recipe: the thread shows no command, image or weights that worked, so label the
workaround "newer image (reporter-claimed)" and keep the 0.14/0.18 boundaries as the verified part.
Embedding/rerank models are a separate axis, not a model-list entry: 310P pooling starts at
`v0.20.2rc1` (`e029f683d`, #8846), and a Qwen3-Embedding serve below it dies while loading weights —
[pooling-embedding-rerank.md](pooling-embedding-rerank.md).

## GDN / linear attention on 310P

`vllm_ascend/_310p/ops/fla/gdn_310.py` subclasses the upstream `GatedDeltaNetAttention`:

- `_310p_get_state_dtype` forces the ssm state dtype to `torch.float16`.
- `npu_recurrent_gated_delta_rule_310(..., num_accepted_tokens=...)` forwards accepted-token counts into
  `torch.ops._C_ascend.npu_recurrent_gated_delta_rule_310`, flattening `ssm_state_indices` with
  `_flatten_state_indices(ssm_state_indices, cu_seqlens, total_tokens)`.
- `_merge_spec_and_non_spec_outputs_310` deliberately avoids `index_copy_`/IndexPutV2 and uses direct
  index assignment (`out[spec_token_indx] = spec_out`) with explicit length validation, because "NPU
  `index_copy_` (IndexPutV2) fails on some layouts". Net effect: 310P GDN does execute a boolean-mask
  `index_put_` in the mixed prefill+spec merge.
- `vllm_ascend/_310p/kv_block_zeroer.py: AscendKVBlockZeroer310` — Triton-free KV block zeroing, needed
  because "for MTP >= 2 hybrid models, newly allocated attention KV blocks must be zeroed via direct
  tensor writes" and Atlas 300I DUO has no Triton.

## 507018 / AICPU `IndexPut` faults (`error code is 507018`)

Where the host message comes from: `Reserve addr is invalid, should call halMemMap` is emitted by the
HDK driver, not by the plugin — `/repos/cann/driver/src/sdk_driver/svm/v2/master/comm/devmm_proc_info.c`
(see the `SUB_RESERVE_TYPE` branch). It is a device page-fault path for a VA in a reserved heap that is
neither host- nor device-mapped, i.e. the device faulted on a reserved address that no `halMemMap`
covered. The reporter's line also carried `ret=-22`, so the driver text/format can differ by version;
re-grep the reported string in `/repos/cann/driver` before leaning on it.

What is established for that family: the failure is *not* a documented known issue — the maintainer
asked for the complete error and operator logs (vllm-ascend#11188, 2026-06-30) — and the 310P GDN path
does contain a boolean-mask index assignment, which executes on AICPU (`libpt_kernels.so`,
`RunCpuKernel`), matching the reported kernel name. Main-only `ccc6564de` (#19295) reworks bool-mask
`IndexPut` in **GDN prefill state init**, a neighbouring but different site.

Keep these as competing candidates: (a) plugin-side boolean-mask `index_put_` on the AICPU path,
(b) driver/SVM mapping of the AICPU-writable address space (mount set, driver/firmware, smem/devmm
configuration). The AICPU/operator log, the driver mount set from the 310P doc, and the exact VA from
dmesg are the discriminators.

## Triage rules for this topic

These rules are kept with the topic they decide; SKILL.md carries the routing line and only
cross-cutting rules.

- **Separate a device operator fault from a plugin bug before tracing Python.** A 310P run
  that dies in `determine_available_memory -> profile_run` while the device log reports
  `fault kernel_name=QuantBatchMatmulV3...`, `runtime result = 507015` and
  `errorStr: L0C read/write conflict` is a CANN W8A8 matmul fault, not a model, checkpoint or
  memory problem, and the Python frame that raises is only the next synchronisation point (it
  moves between runs). Read the reported image's declared CANN (`ASCEND_TOOLKIT_HOME`) and the
  thread's CANN-side fix statement before reading plugin kernels.
  Its dtype-contract sibling, `aclnnQuantMatmulWeightNz ... scale dtype is bfloat16, out dtype ...
  DT_INT32`, rejects a *deliberate* pairing (int32 output from the two-step quantized MLP at
  `deepseek_v2.py:116-121` at `v0.9.0rc2` with a bf16 `weight_scale`; still so through `v0.10.2rc1`),
  so compare the
  pasted `collect_env.py` with that tag's `installation.md` pins (`CANN >= 8.1.RC1`,
  `torch-npu == 2.5.1`) instead of editing the checkpoint
  ([quantization-startup-keyerrors.md](quantization-startup-keyerrors.md)).
