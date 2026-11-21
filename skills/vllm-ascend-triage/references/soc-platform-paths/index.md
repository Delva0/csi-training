# SoC-conditional code paths (Atlas 300I / 310P first)

Scope: the report is on, or only makes sense on, a specific SoC — most often
**310P** (Atlas 300I Duo, `Ascend310P3`, image suffix `-310P`), where vllm-ascend
branches on `is_310p()` and the documented support is experimental. Also the
general question this topic answers for any SoC: *does the pinned version even
execute the code the traceback is in?* `is_310p()` gates 39 call sites in
vllm-ascend v0.11.0rc3 (`grep -rn "is_310p()" vllm_ascend/ | wc -l`), so a
symptom can be structure rather than a defect. The topic's other shape is the
*package-vs-card* question: engine start dying because the installed wheel was
built for a different SoC family than the live card reports
(`device-type-build-vs-runtime.md`).

Inherit the rules in SKILL.md. Rules that decide success here:

1. **Identify the SoC from evidence, not from the image tag alone.** Two
   different things are called "the SoC" and only one of them is checkable on the
   host: the **live** value `torch_npu.npu.get_soc_version()` (numeric: `220-225`
   A2, `250-255` A3, `200-205` 310P, `260` A5 at v0.21.0rc1) and the **build
   stamp** baked into the installed package (`_build_info.__device_type__` from
   v0.18-ish on, `__soc_version__` before that — `is_310p()` reads the stamp). Ask
   for both: `python -c "import torch_npu; print(torch_npu.npu.get_soc_version())"`
   and `python -c "from vllm_ascend import _build_info as b; print(b.__device_type__)"`.
   `npu-smi info` names the card; the tag `-310P`/`-a3` in an image name is a hint,
   not proof, that the wheel matches it.
2. **Do not let "the platform is experimental" be the whole answer.** The pinned
   tree does document that for 310P (below), and it is the right *first*
   sentence, but a report that carries a traceback has a cheaper, more useful
   answer inside it. #4812 got "310P is unsupported, downgrade to v0.10.0rc1" as
   cause 1 while the traceback it later posted was a device allocation failure at
   KV-cache initialisation — a memory budget question
   (`../runtime-memory/kv-init-and-format-cast-alloc.md`).
3. **SoC gates flip behaviour silently.** The same flag can be honoured on A2 and
   ignored/overwritten on 310P (`compilation_config.custom_ops`), and format
   conversions that are no-ops on 910B allocate and copy on 310P. Read the gate
   before attributing behaviour to the reporter's flag.
4. **A dtype or format value in the reporter's own serve line is checked against the
   pinned doc for their SoC before any source walk.** On 310P the doc's own note —
   `docs/source/tutorials/hardwares/310p.md:5`, "only supports eager mode and the
   float16 data type" — plus a `--kv-cache-dtype fp8` or a bf16 checkpoint *is* the
   verdict. Neither value is refused up front: the KV dtype is not read in
   `vllm_ascend/platform.py` at all (`rg -c kv_cache_dtype` → 0 at v0.18.0rc1), and no
   310P dtype gate rejects a bf16 checkpoint (`atlas-300i-310p.md`, verified there at
   v0.9.1rc1 — re-grep per pin). So the engine starts and an op refuses later, as an
   op-shaped error, and that error is not a licence to walk the op's library. Two
   recorded runs spent 25-35 turns inside `reshape_and_cache` / `IndexPut` with the
   offending value visible from the first paste, and shipped no message.
5. **A SoC subpackage is a fork, so diff it before theorising.** `vllm_ascend/_310p/`
   (and any later `_<soc>/`) subclasses the generic worker/platform and
   re-implements a few members — at v0.22.1rc1 `NPUWorker310` overrides exactly five
   (`_310p/worker_310p.py:49,58,80,140,144`), everything else is inherited. A wrong
   number or refusal inside such an override is usually a **missing propagation** of
   a fix the generic path already carries, so compare the override method with
   `vllm_ascend/worker/worker.py` and upstream first: it yields the cause, the patch
   and the precedents together (`../runtime-memory/kv-budget-per-instance-vs-device-global.md`
   is the recorded case — `#7372` kept a formula `#7427` had dropped).

Conditional routes:

- 310P / Atlas 300I deployments: the documented support status, the seven
  platform gates that change behaviour, the NZ (FRACTAL_NZ) format path, and the
  launch-environment items the 300I tutorial requires →
  `atlas-300i-310p.md`. That page also carries the per-release blocks a startup
  report needs: at v0.22.1rc1 the `_310p/` subpackage (the older gate line
  numbers are gone), the release's own validated `310p.md` table, the inert
  `fuse_norm_quant` flag and the `max_model_len`-sized mask — quote the block for
  the reporter's release, not the v0.11 one.
- 310P start-up dies at **KV sizing** with `Available KV cache memory: … (halved
  for workspace)` and `ValueError: To serve at least one request with the model's
  max seq len …`, while `npu-smi` shows free HBM (often the second engine on one
  card) → `../runtime-memory/kv-budget-per-instance-vs-device-global.md`: the
  override charges device-global HBM, `gpu_memory_utilization` is per-instance, and
  `--kv-cache-memory` is inert on 310P.
- The complaint is that **no 310P artefact exists for this machine** (arm64 host,
  an arm64 tag that pulls amd64, an aarch64 wheel) — an image/wheel pipeline
  question, not a platform code path →
  `../image-dependency-faults/image-arch-and-tag-availability.md`.
- Engine start dies at `worker.py` `check_ascend_device_type()` before
  `Starting to load model`: `AssertionError: Current device type: AscendDeviceType.A5
  does not match the installed version's device type: AscendDeviceType.A2`, or
  `RuntimeError: Can not support soc_version: <n>` — a wheel built for another SoC
  family (the release pipeline ships a2/a3/310p variants only, `Dockerfile.a5`
  exists in-tree) or a live SoC code the map lacks →
  `device-type-build-vs-runtime.md`.
- Engine start dies **inside torch_npu device open** on a `-310p`/pairing report —
  `SetPrecisionMode … aclrtSetCompileOpt(aclCompileOpt::ACL_PRECISION_MODE …) 500001`,
  `PlatformInfoInit … SocVersion <name>`, `GEInitialize failed`, `ERR99999` — with no
  `vllm_ascend` frame: the SoC the runtime resolved is the first thing to read against
  `npu-smi`, then the host driver/firmware vs the image's CANN base →
  `../startup-failures/acl-init-500001-and-soc-version.md` (the 300I mount list at
  `single_node_300i.md:20-31` is the cheapest conformance check).
- The traceback on a 310P host ends in `npu_format_cast` /
  `_convert_torch_format` with a device allocation error →
  `../runtime-memory/kv-init-and-format-cast-alloc.md`.
- A **non-fp16 KV cache on 310P** (`--kv-cache-dtype fp8` / `bfloat16` in the serve
  line): the engine comes up, the first request kills every rank at the KV **store**
  (`reshape_and_cache` → `device_op.py:32` → `torch_npu._npu_reshape_and_cache` →
  `ReshapeAndCacheOperation setup failed!`) — a config verdict plus a one-flag
  ablation, not an op bug → `310p-kv-cache-dtype.md`.
- A first-request **op fault naming an ordinary tensor op**, `507018` + "The aicpu
  execution is abnormal" + `kernelName=…, soName=libpt_kernels.so, funcName=RunCpuKernel`,
  or `<Op>Operation setup failed!` out of a `torch_npu._npu_*` call: the op ran on the
  CPU/AICPU fallback, so the owner is the Python statement (or the un-gated config)
  that launched it, and the dmesg `devmm … should call halMemMap` is downstream of the
  fault → `310p-cpu-fallback-op-fault.md`.
- A **penalty parameter** (`presence_penalty` / `repetition_penalty`) is what makes a 310P
  serve fail — "加上 presence_penalty=1.5 一到测试阶段就会崩溃", often with no traceback:
  the branch is printed by the code itself (`Triton not available, falling back to vLLM
  default penalty implementation` ⇒ the shipped 310P image's torch fallback), and the
  #2533/#2544 `_C.apply_repetition_penalties_` precedent no longer holds at v0.23.0 →
  `atlas-300i-310p.md`, its penalty section.
- A **quantized** model on 310P fails at the first quantized linear with an
  op-level dtype/layout complaint instead of a Python raise — e.g. `Bias dtype
  should be INT32, actual dtype is DT_FLOAT16` in the title alone: the
  checkpoint artifact decides (310-suffixed conversion vs the generic `-w8a8`),
  and the 310P `W8A8_DYNAMIC` scheme is the one path that forwards the model's
  fp16 bias → `310p-quant-checkpoint-soc-match.md` (three checks, no engine walk).
- A 310P W8A8 model dies **inside an int8 matmul kernel at start-up** — device DFX
  `QuantBatchMatmulV3_NZ_NZ_int8_int8_fp16_high_performance_*` + `runtime error: 507015` +
  `L0C read/write conflict` during `determine_available_memory -> profile_run`, on a
  `W8A8_DYNAMIC` checkpoint, while the Python frame moves (`aclnnDynamicQuantV2`,
  `aclnnRmsNorm`): the image's CANN/op-binary package and the 310P runner generation are
  the only two variables, and `VLLM_USE_V2_MODEL_RUNNER` is version-gated (no
  `NPUModelRunner310V2` before v0.23.0rc1) — the reporter's retest passed on a later
  `-310p` nightly with the V2 runner, with CANN and runner both changed →
  `310p-int8-matmul-startup-fault-and-runner-v2.md`.
- A **MoE** (A3B) checkpoint on 310P **serves and answers wrong but fluent text** —
  mixed-language CoT, self-contradiction, a final JSON that breaks the prompt's own rule,
  normal tokens/s and `finish_reason: stop`, no 乱码 and no repetition. That is not the
  VL-garbage lane below and not reasoning routing: the 310P selector routes through one
  fused op (`torch_npu.npu_moe_gating_top_k_softmax`,
  `_310p/fused_moe/experts_selector.py:61`) whose in-tree test discards its own
  `torch.allclose`, so probe it before any config ladder →
  `../accuracy-benchmarks/moe-gating-topk-precision.md` (#10892, reporter-confirmed).
- A **VL or OCR** checkpoint on 310P **serves and answers garbage** — 乱码,
  "精度有问题", "随便跑个图片识别就能复现", no traceback, while a text model on the
  same box is fine and the report names a sibling VL model that "seems OK" →
  `310p-vl-ocr-garbage-output.md`. The per-model doc support lookup reproduces
  that split, document parsing is a layout → recognise pipeline (the VLM alone is
  one stage), and the 310P rope-branch map there is a single probe via
  `scripts/310p-rope-branch.py`, not a source walk.
- 310P startup dies in the **first forward** (`profile_run`) with
  `EZ3002 … Optype [TransData] of Ops kernel [AIcoreEngine] is unsupported … not found
  in this op store` + `format and dtype is not precisely equivalent … op information
  library` → `atlas-300i-310p.md`, its `TransData`/`EZ3002` section: a dtype/format
  rejection (bf16 first, rope is always ATB on 310P), not the missing-kernel lane of
  `kernel-package-vs-card.md`.
- The op cannot be resolved for the SoC at all (`EZ1001`, `Get regInfo failed`,
  `does not support opType [X] … socVersion [ascend910_93]`) — a build/install
  fault, not a platform path →
  `../custom-ops-runtime-faults/op-not-registered-for-soc.md`.
- A **built-in** aclnn op fails before launch (`EZ9999 … Parse dynamic kernel
  config fail`, `AclOpKernelInit failed opType`, `<Op>AiCore
  ADD_TO_LAUNCHER_LIST_AICORE failed`) on a card the release's docs exclude, often
  asked as "does image `<tag>` support `<card>`?" → `kernel-package-vs-card.md`
  (image/CANN kernel package vs the live SoC; one standalone op probe).
  The `EZ9999 … Op <Op> does not has any binary` wording (with `call aclnn<Op>
  failed, error code is 561000` + `ERR00100 PTA call acl api failed`) is the same
  section and takes the same probe — on 310P it usually fires only on the
  **multimodal** request path (`aclnnIsNegInf` from a vision/NaN-guard call,
  #6082, v0.10.0rc1-310p), and the Python caller is in no pinned tree, so grep
  the trees once, then probe.
- An **omni / TTS / audio-out** model on 310P — "310p 使用
  quay.io/ascend/vllm-omni:vX.Y.Z 这个镜像来创建容器部署 <TTS 模型> 么?" — is a
  prebuilt-image verdict plus a build recipe, not a bug hunt: the project publishes omni
  images for **A2/A3 only** (the vendor install page, readable from the post's own
  attachment OCR), so the answer's first sentence is "no such tag", the second is the
  source build with `SOC_VERSION=ascend310p1`, the third is the per-stage fp16 +
  `enforce_eager` deploy-config → `310p-omni-model-deploy.md` and
  `templates/omni-310p-deploy-config.yaml`.
- The model **loads and dies in the first forward** with
  `AttributeError: '<SomeClass>310' object has no attribute '_<name>'` where `_<name>`
  is used as `self._<name>(...)` — a `_310p/` class that is a hand-copied sibling of a
  class under `vllm_ascend/ops/`: grep the name package-wide, quote the sibling's
  definition, and check the copy's return contract in the same pass →
  `310p-sibling-class-drift.md` (a Python error inside the copy has no plog line, and
  "the family is preview" does not explain it).
- The report is a version gate ("does V support model M on 310P?") — typically an
  `[Installation]`/`[Bug]` title, a tutorial-copied `vllm serve` line and the only log an
  external attachment URL → **read the pinned `docs/source/tutorials/hardwares/310p.md`
  first** (one `sed`): its validated table and its closing note name both the supported and
  the *preview* families ("preview support for Qwen3.5 and Qwen3.6 … follow #7394", `:207-209`
  at v0.18.0). A family outside that table with a preview note is a **capability verdict owed
  in the first message** — the reporter's next question is verbatim "只支持 #NNNN 中的 POC
  镜像吗?", and #NNNN is usually a future issue this snapshot cannot resolve. Then
  `../startup-failures/model-type-not-recognized.md` for the recognition-error form, and the
  status lines in `atlas-300i-310p.md` (its v0.18.0 block).
- 310P engine start on `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'` **with MTP**
  dies inside `Capturing CUDA graphs (decode, FULL)` with `aclrtMemcpy, error code is 107030`
  (+ `ERR00100`, the canned `EE1006 … Upgrade the CANN software version` block) at
  `_310p/attention/attention_mask.py:90 get_splitfuse_mask`, while the same launch in eager mode
  works: the fallback mask does a host readback inside capture and the capture-safe branch is gated
  on two torch_npu `_npu_*` bindings, so no CANN/toolkit upgrade moves it →
  `310p-mtp-graph-splitfuse-mask.md` (one `hasattr` line decides it).
- A **hybrid linear-attention** family (Qwen3.5/3.6-class GDN) fails on a `-310p` image at the
  first decode step with `AttributeError: 'TritonPlaceholder' object has no attribute 'cdiv'`
  (or `next_power_of_2`) → the image's own "does not support triton" rule meets a decode path
  whose `patch_triton` rebind is `HAS_TRITON`-gated: `atlas-300i-310p.md`, the triton-less
  hybrid section (a Python error before any device op, so no plog line — one grep of the
  traceback settles it).
