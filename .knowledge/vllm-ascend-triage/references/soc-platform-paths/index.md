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
   (`../runtime-memory/kv-init-and-format-cast-alloc.md`). The limit is a tag that
   *has* the lane: when it does not, rule 6 applies and the version is the answer.
3. **SoC gates flip behaviour silently.** The same flag can be honoured on A2 and
   ignored/overwritten on 310P (`compilation_config.custom_ops`), and format
   conversions that are no-ops on 910B allocate and copy on 310P. Read the gate
   before attributing behaviour to the reporter's flag.
4. **A dtype or format value in the reporter's own serve line is checked against the
   pinned doc for their SoC before any source walk.** Locate that doc with
   `rg --files docs/source/tutorials | rg 310` instead of trusting a quoted path (flat
   `docs/source/tutorials/310p.md` at v0.14.0rc1, `single_node_300i.md` at v0.9.x,
   `tutorials/hardwares/310p.md` from v0.18 on), then quote its **validated rows**,
   because the prose note moves and disappears: the "only supports eager mode and the
   float16 data type" sentence exists at v0.11.0rc3 (`single_node_300i.md:3-7`) and
   v0.12.0rc1 (`310p.md:5`) but has **no hit anywhere under `docs/` at v0.22.1rc1** —
   there `hardwares/310p.md:5` is a triton note and `:17-27` instead states the release
   supports `FULL_DECODE_ONLY` (no capturable-graph limit at TP=1), while all three
   rows carry `--dtype float16` and none passes `--enforce-eager` or
   `--kv-cache-dtype`. So a `--kv-cache-dtype fp8` or a bf16 checkpoint *is* outside
   the envelope, and that is the verdict — but never assert an eager-only/float16-only
   limit from memory for a v0.22+ pin. Neither value
   is refused up front: the KV dtype is not read in
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

6. **Separate "the lane is absent at this tag" from "the lane is experimental".**
   Rule 2 forbids stopping at "the platform is experimental" when the tree *has* the
   310P path and the traceback carries a cheaper answer. When the tag has no 310P lane
   at all the version *is* the answer and no mechanism is testable: at v0.7.3 and on a
   generic `v0.9.1` tag `rg -c -i '310' vllm_ascend/` is empty, there is no
   `Dockerfile.310p`, `docs/source/tutorials/` has no 300I page, and
   `faqs.md:14-20` says the series is "NOT supported yet". One grep plus `ls
   Dockerfile*` decides which rule applies; both #1500 (`setup failed!` in
   `_npu_rotary_embedding`) and #2809 (`_C.rotary_embedding … no Meta kernel`) were
   answered by a maintainer in one sentence naming the `-310p` image, while the
   recorded runs ranked a mechanism first and the image last
   (`atlas-300i-310p.md` §*The first official release has no 310P lane either*).

Conditional routes:

- A 300I/310P box on a **tag that carries no 310P lane** (`v0.7.3`, a generic
  `v0.9.1`; no `-310p` suffix and no `_310p/` in the tree) with any
  `profile_run`-phase crash — `setup failed!` out of `_npu_rotary_embedding`, a
  fake-tensor/Meta-kernel error, an op-store refusal: the `<version>-310p` image from
  the 300I tutorial is rung 1 and the dtype/op probe sits below it; the next paste
  after that move is the `TransData`/`EZ3002` block arriving from `npu_format_cast`
  in the attention builder, with `dtype=torch.bfloat16` in the engine line
  (`atlas-300i-310p.md`).

- 310P / Atlas 300I deployments: the documented support status, the seven
  platform gates that change behaviour, the NZ (FRACTAL_NZ) format path, and the
  launch-environment items the 300I tutorial requires →
  `atlas-300i-310p.md`. That page also carries the per-release blocks a startup
  report needs: at v0.22.1rc1 the `_310p/` subpackage (the older gate line
  numbers are gone), the release's own validated `310p.md` table, the inert
  `fuse_norm_quant` flag and the `max_model_len`-sized mask — quote the block for
  the reporter's release, not the v0.11 one. It also carries the pre-lane check: an
  experimental SoC lane can **postdate the reported pin**, so place its first release
  (300I/310P arrives in v0.9.1rc1; v0.9.0rc2 has one `310p` string in the whole tree)
  before quoting any doc, and read the reporter's quoted model list as the version
  marker it is.
- Engine start dies **inside capture/warm-up** on a `-310p` image with
  `RuntimeError: status == aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE
  INTERNAL ASSERT FAILED at "build/CMakeFiles/torch_npu.dir/compiler_depend.ts"` out of
  `collective_rpc("compile_or_warm_up_model")`, while the same command runs on a 910B:
  the 300I lane is **eager-only at that pin** — its tutorial passes `--enforce-eager` in
  every example and its `LLM()` tabs comment "For 300I series, only eager mode is
  supported" (`docs/source/tutorials/single_node_300i.md:65,94,135,194,237` at v0.9.2rc1;
  later pins move and reword that note, see `atlas-300i-310p.md`) — while
  `platform.py:172-181` does not gate the PIECEWISE branch on the SoC, so the default
  serve attempts capture. The assert is torch_npu `NPUGraph::capture_begin`'s post-begin
  check (`NPUGraph.cpp:161-162`, `Ascend/pytorch` `d33df55`), i.e. `AclmdlRICaptureBegin`
  "succeeded" and the stream never entered ACTIVE — rung 1 `--enforce-eager`
  (+ `--dtype float16`); full reading in `../startup-failures/acl-graph-capture.md`
  §Step 0b (#1768 follow-up, Atlas 300I Duo).
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
  `npu-smi`; in a container whose `docker run` already matches the docs, the next step is
  that same command + `--privileged=true` and nothing else (reporter-confirmed #302),
  then the host driver/firmware vs the image's CANN base →
  `../startup-failures/acl-init-500001-and-soc-version.md` (the 300I mount list at
  `single_node_300i.md:20-31` is the other conformance check).
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
  fault → `310p-cpu-fallback-op-fault.md`. That page also owns the **two-op paste**
  (a later `aclnn<Op> failed, error code is 507018` + `launch failed for <Op>,
  errno:361001` is the next launch on the poisoned stream, not a second fault — read
  `kernelName=` and the timestamps) and the **VL/OCR model** case, where no
  `_310p/ops/fla/` indexed write is reachable and the sampler's Triton-absent penalty
  fallback explains an `aclnnInplaceSub` but contains no `index_put_`. It also owns the
  **GDN-family** form on the 310P line (#13449, v0.20.2rc1): the two `index_put_` writes
  are prefill-only (`gdn_310.py:249-269`), the reporting frame is a `.item()`
  (`_310p/ops/causal_conv1d.py:137,142`), `--enforce-eager` removes nothing, and the
  capability verdict is per-size against the pin's own 310P e2e row for
  `Qwen/Qwen3.5-4B` — which is also the config control.
- The **create-time** sibling — `RuntimeError: LinearOperation CreateOperation failed!`
  (or `GroupTopkOperation …`) out of a `torch_npu._npu_*` call, typically the first ATB
  call inside `_warm_up_atb` — is **not** this route and not a pack question: no tensor was
  described, so the owner is the ATB/NNAL build × SoC (the pin's own 310P recipe is based on
  `cann:8.3.rc2-310p` and sources `nnal/atb/set_env.sh`) →
  `../custom-ops-runtime-faults/atb-create-operation-failed.md` (recorded #5170).
- A 310P serve that ran fine then dies **under load** with
  `[Error]: Event resources are insufficient.` / `Event id alloc error …
  retCode=0x7020019` / `rtEventRecord … no event resource` / `207007`, the Python
  frame in `_310p/sample/sampler.py` (`q = torch.empty_like(probs).cpu()`) and the
  op line naming `aclnnInplaceDiv`/`RealDiv`: a device **event-table** refusal, not
  a memory leak, and the frame is one sync point off the per-step
  `wait_stream(global_stream())` that requested the event — `--enforce-eager`
  splits graph-capture reservations from per-step churn, then sample the counter
  (`aclrtGetEventAvailNum`) → `../hccl-and-hardware/acl-event-resource-exhaustion.md`
  (#13007).
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
- A 310P **MoE** checkpoint dies at **launch** instead — `aclnn<Op>` gating op,
  `EZ1001` wrapped round `NnopbaseRunWithWorkspace … Assert ((executor) != nullptr)`,
  first request of a serve the reporter says fails on two different `-310p` tags — is a
  capability verdict before it is a code answer. The release that re-opened 310P scoped
  it: `docs/source/user_guide/release_notes.md:9` "only basic dense and vl models are
  supported with eager mode", and that build ships **no** custom aclnn op at all
  (`csrc/build_aclnn.sh:5-11` exits 0 for `^ascend310`). So the op in the paste is a
  CANN/`torch_npu` built-in, its name dates the run to a pre-0.14 selector generation,
  and the answer is the verdict plus the standalone op probe — **not** a hunt for the
  Python caller, which is in no pinned export (recorded #6488: 21 turns, 39 tool calls,
  81 characters of text, no answer) → `atlas-300i-310p.md` (its v0.14.0rc1 bullets) and
  `kernel-package-vs-card.md`; for the assert's inner `Check` lines,
  `../custom-ops-runtime-faults/index.md` (sixth pre-launch wording).
- The question is a **capability/status one at an older rc** with no log — "does
  `vllm-ascend:v0.9.1rc1-310p` support 300I Duo + MRoPE, e.g. qwen2.5-vl-7b and
  qwen2.5-vl-32b?" (#1684): the pin's docs are laid out differently (flat
  `docs/source/tutorials/`, matrix at `user_guide/supported_models.md` with no
  hardware column, no `_310p/` package), its three status statements disagree, and
  the closest VL recipe is `single_npu_multimodal.md`, not the 300I page — answer
  with those plus the sizing verdict, not with a rope-code walk →
  `atlas-300i-310p.md`, its "capability question at an old pin" section.
- A **VL or OCR** checkpoint on 310P **serves and answers garbage** — 乱码,
  "精度有问题", "随便跑个图片识别就能复现", no traceback, while a text model on the
  same box is fine and the report names a sibling VL model that "seems OK" →
  `310p-vl-ocr-garbage-output.md`. The per-model doc support lookup reproduces
  that split, document parsing is a layout → recognise pipeline (the VLM alone is
  one stage), and the 310P rope-branch map there is a single probe via
  `scripts/310p-rope-branch.py`, not a source walk.
- The **same 310P VL/OCR family fails at engine start instead**: `profile_run` →
  `_rope_forward_oot` → `npu_apply_rotary_pos_emb`, `call aclnnApplyRotaryPosEmbV2
  failed, error code is 161001` + `AclNN_Parameter_Error(EZ1001): Check cos !=
  nullptr failed`, while a text model on the box serves → `310p-vl-rope-cos-nullptr-startup.md`.
  One gate owns it (`set_cos_and_sin` allocates `_cos`/`_sin` only for non-VL
  configs, `ops/rotary_embedding.py:78`), the paste names the branch, and the
  support matrix already reads `MiniCPM-V … A2/A3`: verdict + owner + repair, no
  source walk. Do not derive it from the family or hunt the op-plugin `.cpp`
  (not in any pinned export) — and do not confuse it with the bullet above: a
  traceback means this page, wrong text without one means that page.
- 310P startup dies in the **first forward** (`profile_run`) with
  `EZ3002 … Optype [TransData] of Ops kernel [AIcoreEngine] is unsupported … not found
  in this op store` + `format and dtype is not precisely equivalent … op information
  library` → `atlas-300i-310p.md`, its `TransData`/`EZ3002` section: a dtype/format
  rejection (bf16 first, rope is always ATB on 310P), not the missing-kernel lane of
  `kernel-package-vs-card.md`.
- The op cannot be resolved for the SoC at all (`EZ1001`, `Get regInfo failed`,
  `does not support opType [X] … socVersion [ascend910_93]`) — a build/install
  fault, not a platform path →
  `../custom-ops-runtime-faults/op-not-registered-for-soc.md`. A sibling wording,
  `aclnn<Op> or aclnn<Op>GetWorkspaceSize not in libopapi.so`, is **not** a SoC
  statement either: that macro searches the vendored custom-op package before the
  toolkit, so read the package and `ASCEND_CUSTOM_OPP_PATH` before the vendor's
  per-SoC op table → `../custom-ops-runtime-faults/opapi-symbol-absent.md`.
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
  A further wording in the same section, `[<dir>//<op>.json] is invalid or does not
  exist` in the plog under a console line naming the op (e.g. `ZerosLike` →
  `zeros_like.json`, `aclnnInplaceZero`, v0.14.0rc1 x86 image, #6370), is the
  per-op **file** inside the installed OPS package: **reporter-confirmed fix is a
  reinstall of the CANN OPS package + NNAL**, so say that first on a card
  `faqs.md:13` lists and do not lead with an image-tag change; `EI9999` in such a
  line is HCCL's fixed report code in the export and owns nothing.
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
  `310p-mtp-graph-splitfuse-mask.md` (one `hasattr` line decides it). The same block may also carry
  `Not allow to synchronize captured-stream` + `107027`; it is one refused readback, not the
  `captured-stream-host-sync.md` lane. From v0.23.0 the host qlens half is precomputed while the
  mask half still reads back, so "upgrade rc1 → final" fixes nothing there (#15536).
- A **hybrid linear-attention** family (Qwen3.5/3.6-class GDN) fails on a `-310p` image at the
  first decode step with `AttributeError: 'TritonPlaceholder' object has no attribute 'cdiv'`
  (or `next_power_of_2`) → the image's own "does not support triton" rule meets a decode path
  whose `patch_triton` rebind is `HAS_TRITON`-gated: `atlas-300i-310p.md`, the triton-less
  hybrid section (a Python error before any device op, so no plog line — one grep of the
  traceback settles it).
