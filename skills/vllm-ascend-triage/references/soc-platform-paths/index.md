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

Conditional routes:

- 310P / Atlas 300I deployments: the documented support status, the seven
  platform gates that change behaviour, the NZ (FRACTAL_NZ) format path, and the
  launch-environment items the 300I tutorial requires →
  `atlas-300i-310p.md`. That page also carries the per-release blocks a startup
  report needs: at v0.22.1rc1 the `_310p/` subpackage (the older gate line
  numbers are gone), the release's own validated `310p.md` table, the inert
  `fuse_norm_quant` flag and the `max_model_len`-sized mask — quote the block for
  the reporter's release, not the v0.11 one.
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
- A **quantized** model on 310P fails at the first quantized linear with an
  op-level dtype/layout complaint instead of a Python raise — e.g. `Bias dtype
  should be INT32, actual dtype is DT_FLOAT16` in the title alone: the
  checkpoint artifact decides (310-suffixed conversion vs the generic `-w8a8`),
  and the 310P `W8A8_DYNAMIC` scheme is the one path that forwards the model's
  fp16 bias → `310p-quant-checkpoint-soc-match.md` (three checks, no engine walk).
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
- A **hybrid linear-attention** family (Qwen3.5/3.6-class GDN) fails on a `-310p` image at the
  first decode step with `AttributeError: 'TritonPlaceholder' object has no attribute 'cdiv'`
  (or `next_power_of_2`) → the image's own "does not support triton" rule meets a decode path
  whose `patch_triton` rebind is `HAS_TRITON`-gated: `atlas-300i-310p.md`, the triton-less
  hybrid section (a Python error before any device op, so no plog line — one grep of the
  traceback settles it).
