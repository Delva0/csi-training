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
  `atlas-300i-310p.md`.
- Engine start dies at `worker.py` `check_ascend_device_type()` before
  `Starting to load model`: `AssertionError: Current device type: AscendDeviceType.A5
  does not match the installed version's device type: AscendDeviceType.A2`, or
  `RuntimeError: Can not support soc_version: <n>` — a wheel built for another SoC
  family (the release pipeline ships a2/a3/310p variants only, `Dockerfile.a5`
  exists in-tree) or a live SoC code the map lacks →
  `device-type-build-vs-runtime.md`.
- The traceback on a 310P host ends in `npu_format_cast` /
  `_convert_torch_format` with a device allocation error →
  `../runtime-memory/kv-init-and-format-cast-alloc.md`.
- The op cannot be resolved for the SoC at all (`EZ1001`, `Get regInfo failed`,
  `does not support opType [X] … socVersion [ascend910_93]`) — a build/install
  fault, not a platform path →
  `../custom-ops-runtime-faults/op-not-registered-for-soc.md`.
- A **built-in** aclnn op fails before launch (`EZ9999 … Parse dynamic kernel
  config fail`, `AclOpKernelInit failed opType`, `<Op>AiCore
  ADD_TO_LAUNCHER_LIST_AICORE failed`) on a card the release's docs exclude, often
  asked as "does image `<tag>` support `<card>`?" → `kernel-package-vs-card.md`
  (image/CANN kernel package vs the live SoC; one standalone op probe).
- The report is a version gate ("does V support model M on 310P?") →
  `../startup-failures/model-type-not-recognized.md` plus the status lines in
  `atlas-300i-310p.md`.
