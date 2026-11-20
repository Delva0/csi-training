# SoC-conditional code paths (Atlas 300I / 310P first)

Scope: the report is on, or only makes sense on, a specific SoC — most often
**310P** (Atlas 300I Duo, `Ascend310P3`, image suffix `-310P`), where vllm-ascend
branches on `is_310p()` and the documented support is experimental. Also the
general question this topic answers for any SoC: *does the pinned version even
execute the code the traceback is in?* `is_310p()` gates 39 call sites in
vllm-ascend v0.11.0rc3 (`grep -rn "is_310p()" vllm_ascend/ | wc -l`), so a
symptom can be structure rather than a defect.

Inherit the rules in SKILL.md. Rules that decide success here:

1. **Identify the SoC from evidence, not from the image tag alone.**
   `is_310p()` reads the build stamp: `vllm_ascend/utils.py:66-71`
   (`_build_info.__soc_version__.startswith("ascend310p")`). On the reporter's
   host `python -c "from vllm_ascend import _build_info; print(_build_info.__soc_version__)"`
   plus `npu-smi info` settles it; the tag `-310P` in an image name is a strong
   hint but not proof of the build stamp.
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
- The traceback on a 310P host ends in `npu_format_cast` /
  `_convert_torch_format` with a device allocation error →
  `../runtime-memory/kv-init-and-format-cast-alloc.md`.
- The op cannot be resolved for the SoC at all (`EZ1001`, `Get regInfo failed`,
  `does not support opType [X] … socVersion [ascend910_93]`) — a build/install
  fault, not a platform path →
  `../custom-ops-runtime-faults/op-not-registered-for-soc.md`.
- The report is a version gate ("does V support model M on 310P?") →
  `../startup-failures/model-type-not-recognized.md` plus the status lines in
  `atlas-300i-310p.md`.
