# Version / implementation correspondence

Purpose: decide whether a surprising startup failure is a genuine defect, a
missing/extra feature registration, or a configuration that differs from a
known-good path — before writing a theory about model code.

## Build the table first

1. From the evidence capsule: `/workspace/repos/SOURCES.json` lists
   `repo`/`ref`/`commit`/`basis` (and coverage limits per repo) for the versions
   frozen at issue opening. These are the only versions you may cite; do not
   substitute current knowledge for a `null` runtime verification.
2. From the reporter: image tag, `vllm` and `vllm-ascend` versions, `torch` /
   `torch_npu`, `ascend-toolkit`, `nnal`, `ASCEND driver`, plus the full command.
   Where the pair is actually recorded (two files, not `requirements.txt`):
   `docs/source/community/versioning_policy.md` release compatibility matrix, and
   the `.github/workflows/pr_test_*.yaml` `vllm_version` matrix. At v0.14.0rc1 the
   matrix row is `v0.14.1rc1 | v0.14.0/v0.14.1`
   (`docs/source/community/versioning_policy.md:26`) and CI pins
   `[d68209402ddab3f54a09bc1f4de9a9495a283b60, v0.14.1]`
   (`.github/workflows/pr_test_full.yaml:78`). `requirements.txt` does not pin
   vllm at that commit (its only `vllm` mention is a comment at `:19`), so a grep
   for `vllm==` there returns nothing.
   **The snapshot's vllm pin can be older than the image's.** Recorded #6784:
   `SOURCES.json` pinned vllm `v0.13.0` while the image is vllm-ascend
   `v0.14.0rc1`, which pairs with vllm `v0.14.0/v0.14.1`. A traceback frame
   naming a vllm file absent from the pinned tree is therefore a
   snapshot-coverage gap, not evidence about upstream — say so, ask for
   `pip show vllm`, and do not hunt for the missing line.
3. Recorded pairs:
   - vllm-ascend `v0.11.0rc0` (`00ba07102212c7c7a40de427f09848f2e203c498`) with
     vllm `v0.11.0rc3` (`8ce5d3198d00631a76e1aa02a57947b46bc7218c`), image
     `quay.io/ascend/vllm-ascend:v0.11.0rc0`.
   - vllm-ascend `v0.17.0rc1` (`e20f0b1a...`) with vllm `0.17.0+empty`,
     `torch`/`torch_npu` 2.9.0, driver 25.2.0, `ascend-toolkit`/`nnal` 8.5.1.

## Check "is model X supported here" in the source, once

- Registration: `vllm_ascend/models/__init__.py` `register_model()` is the
  authoritative list of Ascend overrides. In v0.11.0rc0 it registers
  `Qwen3VLMoeForConditionalGeneration` and `Qwen3VLForConditionalGeneration`
  (both to `vllm_ascend.models.qwen2_5_vl_without_padding`), `Qwen2VL...`,
  `Qwen2_5_VL...`, DeepSeek V2/V3/V3.2, `Qwen3MoeForCausalLM`,
  `Qwen3NextForCausalLM`, `PanguProMoEForCausalLM`. A VL key registered **twice**
  under an env gate (Qwen2.5-VL: `USE_OPTIMIZED_MODEL` picks `qwen2_5_vl` vs
  `qwen2_5_vl_without_padding` at v0.9.2rc1) means the flag is itself a
  known-good-baseline row — that is the first ablation for a VL accuracy report,
  see `../accuracy-benchmarks/vl-vision-tower-degeneracy.md`.
- Upstream files: vllm v0.11.0rc3 ships `vllm/model_executor/models/qwen3_vl.py`
  and `qwen3_vl_moe.py`. So "Qwen3-VL unsupported in this image" is **false**
  for that pair — verify before suggesting it.
- Release notes and `docs/source/user_guide/support_matrix/supported_models.md`
  state what the release claims (v0.11.0rc0 claims Qwen3-VL support, PR #3103).
- A model name newer than the image (e.g. a `Qwen3.5` path against a snapshot
  whose vllm has `qwen3_5.py`) means: check both sides' model/config files exist
  at the pinned commits, then rely on CI tests
  (`tests/e2e/multicard/4-cards/test_qwen3_5.py` for TP=4 Qwen3.5-27B in
  v0.17.0rc1) as the known-good baseline.

## Read the environment dump's gaps before its contents

An empty `CANN:` field (or a missing component version) in the issue's
`collect_env.py` block is usually the *collector*, not the install:
`collect_env.py` reads hard-coded default roots (`collect_env.py:300` at
v0.20.2rc1 reads
`/usr/local/Ascend/ascend-toolkit/latest/<arch>-linux/ascend_toolkit_install.info`),
so an install under `/usr/local/Ascend/cann-9.0.0` or a conda prefix reports
`None`. Reporter-confirmed on #10525: the path existed, "但这个检查脚本确实没检查到".
Cross-check `ASCEND_HOME_PATH`, `python3 -c "import acl; print(acl.get_soc_name())"`
and `which npu-smi`, and ask for the actual install root before building a
thesis on a missing component. The same rule applies to components the image
installs but a hand-built environment must add itself (NNAL/ATB:
`../image-dependency-faults/nnal-atb-missing-library.md`).

## Diff the reporter's configuration against a known-good baseline

CI e2e tests and the pinned vllm-ascend `docs/source/tutorials/*.md` are the shipped known-good command
lines; they carry env vars (`HCCL_BUFFSIZE`, `TASK_QUEUE_ENABLE`,
`PYTORCH_NPU_ALLOC_CONF`, `ASCEND_RT_VISIBLE_DEVICES`) and flags
(`--max-model-len`, `--compilation-config`, `--async-scheduling`,
`--additional-config`) the reporter may have added or dropped. Enumerate the
differences, then rank them by whether they can plausibly produce the observed
error class. In the recorded #7477 case the config matched the CI shape closely
and the failure turned out to be hardware; in #3543 the single missing flag was
decisive.

## Reporting a non-reproduction

Say exactly what was compared: same vllm/vllm-ascend pair, NPU model and per-card
HBM, driver/CANN, command line. A clean run on different hardware is a useful
data point, not a refutation of a reporter-confirmed fix on theirs.
