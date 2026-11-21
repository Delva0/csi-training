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

## A `vllm_ascend/**` frame the pin does not contain: vendored file, not a gap

The mirror of the rule above. There the missing file is a **snapshot** limitation
and the finding is a data gap; here the missing file is **the reporter's own
code**, and the finding is that the release does not own the failing path.

- Trigger: the paste's `file:line`, or a traceback frame, points inside
  `vllm_ascend/models/<x>.py`, `vllm_ascend/ops/<x>.py` etc. and
  `ls vllm_ascend/models/` at the pin has no such file. One call each:
  `ls <repo>/vllm_ascend/models/`, `rg -ln -i '<model-or-symbol>' vllm_ascend/`,
  `rg -n '<Class>' vllm_ascend/models/__init__.py`.
- Discrimination: an in-tree file whose *line numbers* shifted is version drift
  (re-grep, cite the pin); a file that does not exist at the pin while the same
  module is mentioned only in a comment or a config docstring is a **site-vendored
  file**. Recorded #3107 at vllm-ascend v0.9.1rc2 (`b9f715d`): the log names
  `vllm_ascend/models/pangu_moe.py` at lines 526-877, the pin's
  `vllm_ascend/models/` has no `pangu_moe.py`, `models/__init__.py` registers no
  Pangu class, and the only in-tree mention of the family is the torchair comment
  at `vllm_ascend/ascend_config.py:78`. By v0.11.0rc0 `PanguProMoEForCausalLM`
  **is** registered (see the section below), so the out-of-tree file is a
  pre-upstream port, and "drop it and use the release's class" is a real rung.
- What the answer may still claim: the framework functions that file calls
  (`set_ascend_forward_context`, `ops/fused_moe.py`, the attention metadata
  builder) are in-tree, so their behaviour, their fallbacks and their
  `path:line` are citable — say explicitly that the model arithmetic is not.
- What to ask for: the file itself with its line numbers, its registration patch
  (`ModelRegistry.register_model` / `register_oot`), and whether the same failure
  reproduces with the release's own class. Do not reconstruct their module from
  the log's debug prints; the printed shapes and values are their instrumentation,
  not the framework's.

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

## The launch environment is one environment per worker process

On `--distributed_executor_backend ray` (and any multi-node launch) every worker is
a separate interpreter on its own node, started by that node's raylet: a `source
set_env.sh` or `export …` in the driver's shell does **not** reach it. Requirements
that are per-process therefore have to be checked inside each node's worker, not in
the shell that submitted the job:

```bash
pip show vllm vllm-ascend | grep -E 'Name|Version'
python -c "import triton; print(triton.__version__, triton.__file__)"
which bishengir-compile
```

Run it once per node (or carry the values through ray `runtime_env` `env_vars` /
`worker_process_setup_hook`), and ask the reporter for that output rather than for
"the versions".

Before the per-worker environment, check the cluster's resource *view*: a paste that
carries `Current node has no NPU available … 'GPU': 8.0` (or the preceding `The
number of required NPUs exceeds …` warning) never reached a worker at all — the
nodes advertise `GPU` where the platform asks for `NPU`, and one `ray status` per
node settles it: `ray-npu-resource-label.md`.

Recorded #3098 (vllm-ascend v0.10.2rc1, ray with 2 nodes x 8 910B4,
Qwen3-Next-80B-A3B): that model imports Triton at module import
(`vllm_ascend/ops/fla.py:11-12`, `ops/casual_conv1d.py:14-15`), and the release's
own tutorial makes triton-ascend plus the BiSheng toolkit a manual, **aarch64-only**
install (`docs/source/tutorials/multi_npu_qwen3_next.md:41-68`, with
`source /usr/local/Ascend/8.3.RC1/bisheng_toolkit/set_env.sh` at `:54` and repeated
at `:75-78` as a precondition for inference; the x86_64 tab is "Coming soon" at
`:67-68`). In a multi-node launch that install and that `source` have to exist in
every node's worker, and a node that skipped either fails alone — which is exactly
the "model X deploys here, model Y 报错" split of that report, and why the pair of
models is not the first evidence. The same release note
(`docs/source/user_guide/release_notes.md:9`) says EP and MTP "doesn't work with
this release" (Qwen3 Next, PR #2917), so those flags are ablation rows there, and
its `:89` command (`--tensor-parallel-size 4 --max-model-len 4096
--gpu-memory-utilization 0.7 --enforce-eager`) is the only validated shape — a
cross-node TP16 launch has no published recipe at that ref. For the wheel/compiler
pairing itself, `../image-dependency-faults/triton-ascend-pair.md`.

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
