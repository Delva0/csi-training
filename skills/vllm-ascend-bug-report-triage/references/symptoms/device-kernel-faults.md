# Device-kernel faults: trust the DFX log, not the Python traceback

One class, many messages: an aclnn/aclrt failure whose real cause is an AICore kernel. The Python
traceback is a neighbour of the fault, and the owning layer is the CANN / torch_npu operator build,
not the vllm-ascend Python you can read. Worked case: vllm-ascend #17082 — Qwen3.5-9B ModelSlim
`W8A8_DYNAMIC` (`--quantization ascend`), Atlas 300I Duo / 310P3, TP=1, image
`quay.io/ascend/vllm-ascend:v0.23.0rc1-310p` (CANN `9.1.0-beta.1`), vLLM `0.23.0`,
torch_npu `2.10.0.post2`. Weights load, then `EngineCore.determine_available_memory -> profile_run`
fails on every attempt.

## Signature and reading order

1. The device DFX log names a kernel plus its hash and a runtime error, and the hash repeats across
   attempts:

   ```text
   QuantBatchMatmulV3_NZ_NZ_int8_int8_fp16_high_performance_21
   kernel hash: 5247287448945562503
   runtime error: 507015
   error: L0C read/write conflict
   ```

   The same kernel name and hash on every attempt points inside that kernel variant (tiling or a
   pipeline hazard), not at the checkpoint, the memory budget or "W8A8 unsupported" — the reporter
   ruled those out separately (an FP16 checkpoint runs; another W8A8 checkpoint on the same box served
   requests; a third run with a smaller `--max-num-batched-tokens` hit the same hash).
2. The Python traceback may point at `aclnnDynamicQuantV2` or `aclnnRmsNorm` instead. Kernel
   execution is asynchronous, so the reported frame is whichever op next synchronised. Keep the
   device log as the owning evidence and say which frame was the sync point.
   **Use hash stability as the discriminator**: a vllm-ascend shape/layout bug moves the kernel hash
   with the shape, whereas one fixed tiling faulting keeps the hash. Dedupe the per-rank repeats.
3. Read the failing stage literally. `EngineCore.determine_available_memory -> profile_run` means the
   dummy forward died inside profiling; do not reframe it as a KV-cache sizing problem. The reporter
   varied the model path (`--language-model-only`), `--max-num-batched-tokens` (512 too) and
   `--enforce-eager` and got the same kernel each time, so neither the multimodal path, the batch size
   nor eager mode is the lever.
4. `NZ_NZ` in the kernel name is layout information: the dispatcher picked a variant whose operands
   are `ACL_FORMAT_FRACTAL_NZ` (below). That is why config/flag "layout" workarounds do not move this
   failure.

## What vllm-ascend contributes (verified at v0.23.0rc1)

`vllm_ascend/_310p/quantization/methods/w8a8_dynamic.py`, linear `W8A8_DYNAMIC` scheme
(`AscendW8A8DynamicLinearMethod310`, `:167`):

- `apply` (`:184-212`) runs `:193 quantized_x, pertoken_scale = torch_npu.npu_dynamic_quant(x)`, then
  `:202 torch_npu.npu_quant_matmul(quantized_x, layer.weight.data, layer.weight_scale, ...)`
  (`pertoken_scale=`, `bias=`, `output_dtype=x.dtype`).
- `process_weights_after_loading` (`:214-218`) does
  `:216 layer.weight.data = maybe_trans_nz(layer.weight.data).transpose(0, 1)` — the weight is cast
  to `ACL_FORMAT_FRACTAL_NZ` (`utils.py:295 maybe_trans_nz` → `torch_npu.npu_format_cast`), which is
  what selects the `_NZ_NZ_int8_int8_` kernel variant named in the DFX log. Tying that suffix to this
  specific cast is an inference from the code, worth stating as one, but the weight is certainly NZ on
  this path. The MoE counterpart is `AscendW8A8DynamicFusedMoEMethod310` (`:158-159`).
- The base scheme (`_310p/quantization/methods/w8a8_base.py`) only allocates `weight` (int8),
  `input_scale` / `input_offset`, `quant_bias` (int32) and `deq_scale` (int64); the dynamic variant
  replaces those with per-channel `weight_scale` / `weight_offset` (fp32).
- There is **no switch to turn the NZ cast off on 310P**: `_should_trans_nz` (`utils.py:261`) returns
  True **before** reading `ascend_config.weight_nz_mode` when `is_310p()` (`:271-272`). Do not offer
  `VLLM_ASCEND_ENABLE_NZ=0` / `additional_config.weight_nz_mode=0` as a workaround here; on A2/A3 the
  same flag *is* honoured, which is why #11624 could disable NZ on a 910B3
  (`../vllm-ascend/config-flag-lifetimes.md` §7).
- The policy survives later refactors, expressed differently: from `v0.27.1rc1` the same early return
  is `get_current_hardware_profile().weight_layout_policy is WeightLayoutPolicy.FORCE_NZ`
  (`utils.py:252` def, `~:262`; `:350`/`:368` at `main`), backed by the device table where
  `AscendDeviceType._310P` sets `FORCE_NZ` while A2/A3 set `CONFIGURABLE`
  (`device/hardware_profile.py:322` vs `:294`/`:306` at main). So "310P ignores the NZ switch" is
  checkable at any ref by grepping `weight_layout_policy`. `main` adds a further guard: weights whose
  reduction or output dim is 1 stay ND (`utils.py:362-365`), because `aclnnMatmulWeightNZ` cannot
  handle them.
- The same file admits a path-level defect in a comment: `NOTE(310P): There is an accuracy issue
  currently, which is expected to be fixed in the next version` (`:191-192` at `v0.23.0rc1`). Quote it
  as a verified comment about the path, not as the cause of a kernel runtime error.
- Reconcile the image before blaming a version: `Dockerfile.310p:18` at `v0.23.0rc1` is
  `FROM quay.io/ascend/cann:9.1.0-beta.1-310p-ubuntu22.04-py3.12`, matching the reported CANN, and
  the tag pins `torch-npu 2.10.0.post2`, matching the report. The tag's Python is therefore the code
  that ran; no version archaeology is needed beyond this check.

## The gate is the CANN/operator build, not the vllm-ascend tree

- Maintainer comment (`reported`, Tflowers-0129, 2026-09-23): 请升级CANN版本，9.1.0商发版已修复 —
  "upgrade CANN; the 9.1.0 commercial release has fixed it". Attributed, and consistent with the
  reporter's own note that #10122 reports the same kernel name and hash with a CANN-side fix in
  `9.1.0.beta2`. Nothing in the vllm-ascend tree can switch this kernel off.
- Reporter retest (2026-09-21, `reported`): `nightly-main-310p`, CANN `9.1.0`
  (`V100R001C11SPC001B243`), vLLM `0.29.0+empty`, vllm-ascend `c173a64a4`,
  `VLLM_USE_V2_MODEL_RUNNER=1` → `NPUModelRunner310V2`: startup profile, health check and greedy
  generation passed, plus an 8-case benchmark, no `507015`, same checkpoint revision. **Confounded**:
  CANN and the model runner both changed; the reporter says the isolation is unresolved and leaves the
  V1 path open. The issue is closed on that basis.
- Their controls on the same server family: FP16 TP=1 ran; other W8A8 checkpoints ran at other
  shapes/TP — consistent with a shape/tiling-specific kernel fault, not a damaged checkpoint.

## The model-runner question: check the ref before promising MRv2

- `vllm_ascend/_310p/worker/v2/` (home of `NPUModelRunner310V2`) exists from **`v0.27.1rc1`** onward
  (`75133ed73`, #15006, 2026-08-31); it is absent at `v0.23.0`, `v0.24.0rc1`, `v0.25.1rc1`,
  `v0.26.0rc1` and `v0.26.0rc2`. Default-on for 310P only from `7abb7cfbf` (#18995, "Default Model
  Runner V2 on Ascend 310P", main, 2026-10-19), which is in **no** tag.
- At `v0.23.0rc1`, `VLLM_USE_V2_MODEL_RUNNER` exists but only flips **attention-backend selection**
  (`attention/attention_v1.py:86`, `mla_v1.py:88`, `sfa_v1.py:135`, `fa3_v1.py:18`, `dsa_v1.py:199`
  return `CUSTOM`/`ASCEND_*` when unset and `FLASH_ATTN` when set), and `_310p/worker_310p.py`
  instantiates `NPUModelRunner310` unconditionally. "Set `VLLM_USE_V2_MODEL_RUNNER=1` on
  `v0.23.0rc1-310p`" is therefore **not an available option**: name the first tag that has the
  runner.
- Log discriminator: `Using NPUWorker310 and NPUModelRunner310V2` versus `...NPUModelRunner310`
  (`_310p/worker_310p.py:60`/`:66` at main). `mrv2_utils.py:111-142` (main) states V2 is the default
  and `VLLM_USE_V2_MODEL_RUNNER=0` forces V1.

## Workaround

1. Move the image to a CANN base of 9.1.0 GA or later. This is the only change that touches the
   faulty kernel and is the maintainer's answer. Verify the CANN version inside the image
   (`ASCEND_HOME_PATH`, `ASCEND_OPP_PATH`, the `collect_env` CANN block), not the image tag name.
2. On a build ≥ `v0.27.1rc1`, run the 310P MRv2 runner (`VLLM_USE_V2_MODEL_RUNNER=1`) — reporter-
   verified only as a bundle with (1), so do not attribute the fix to the runner alone.
3. Serve the FP16 checkpoint while the fused W8A8 kernel is unfixed (the reporter's own control:
   FP16 Qwen3.5-9B TP=1 works). Trades memory (~2×) and throughput.
4. Not workarounds: `--gpu-memory-utilization`, `--max-model-len`, `--max-num-batched-tokens`,
   `--enforce-eager`, `--dtype`, `weight_nz_mode` / `VLLM_ASCEND_ENABLE_NZ` (ignored on 310P). And
   check first whether `--quantization ascend` is even set: an unquantised build of a quantised
   checkpoint produces the opposite failure (`quantization-load.md` §3).

## Where not to look

- The `cann/*` clones are not where built-in ops live: a whole-tree search for `QuantBatchMatmulV3`
  across `/workspace/repos/cann/*` hits only incidental mentions (oam-tools scripts, graph-autofusion
  op lists), because the kernel and its tiling ship as a compiled CANN package identified by hash.
  Do not spend calls grepping those repos for the kernel; the owner is the CANN release, cited by
  version. Operator-tiling fixes live in CANN/`ops-nn`/`ops-math`, not in vllm-ascend.
- Conversely, the vllm-ascend side does hold the useful half: which layout and op the quantization
  scheme feeds (`_310p/quantization/methods/`, the `utils.py` NZ policy) and the image's CANN pin.

## Sibling class: "op type <Op> is not found in this op store"

The same owning layer, a different message shape: no kernel hash, no runtime error code, and the
complaint is about the **operator store**, not a kernel variant.

```text
Optype [TransData] of Ops kernel [AIcoreEngine] is unsupported.
Reason: [tbe-custom]:op type TransData is not found in this op store.
[Dynamic shape check]: The format and dtype is not precisely equivalent to ...
```

Worked case: vllm-ascend #1383 (`v0.9.1rc1`, Atlas 300I Duo / 310P3), 0 comments, closed. The visible
failure is the wrapper pair only — `RuntimeError: Engine process failed to start` plus `ERR99999
UNKNOWN applicaiton exception` — so the report looks like a startup bug with no traceback; the decisive
text is the `[tbe-custom]` line above it.

Reading it:

- `not found in this op store` means the operator package the runtime loaded does not contain that
  op **for the installed SoC**. That is an image/CANN/op-plugin build question (a build for the wrong
  chip, a missing 310P op store, a mismatched `torch_npu`), not a shape or dtype bug in vllm-ascend.
  Keep both halves: which `torch_npu`/ATB op the Python layer selected, and which image/CANN ships it.
- The dispatch half differs from the kernel-hash class above: here the Python does reach a real op, and
  the op resolution fails inside `torch_npu`/ATB. Read the deepest frame of the report to see whether it
  is `op_plugin/atb/*`, an aclnn wrapper or `torch.ops._C*`, and check whether the ref even has a
  `vllm_ascend/_310p/` tree (first `v0.14.0rc1`, #5776) — before that tag the 310P deliverable is the
  separate `-310p` image, not a code path (`../vllm-ascend/code-layout.md`).
- The verified chain for #1383, the two rope dispatch paths at `v0.9.1rc1` and the reason no fixing
  commit can be named: `graph-capture-and-custom-ops.md` §B. Route a graph-capture *trace* failure
  (`attempted to run this operator with Meta`) to §A of that page instead — nothing runs on the NPU
  there.
- Do not grep the `cann/*` clones for the missing op: built-in and custom operator packages are
  compiled artefacts (same rule as the kernel hash above).

Falsifier: if the same build serves a different model, or the same model runs after only an image
change that keeps CANN/`torch_npu` fixed, this class is wrong and the fault is in the selected path.

## Sibling class: `call aclnn<Op> failed … AclOpKernelInit failed opType <Op>` — the image's CANN base is another SoC's build

The same owning layer again, with no kernel hash and no runtime error code: the operator package is
present, but it was built for a different chip, so `aclnn` cannot initialise the op at all.

```text
RuntimeError: call aclnnArange failed, detail:EZ9999: Inner Error!
EZ9999: [PID: 2075] 2025-07-09-06:39:54.780.150 Parse dynamic kernel config fail.
        TraceBack (most recent call last):
       AclOpKernelInit failed opType
       ArangeAiCore ADD_TO_LAUNCHER_LIST_AICORE failed.
```

Worked case: vllm-ascend #1696 — `quay.io/ascend/vllm-ascend:v0.7.3`, card `910C1` in `npu-smi`,
CANN `8.1.RC1`, dying during `LLM("Qwen/Qwen2.5-0.5B-Instruct")` construction. The reporter's actual
question was "does this image support 910C?" — it does not, and nothing in the script is wrong.

Reading it:

- `AclOpKernelInit failed` / `Parse dynamic kernel config fail` / `ADD_TO_LAUNCHER_LIST_AICORE failed`
  is CANN's kernel-selection layer refusing: the kernel package in the image carries no tiling config
  for this SoC. Property of the **image's CANN base**, not of the model, the request, `mindie_turbo` or
  the `aclnnArange` call site.
- Answer it from the tag's Dockerfiles. The default `Dockerfile` is the **A2 (`910b`)** build; each other
  SoC has its own file and base: at `v0.18.0` `Dockerfile` `cann:8.5.1-910b-…`, `Dockerfile.310p`
  `…-310p-…`, `Dockerfile.a3` `…-a3-…`; `v0.23.0` adds `Dockerfile.a5` `…-950-…`. `Dockerfile.a3`
  first exists at `v0.10.0rc1`, `Dockerfile.a5` at `v0.23.0`; `v0.7.3` ships only
  `Dockerfile`/`Dockerfile.openEuler`, both `FROM quay.io/ascend/cann:8.1.rc1-910b-ubuntu22.04-py3.10`.
  So the published v0.7.3 image holds A2 kernels only, and a 910C (A3) card has none for any op.
- The install page at that tag agrees: `docs/source/installation.md:9` "usually the Atlas 800 A2
  series", `:87-89` installs `Ascend-cann-kernels-910b_8.1.RC1_linux-$(uname -i).run`.
- The maintainer answer is the lever (#1696, Yikun 2025-07-10): "The A3 image haven't supported yet …
  replace the base image with `quay.io/ascend/cann:8.1.rc1-a3-ubuntu22.04-py3.11`" — i.e. rebuild from
  source with that base. Cost: the a3 base is py3.11 against the tag's py3.10, so every wheel
  (`torch-npu==2.5.1`, torch, torchvision) must be the cp311 build; the cheaper route is a later image
  whose release already publishes an A3 target (`Dockerfile.a3`).
- `910C` is only the card name in `npu-smi`; the platform is A3 (Atlas 800T/900 A3 family). No
  `vllm_ascend/` code selects a per-SoC kernel set, so do not hunt a project frame — and do not read the
  failure as "the model/op is unsupported": the same script on an A2 card works.
- Not causes here, whatever the report suggests: `mindie_turbo`, the `example.py` body, `--device` /
  `ASCEND_RT_VISIBLE_DEVICES`, and the CANN version string alone (the same `8.1.RC1` works on A2). Do not
  grep the `cann/*` clones for `Arange` either — the kernel package is a compiled artefact.
- The symptom-side reading is this section; the per-ref `FROM` inventory, the A3/A5 file history and the
  user-visible levers are §1 of `../vllm-ascend/install-and-image.md`.

Falsifier: the same image and script succeeding on an `910B` card, plus the same 910C card working with
an `-a3-` CANN base at the same CANN version, isolates the kernel package from the code. Keep CANN
version and SoC support apart — a newer CANN on a `910b` base is still an A2 kernel set.

## Check before reuse

```bash
git -C /workspace/repos/vllm-project/vllm-ascend show <tag>:vllm_ascend/_310p/quantization/methods/w8a8_dynamic.py \
  | grep -n 'npu_dynamic_quant\|npu_quant_matmul\|maybe_trans_nz'
git -C /workspace/repos/vllm-project/vllm-ascend show <tag>:vllm_ascend/utils.py | sed -n '/def _should_trans_nz/,+18p'
git -C /workspace/repos/vllm-project/vllm-ascend ls-tree --name-only <tag>:vllm_ascend/_310p/worker/v2   # MRv2 availability
git -C /workspace/repos/vllm-project/vllm-ascend grep -n 'weight_layout_policy' <tag> -- vllm_ascend
git -C /workspace/repos/vllm-project/vllm-ascend show <tag>:Dockerfile.310p | grep -n '^FROM'
git -C /workspace/repos/vllm-project/vllm-ascend ls-tree --name-only <tag>: | grep -i docker   # is there an a3/a5 target at all?
git -C /workspace/repos/vllm-project/vllm-ascend show <tag>:Dockerfile.a3 2>/dev/null | grep -n '^FROM'
```

The kernel name, hash and the `_310p/quantization/methods/` layout are tag-specific; the 310P scheme
directory and the worker file (`_310p/worker_310p.py` at v0.23.0rc1 → `_310p/worker/` later) move
between releases. Falsifier for "device kernel, not Python": if the DFX log names a different
kernel/hash, or there is no DFX line at all, this class does not apply — fall back to the traceback
and `quantization-load.md`.

## Sibling family (the aivec sibling): a bare `DDR address of the MTE instruction is out of range`

The same owning layer as this page, with a message that names **no** kernel and usually arrives as a
screenshot of the log. It is a family rather than one class — 57 issues in the fork carry that exact
string (GitHub search, 2026-11-17) — and two decisions route it before any code is read.

- **Was a graph captured?** `FULL_DECODE_ONLY`, or a PD decode node's dummy batch, is the
  padded-metadata class in `aclgraph-mode-and-capture.md` §G (#7290/#7460/#7401) — go there. A run whose
  argv carries `--enforce-eager` and no PD never executes that path, so §G's mechanism is not available
  there at all; do not stretch it. The `rtMemcpy … the current capture mode does not support this
  operation` sibling (#7448) also needs a capture.
- **Does the pasted log carry a kernel name, or only `aivec error` / `EL0004`?** The localising payload
  looks like this (#3263, v0.7.3): `errorStr: The DDR address of the MTE instruction is out of range …
  AICORE Kernel task happen error, retCode=0x26 … fault kernel_name=paged_attention_mask_1_mix_aic,
  hash=7917278553613333893`, then `synchronize stream failed, runtime result = 507053`. With that line
  the owner is a compiled CANN kernel ("Where not to look" above); without it the message localises
  nothing in the reported ref's tree. (The fork *can* raise it, and the string survives only as two
  comments at `main`: `patch/platform/patch_example_connector.py:25` — the upstream EC-cache loader pins no
  device index, Ascend's `is_cuda_alike()` is False, so in TP>1 every rank loads EC cache to `npu:0` and
  the fault is `errcode=95 (MTE DDR address out of range)`; fix #18284, first tag `v0.30.0rc1` — and
  `ops/triton/dspark_swa_indices.py:59`, a floor division putting tail items past `num_slots`. Check those
  two only when the report runs an encoder-cache connector in TP>1 or DSpark SWA on such a build.)

**What the reply ships when there is no kernel name**: the plog request, one attributed lever, and nothing
else.

- The ask, in the wording the maintainers use: `/root/ascend/log/debug/plog/plog-<pid>_<date>.log`
  (`$ASCEND_PROCESS_LOG_PATH` when set), grep
  `fault kernel_name|Aicore kernel execute failed|507015|507053`. `leo-pony` in #4665 — essentially the
  same environment as #3911 (vLLM `0.11.0rc3` + vllm-ascend `0.11.0rc0`, Qwen3-VL-235B-A22B, transformers
  `4.57.3` against `4.57.0`), closed stale by the same bot flow: "Could you upload the plog file in
  /root/ascend/log/debug/plog/…? as from the above error log we can not find the root cause";
  `Dawn952` opens #11859 with 「请提供Plog日志」.
- `--enforce-eager` is already in #3911's argv, so it is not a lever for that report, and
  `--max-num-batched-tokens` / `--max-model-len` only matter once a plog shows a shape-dependent kernel.
- The release lever is the thread's whole content: `earthmanylf` 2026-05-28 「use the latest release or rc
  version … If the issue persists, feel free to submit a new issue」, then a stale-bot close on 2026-08-11.
  Relay it *attributed* as the workaround; that comment names no tag, so do not claim a fix from it
  (`../workflow/fix-discovery.md`).
- A VL model's own multi-node tutorial is **not** the lever here: #3911's argv (2 nodes × 8×910B3, DP2
  with `--api-server-count 2 --data-parallel-size-local 1`, TP8, `--enable-expert-parallel`,
  `--max-model-len 32768 --max-num-batched-tokens 4096 --no-enable-prefix-caching`, `VLLM_LOGGING_LEVEL=DEBUG`
  already on) matches `docs/source/tutorials/multi_node_qwen3vl.md` at `v0.11.0rc0` except
  `--max-num-seqs 2`, `--gpu-memory-utilization 0.75` and `--enforce-eager`, leaving nothing for the
  doc-vs-capability branch to quote.
- What that report already rules out, so the next run does not re-walk it: HBM ("HBM is not exhausted by
  that time"), a graph capture (`--enforce-eager`), the notebook/engine configuration (healthy until the
  benchmark's case 146 of MMMU_Pro_V, evalscope + vlmevalkit client), and the log (only three
  screenshots; nothing to grep). The missing artifact is the plog.

Recorded wrong turn: a run spent four tool batches on #3911 — grep-ing `vllm_ascend/` for that string and
reading that tutorial — and shipped nothing. The page it needed was this section, one row away from the §G
route it took.
