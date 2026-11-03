# Qwen3-Next-80B-A3B reports: hybrid GDN + MoE start-up and steady-state classes

Scope: reports naming `Qwen3-Next-80B-A3B-Instruct` / `-Thinking` (`Qwen3NextForCausalLM`) on the
0.10-0.13 plugin line. The recorded forwards on this model (two at `v0.10.2rc1`, one at `v0.11.0`,
one at `v0.13.0rc2`) fall into three signatures below plus one model-independent one: the
`v0.13.0rc2` start-up is the torch-npu launcher-stub compile failure, which is not a hybrid-model
defect (section 2 of [deepseek-v3.2.md](deepseek-v3.2.md)). Classify from the log's own last error,
not from the model name. Everything below was read from tag-pinned `vllm-project/vllm-ascend` and
`vllm-project/vllm` sources plus the recorded threads; no NPU run was made, so runtime-only claims
are labelled.

## What the architecture settles before reading a log

Qwen3-Next is a **hybrid** model: MoE feed-forward plus linear attention (GDN) in three of every
four layers, full attention in the fourth. The split is hardcoded in the model's own config —
`vllm/transformers_utils/configs/qwen3_next.py:251` at vLLM `v0.11.0` and still at `:253` in
`v0.18.0`, `"linear_attention" if bool((i + 1) % 4) else "full_attention"`; the
`full_attention_interval` field belongs to the Qwen3.5 config, not this one. Consequences to keep in
mind when a report hedges between classes:

- Hybrid detection changes behaviour at start-up and prints it (recorded v0.11.0 log):
  `config.py:297 Hybrid or mamba-based model detected: disabling prefix caching since it is not yet
  supported` and `config.py:308 ... setting cudagraph mode to FULL_AND_PIECEWISE in order to
  optimize performance`. A capture-mode question is then the hybrid capability downgrade
  ([full-decode-only-graph-mode.md](full-decode-only-graph-mode.md)), and a "prefix cache never
  hits" observation is the disabled line, not a cache defect
  ([prefix-cache-and-chunked-prefill.md](prefix-cache-and-chunked-prefill.md)).
- The linear-attention layers own per-request state, and the plugin profiles it together with the
  dummy forward; that is the connection to the memory-profile class in section 2.
- On the 0.10 line the model module imports the plugin's own Triton kernels at module scope, which
  is section 1. Do not carry that requirement to 0.11+ (boundary in section 1).

Attention/MoE kernels, MTP drafter metadata and acceptance belong to
[gdn-mtp-spec-decode.md](gdn-mtp-spec-decode.md) and
[spec-decode-metadata-failures.md](spec-decode-metadata-failures.md).

## 1. 0.10-line start-up: `No module named 'triton'` during model-architecture inspection

Recorded signature (image `quay.io/ascend/vllm-ascend:v0.10.2rc1`, Qwen3-Next-80B-A3B-Instruct,
910B4 x8, TP8, CANN 8.2.RC1, torch_npu 2.7.1.dev20250724, paired vLLM 0.10.2):

```text
INFO  [importing.py:63] Triton not installed or not compatible; certain GPU-related functions will not be available.
ERROR [registry.py:449] Error in inspecting model architecture 'Qwen3NextForCausalLM'
  ... RuntimeError: Error raised in subprocess:
  File "/vllm-workspace/vllm-ascend/vllm_ascend/ops/casual_conv1d.py", line 14, in <module>
    import triton
ModuleNotFoundError: No module named 'triton'
pydantic_core._pydantic_core.ValidationError: 1 validation error for ModelConfig
  Value error, Model architectures ['Qwen3NextForCausalLM'] failed to be inspected.
```

- **Read the subprocess traceback, not the pydantic error.** `failed to be inspected` only says the
  registry's inspection subprocess died; any module-level import failure of the model module lands
  there, so the last `File ...` inside the subprocess output is the finding.
- Mechanism at `v0.10.2rc1`, verified: `vllm_ascend/models/qwen3_next.py:65` imports
  `vllm_ascend.ops.casual_conv1d` and `:67` imports `vllm_ascend.ops.fla`, and both files
  bare-`import triton` (`casual_conv1d.py:14`, `fla.py:11`) with no guard and no
  `vllm.triton_utils` wrapper. Importing the model class therefore requires the `triton` module
  before any kernel runs. The misspelling `casual_conv1d.py` is the tag's real filename.
- The image removes triton on purpose, so this is an environment step, not a broken image or a
  plugin defect. `Dockerfile:42-44` at `v0.10.2rc1` explains it ("In x86, triton will be installed
  by vllm. But in Ascend, triton doesn't work correctly. we need to uninstall it") and runs
  `python3 -m pip uninstall -y triton`; `requirements.txt` pins none. The requirement lives in the
  tag's own model tutorial, which is the recipe to quote:
  `docs/source/tutorials/multi_npu_qwen3_next.md:41-61` at `v0.10.2rc1` — "The Triton Ascend is
  required when you run Qwen3 Next", install the Ascend BiSheng toolkit
  (`Ascend-BiSheng-toolkit_aarch64.run`, then `source /usr/local/Ascend/8.3.RC1/bisheng_toolkit/set_env.sh`)
  and then the pinned wheel
  `triton_ascend-3.2.0.dev20250914-cp311-cp311-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl`.
  The `triton` module comes from `triton-ascend`, not from `pip install triton`. The image-side
  timeline (which tags uninstall, which first ship triton-ascend, the `HAS_TRITON` gates) is in
  [image-and-build-reports.md](image-and-build-reports.md).
- Thread evidence (issue #3029, closed): maintainer "need install triton-ascend and bisheng first";
  a reporter who installed a newer `triton-ascend` hit
  `ImportError: cannot import name 'OutOfResources' from 'triton.triton_patch.runtime.autotuner'`
  (raised through `torch._inductor.runtime.triton_compat`, which aborts `import torch` itself when
  `TORCH_DEVICE_BACKEND_AUTOLOAD=1`) and resolved it "install a historical version" — i.e. the
  tutorial's pinned build. Another reply confirms
  `triton_ascend-3.2.0.dev20250914` and points at the same tutorial page.
- Boundary, verified: the inspection failure is a **0.10-line** signature. From `v0.11.0rc1` /
  `v0.11.0`, `vllm_ascend/models/qwen3_next.py` imports upstream
  `vllm.model_executor.layers.mamba.ops.causal_conv1d` and `vllm.model_executor.layers.fla.ops`,
  which import through `vllm.triton_utils` (`vllm/triton_utils/importing.py` at v0.11.0: `HAS_TRITON`
  from `find_spec`, else a `TritonPlaceholder` whose `jit`/`autotune`/`heuristics` are no-ops), so a
  missing triton no longer fails model inspection. `d5f77f14d` (#4420, "mkdir triton package and
  move triton files") first ships in `v0.12.0rc1` and is a later relocation, not this fix. Whether
  the post-0.11 runtime still needs triton-ascend for these kernels was not tested here — say so
  instead of promising that an upgrade removes the requirement.

## 2. v0.11.0 start-up: `Available memory: 0` then `No available memory for the cache blocks`

Recorded signature (image `vllm-ascend:v0.11.0`, vLLM 0.11.0, TP8, 910B4, `--gpu-memory-utilization
0.5`, no `--max-model-len` so the model's 262144 is used):

```text
INFO [worker_v1.py:259] Available memory: 0, total memory: 65452113920        (every TP rank)
ERROR [core.py:708]     raise ValueError("No available memory for the cache blocks. "
ValueError: No available memory for the cache blocks. Try increasing `gpu_memory_utilization` ...
```

- This is the profiling budget, not a proof that the cards are too small. At `v0.11.0`
  `vllm_ascend/worker/worker_v1.py:219-259` computes
  `available_kv_cache_memory = max(total_npu_memory * gpu_memory_utilization - peak_memory, 0)`,
  where `peak_memory = torch_npu.npu.memory_stats()["allocated_bytes.all.peak"]` plus the
  non-torch part (`mem_get_info` used minus `allocated_bytes.all.current`), and prints it. The `0`
  is that clamp, and the upstream caller `check_enough_kv_cache_memory`
  (`vllm/v1/core/kv_cache_utils.py:678`, raises at `:698` at vLLM `v0.11.0`) raises as soon as it
  receives `<= 0`.
- `profile_run` at `v0.11.0` (`model_runner_v1.py:2567`) runs `_dummy_run(self.max_num_tokens,
  with_prefill=True)` and, when `max_num_tokens > mc2_tokens_capacity` and the MoE comm method for
  that size is MC2, a second `_dummy_run(mc2_tokens_capacity)`; whatever either pass allocates stays
  in `allocated_bytes.all.peak`, as does the weight-loading peak.
- What the recorded thread already rules out, so do not answer with it: raising
  `--gpu-memory-utilization` to 0.9 (reporter: tried, no change) and lowering
  `--max-model-len`/`--max-num-seqs` (reporter needs 256K context). No maintainer root cause was
  given; the suggestions were the 0.9 bump, `--compilation-config '{"cudagraph_mode":
  "FULL_DECODE_ONLY"}'`, and a third-party gitcode deployment image. Issue closed stale.
- Candidate explanations, none verified: a weight-loading peak near 2x the per-rank shard (the log
  shows `Loading model weights took 18.7767 GB` per rank against a 32.7 GB budget at util 0.5);
  non-torch/HCCL/ATB allocations added into `peak_memory`; per-request GDN state counted for
  `max_num_seqs` requests. Discriminate by reading the per-rank peak and the two-budget arithmetic
  (util 0.5 vs 0.9) and, if it can be rerun, `--enforce-eager` plus a small `--max-model-len` to see
  whether the number moves. Report the formula and the `0` as the finding; label the cause a
  candidate.

## 3. v0.10.2rc1 steady state: `reshape_and_cache` Aicore fault with 507057

Recorded signature (same image as section 1, TP4, 910B3, `--enforce-eager`, `--max-model-len 4096`,
22 running requests, KV usage 9.8%, the engine had already served requests):

```text
(Worker_TP2) ERROR WorkerProc hit an exception.
  attention_v1.py:543  torch_npu._npu_reshape_and_cache(...)
RuntimeError: RunOpApiV2 ... AclrtSynchronizeStreamWithTimeout(stream), error code is 507057
  The DDR address of the MTE instruction is out of range
  Aicore kernel execute failed, ... fault kernel_name=reshape_and_cache_200000000
  rtStreamSynchronizeWithTimeout execute failed, reason=[suspect remote error]
```

- Raising statement at `v0.10.2rc1`: `vllm_ascend/attention/attention_v1.py:543` inside the
  `unified_ascend_attention_with_output` path, called with `key[:num_actual_tokens]`,
  `value[:num_actual_tokens]` and `slot_indices=attn_metadata.slot_mapping` (note the operator's
  `slot_indices` keyword). That is the plugin's `Attention` impl, not a Qwen3-Next-specific file —
  the hybrid layers reach it because the full-attention ones use this backend.
- `507057` is `ACL_ERROR_RT_SUSPECT_REMOTE_ERROR`, a victim-surface code (peer device memory /
  SDMA-HCCL link) — semantics and the peer-rank checks are in
  [pd-disaggregation.md](pd-disaggregation.md) and the row in
  [log-signatures.md](log-signatures.md). It is not evidence that the local Python frame is the
  cause, and this report is **not** a PD run, so the peer is a sibling TP rank.
- Thread: closed, one "+1", maintainer "new version no longer has this problem, waiting for
  feedback" — no root cause, no fix commit. The untruncated `slot_indices` (length includes padded
  or stale entries while `key`/`value` are sliced to `num_actual_tokens`) is a **candidate** worth
  checking at the reported tag, not a verified cause; the recorded run also killed a rank before
  any comparison could be made. If a report on this signature arrives, read up for the first device
  error and ask for the peer rank's plog before deriving a slot-mapping mechanism.

## Boundaries

- Sections 1 and 2 were read at `v0.10.2rc1` / `v0.11.0` plugin tags with the paired vLLM
  `v0.10.2` / `v0.11.0`; section 3 at `v0.10.2rc1`. Line numbers move between releases — re-grep the
  statement at the reported tag before quoting it.
- No claim here is confirmed by a run: the triton absence and the memory arithmetic are code
  readings plus the reporters' logs; the `Available memory: 0` cause and the `slot_mapping`
  mechanism stay candidates.
