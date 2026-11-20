# Startup failures (engine/worker never becomes ready)

Scope: the process tree dies or stalls before the server accepts requests —
`EngineCore failed to start`, `WorkerProc initialization failed`, model load or
memory-profiling windows, OOM, wrong/unsupported config, version mismatch.

Inherit the rules in SKILL.md. Conditional routes:

- No per-rank worker traceback in the pasted log → `silent-worker-exit.md`
  (how the vLLM v1 multiproc executor reports each lane, what to request, and
  why `Parent process exited` / `TBE Subprocess` lines are consequences). When
  the paste also carries `!!!!!!! Segfault encountered !!!!!!!` and the parent's
  `Failed core proc(s): {'EngineCore': -11}`, finish in the startup section of
  `../hccl-and-hardware/worker-segfault-no-python-frames.md`: a trivial CPU
  `torch.zeros` at the top of the dump is not the cause.
- Nothing raised and nothing printed past `Loading model weights took X GB`,
  with the HTTP port still refused → `silent-stall-after-weight-load.md` first:
  bound the phase from the pinned worker (`determine_num_available_blocks` →
  `profile_run` → `npu.synchronize`, no capture at that ref), fingerprint V0 vs V1
  from the log's own `model_runner.py:<line>`, then ask for the full `vllm serve`
  line and take the liveness checks. Do not lead with `--enforce-eager` where the
  platform already sets `simple_compile_backend = "eager"`.
- Startup passes the model load and dies with `KeyError: '<...>.self_attn.attn'`
  inside upstream `offloading_connector.py register_kv_caches`, right after
  `Creating offloading spec with name: NPUOffloadingSpec` (CPU offload via
  `--kv-transfer-config` on an MLA checkpoint such as GLM-5 / Kimi-K2.5) →
  `../upstream-downstream-drift/kv-offloading-layer-coverage.md`: the connector's
  `Attention`-only layer filter drops `MLAAttention`, so this is a layer-type
  coverage gap, not a worker death.
- The paste's `file:line` names a `vllm_ascend/**` module the pin does not
  contain (e.g. `vllm_ascend/models/<x>.py`) → `environment-correspondence.md`,
  its vendored-file section: the model code is the reporter's, the framework calls
  it makes are still citable.
- Failure at load/profile/graph-capture time on a machine with 64 GB cards, or a
  deployment command that omits `--max-model-len` → `max-model-len-budget.md`
  first. This is the reporter-confirmed answer for the recorded Qwen3-VL-30B-A3B
  case.
- The OOM happens at engine start with `--max-model-len` already explicit, and a
  smaller `--tensor-parallel-size` on the same node starts (TP4 up / TP8 OOM), or
  only two serve commands are pasted → this is a budget question, not a startup
  lane: `../deploy-sizing/tp-scaling-oom-at-startup.md` (one call:
  `../../scripts/oom-budget-split.sh`).
- `value error, the checkpoint you are trying to load has model type '<mt>' but
  Transformers does not recognize this architecture`, `Model architectures [...]
  are not supported`, or the bare question "can version V deploy model M?" →
  `model-type-not-recognized.md` first. This is a version gate, not a defect:
  answer no/yes + the earliest release that works, then show the gates.
- Suspicion of "model X is unsupported", wrong image/vllm pairing, or an
  obvious maintainer "works on my machine" reply → `environment-correspondence.md`.
  That page also carries the scope of the question: the launch environment is
  **one environment per worker process**, so on `ray`/multi-node a driver-shell
  `source set_env.sh` proves nothing about a worker on another node — ask for
  `pip show vllm vllm-ascend`, `import triton` and `which bishengir-compile`
  per node (#3098).
- The per-rank traceback is `AssertionError: Current device type:
  AscendDeviceType.A5 does not match the installed version's device type:
  AscendDeviceType.A2, please check your installation package.` (or
  `RuntimeError: Can not support soc_version: <n>`) at
  `vllm_ascend/worker/worker.py:113` → `check_ascend_device_type()`, with no
  `Starting to load model` line: the installed package was **built** for another
  SoC family than the card reports. Two one-liners settle it and the guard must not
  be patched → `../soc-platform-paths/device-type-build-vs-runtime.md`.
- `ValueError: Non-MoE models do not support external data parallel mode`, or any
  `ValueError`/`AssertionError` from `vllm/engine/arg_utils.py` whose text names a
  parallel mode or flag combination (no `WorkerProc` line, no rank prefix) →
  `arg-validation-errors.md`. Upstream argument validation: the fix is a launch
  command, and when the shape came from a tutorial it is also a doc bug.
- Every rank dies in `torch.arange(world_size).reshape(...)` with
  `RuntimeError: shape '[-1, <dp>, <pp>, <tp>]' is invalid for input of size <W>`,
  frames `vllm_ascend/worker/worker_v1.py` → `ensure_model_parallel_initialized` →
  `initialize_model_parallel`, often from `examples/offline_external_launcher.py` or
  any `distributed_executor_backend="external_launcher"` launch → same page, third
  raise site: the trailing literal is the requested TP and `size W` is the real
  world size, which the launcher computed from `--node-size × --proc-per-node`.
  No `raise` to grep and no automatic validation — the message's two numbers are
  the whole diagnosis, before any weight load. Report the doc defects too.
- The startup `ValueError` is raised by **vllm-ascend**, not by `arg_utils`:
  `KV transfer '<role>' config has a conflicting data parallel size. Expected 1,
  but got 2.` (or `tensor parallel size`) on a PD deployment whose
  `--kv-transfer-config` sets `kv_connector_extra_config` `dp_size`/`tp_size`,
  usually with connector and model-load lines *before* the raise →
  `arg-validation-errors.md` (its second raise site, #8345). The compared number
  is `parallel_config.data_parallel_size` after upstream rewrote a **dense**
  engine's DP to 1, so the fix is the block's number, not the launch DP flag.
- The startup dies at the **first weight group** — `Starting to load model` was
  printed, `Loading safetensors checkpoint shards: 0% ... 0/N`, and the deepest
  frames are the model loader reaching
  `vllm/model_executor/models/utils.py` `raise ValueError` with
  `There is no module or parameter named '<name>' in <Class>` — while the init
  line says `pooler_config=None` and the class name ends `ForCausalLM`: the
  report is an embedding/pooling checkpoint served as a generation model, so use
  `../pooling-embedding/embedding-checkpoint-as-generate.md` (one call:
  `../../scripts/checkpoint-key-prefix.py <model dir>`). The message's prefix and
  class name are computed, not part of the checkpoint's text — read them before
  the loader.
- A pydantic `ValidationError` whose text is `Unexpected keyword argument
  [type=unexpected_keyword_argument, input_value=..., input_type=bool]` naming a key
  the reporter put in `--additional-config`, often followed by
  `[ERROR] ... ERR99999 UNKNOWN application exception`, with no `WorkerProc` line →
  `additional-config-schema.md` first. The key is nested-only (or was removed) and
  `AscendConfig` is `extra="forbid"`; the fix is one JSON edit, and a tutorial
  serve command is usually the source.
- `<method> quantization is currently not supported in npu` (a pydantic
  `ValidationError` around it, e.g. `--quantization awq`), or
  `Unknown quantization method: <x>` — the platform/quantization gate, raised while
  building `ModelConfig`, before weights and before any worker →
  `quantization-method-gate.md`. The NPU platform declares only `ascend` and
  `compressed-tensors`; the fix is a re-quantized checkpoint, not a launch tweak.
  The **inverse** report lands on the same page and looks nothing like it: an
  msmodelslim W8A8 checkpoint has no `quantization_config` in `config.json` by
  design, so nothing is rejected — the engine starts and a rank dies loading int8
  tensors into a dense model, behind the bare `Engine core initialization
  failed` (#2015). Ask for the per-rank traceback and the `quantization=` /
  `--quantization ascend` flag before ranking causes.
- `AttributeError: '<Model>Config' object has no attribute '<attr>'` raised inside
  `vllm_ascend/distributed/*connector*.py` from `ensure_kv_transfer_initialized`
  (typically a VL/MoE model under PD disaggregation) → `connector-config-attributes.md`.
- Python `AttributeError`/`TypeError` in a `vllm_ascend/{ops,patch,models}` frame
  whose class subclasses or re-implements an upstream vLLM/transformers class
  (e.g. `'AscendQwen2DecoderLayer' object has no attribute 'attention_type'`),
  **or the same error inside an upstream vllm file on a class vllm-ascend
  replaces** (e.g. `'ZeroExpertFusedMoE' object has no attribute
  'select_experts'` — the replacement is keyed by class name)
  → `../upstream-downstream-drift/index.md`; this is drift, not a config value.
- The per-rank traceback has `torch/_subclasses/fake_tensor.py` and
  `torch/_ops.py` frames and ends `NotImplementedError: <ns>::<op>: attempted to
  run this operator with Meta tensors` (e.g. `_C::rotary_embedding`), after the
  weights loaded and before graph capture → `meta-kernel-missing.md` first. A
  custom op registered for `PrivateUse1` only, reached because PIECEWISE
  compilation traces with fake tensors; `--enforce-eager` is the first rung, the
  dtype that skips the gated branch is the second, and the namespace in the message
  (`_C` vs `_C_ascend`) dates the tree.
- The traceback reached `Starting to load model`, has **no** `vllm_ascend` frame
  below the loader, and ends `AssertionError: Torch not compiled with CUDA
  enabled` (typically from a `torch.cuda.current_device()` inside an upstream
  `__init__`) → `checkpoint-config-cuda-only-path.md`. A checkpoint
  `config.json` key selected an upstream CUDA-only class that vllm-ascend does
  not implement; the env dump's `torch==2.7.1+cpu` is normal on NPU and is not
  the cause. One call: `../../scripts/cuda-only-path-audit.sh`.
- If the log names a collective (`HcclAllreduce`, `hcclRet`, `ERR00100`) instead
  of a Python-level cause, leave this topic and use
  `../hccl-and-hardware/index.md`.
- If instead the server was already serving and died on memory later ("显存不释放"
  then crash, `NPUWorkspaceAllocator` OOM after minutes/hours), this is not a
  startup lane — use `../runtime-memory/index.md`.
- A missing system library or Python module the tutorials install (`cannot open
  shared object file`, `No module named`), or the interpreter aborting inside a
  native module the image shipped (glibc allocator text, no Python frame, no
  `WorkerProc` line) → leave this topic for `../image-dependency-faults/index.md`:
  the image or its build recipe owns that failure, not vllm-ascend. When the
  missing library is named by the *connector's own* `from mooncake.engine import
  TransferEngine` line (`ImportError: ascend_transport.so`) and the deployment was
  started by a launcher's `args:`/`command:` block, read
  `../image-dependency-faults/launch-env-and-library-path.md` first — the
  `collect_env.py` ENV block describes the shell, not the launched process.
- The per-rank traceback is `OSError: libatb.so: cannot open shared object file`
  → `OSError: Could not load this library: .../torch_npu/lib/libop_plugin_atb.so`
  → `_register_atb_extensions()` at `vllm_ascend/worker/worker.py`, plus an outer
  `OSError: [Errno None]`, with no `Starting to load model` line anywhere → the
  NNAL/ATB component is uninstalled or (more often) installed but its own
  `set_env.sh` was never sourced: `../image-dependency-faults/nnal-atb-missing-library.md`.
- Nothing crashed and the complaint is that a *graph/compilation setting* did not take
  effect ("`--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'` 或 yaml 方式
  均未生效", no capture log, often no log pasted at all) → `graph-mode-not-applied.md`
  first: one gate chain mutates `cudagraph_mode` between argument parsing and the
  worker, every gate prints its own line, and a stage yaml with `level: 1` asks for
  `STOCK_TORCH_COMPILE` — refused at `platform.py:241-245` while the reporter sees
  only a warning. The success line (`FULL_DECODE_ONLY compilation enabled on NPU`,
  `platform.py:313`) is the one that declares the mode live.
- The log reached graph capture — `Starting to capture ACL graphs for cases:
  [...]`, `Capturing ACL graphs (...): 0/N`, `ACLgraph sizes capture fail`,
  `NPUGraph.cpp` "Waiting for pending NCCL work" — then failed or stalled →
  `acl-graph-capture.md` first. If instead the startup was clean and the
  capture error arrives on a **request** (`CUDA graph capturing detected at an
  inappropriate time`, `Worker failed with error …`), that is a descriptor the
  startup never captured →
  `../lora-adapters/graph-capture-descriptor-mismatch.md`. The `ACLgraph has
  insufficient available streams` sentence there is canned wrapper text printed
  for every capture exception, and the capture-size set may already be the
  computed maximum.

Common rules for this topic:

1. Establish which lane the failure is in before reading implementation source,
   and answer the model-support question before the source walk when that is
   what was asked.
2. Ask for the complete per-rank log plus `npu-smi info`, the exact `vllm serve`
   command, the image tag, and the `Using max model len N` line. Trimmed logs are
   the norm; name the specific lines you need.
3. A start-up flag difference (explicit `--max-model-len`, `--enforce-eager`,
   `--compilation-config`) is cheaper to test than a source-level theory. Ask for
   the re-run before proposing a bug.
4. A configuration flag that the reporter relies on belongs in the pinned
   version's parser before it belongs in a theory: `rg -n '<key>' vllm_ascend/`
   (unknown `additional_config` keys are ignored without a warning).
5. A count or list derived in the log that does not match the upstream default
   is usually a vllm-ascend rewrite, not a reporter mistake; grep
   `vllm_ascend/` for the config field before theorising. For graph-capture
   sizes specifically, `acl-graph-capture.md` step 3 names the rewriting
   function and `../../scripts/aclgraph-size-budget.sh` reproduces the number.
