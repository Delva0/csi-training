# Startup failures (engine/worker never becomes ready)

Scope: the process tree dies or stalls before the server accepts requests —
`EngineCore failed to start`, `WorkerProc initialization failed`, model load or
memory-profiling windows, OOM, wrong/unsupported config, version mismatch.

Inherit the rules in SKILL.md. Conditional routes:

- No per-rank worker traceback in the pasted log → `silent-worker-exit.md`
  (how the vLLM v1 multiproc executor reports each lane, what to request, and
  why `Parent process exited` / `TBE Subprocess` lines are consequences). Its
  no-traceback section is the route when the paste *cannot* carry one (screenshot OCR
  is noise): `WorkerProc failed to start.` is `logger.exception`, so the stack is the
  block **below** the message, and the lane is classified from the serve path's quant
  suffix (`…-W8A8`) plus that matrix cell — a quantised variant with an unvalidated
  cell ranks the load-phase aux-key family first, never the oddest flag in the command. When
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
- Nothing raised, and the last worker line is `Encoder cache will be initialized with a
  budget of N tokens, and profiled with M <modality> items of the maximum feature size`
  while EngineCore repeats `[shm_broadcast.py:…] No available shared memory broadcast
  block found in 60 seconds` → `mm-encoder-profile-stall.md` first: the phase is the
  vision-tower dummy forward inside `profile_run`, `N` is the reporter's own
  `--max-num-batched-tokens` and `M` its `--max-num-seqs` × `--limit-mm-per-prompt`, the
  shm lines are the wait. One flag (`--limit-mm-per-prompt '{"image":0,"video":0}'`) or
  `--skip-mm-profiling` skips the block entirely and decides whether the encoder owns it;
  then cut the budget to the tree's own 310P recipe (2048/16384) and ask for `npu-smi`
  during the stall.
- The start **does** raise, from the multimodal budget before any EngineCore or NPU
  init — a stack through `input_processor.py` → `encoder_budget.py` →
  `registry.py get_dummy_mm_inputs` → the model file's `get_dummy_mm_data` ending in
  `AttributeError: '<HfProcessorClass>' object has no attribute '<x>'` →
  `hf-processor-attribute-at-mm-budget.md` first (#13491: MiniCPM-V-4.6 on 310P,
  `transformers==5.7.0` against the release's `==5.5.4` pin). The phase itself deletes
  every device lane; the deliverable is the processor/version contract, one command that
  prints the class module, and the support row.
- The profiling window **does** raise, and the same error appears once per rank but the
  driver: `EOFError: Ran out of input`, bottom frame
  `torch_npu/distributed/distributed_c10d.py` `_gather` →
  `dist.broadcast_object_list`, above
  `logits_processor._gather_logits` → `tensor_model_parallel_gather`, on a TP > 1
  dense serve whose log also prints `npu is experimental on VLLM_USE_V1=1. Falling back
  to V0 Engine.` → `profile-run-logits-gather-eof.md` first: HCCL has no `gather`, so
  torch_npu emulates it with an object-broadcast size handshake that handed the
  non-destination ranks an empty size payload. The driver's `HcclAllgather` `EI0006` +
  `DestLink` block ~120 s later is that stalled collective timing out, not a link fault,
  and the V0 notice is an opt-in (`VLLM_USE_V1=1` is the first rung), not a platform
  verdict.
- Nothing raised, and the paste stops **inside** weight loading — `Loading
  safetensors checkpoint shards:   0% Completed | 0/N`, or a bar that never
  advances, with no `Loading model weights took X GB`, the port still refused and
  the worker processes alive → `weight-load-stall-in-shard-loop.md` first.
  Resident HBM equal to checkpoint-size/TP is the *pre-load* state — the loader
  builds the model on the device before it reads a shard
  (`vllm/model_executor/model_loader/base_loader.py:53-64` at vllm v0.21.0) — so
  it does not move the stall after the copies; rank the checkpoint **source**
  (bind-mounted host dir, mount, disk, a copy in flight) beside the post-copy
  code paths, and take the `/proc/<pid>/io` sample
  (`../../scripts/shard-source-probe.sh <model dir> <pid>`).
- Nothing raised, and the silence starts **earlier**: the log stops right after the
  worker's own device line with no traceback and no `Starting to load model`, while
  the report says "TP=1 works, TP>1 deadlocks during driver initialization" and the
  launch used `distributed_executor_backend="external_launcher"` or any torchrun
  SPMD script → `multiprocess-launch-hang.md` first. `init_process_group` is handed
  `world_size=parallel_config.world_size` explicitly, so fewer processes than TP wait
  at the rendezvous forever instead of raising; count processes and compare
  `MASTER_ADDR`/`PORT` before any HCCL theory, and do not read one process's `npu:0`
  as the other ranks' device.
- Nothing raised on a **bare `python <example>.py`** (no launcher, TP=1, an upstream
  example the plugin also ships under `examples/`), the body says the same script ran
  on an older image, and HBM is already taken → `multiprocess-launch-hang.md` "second
  shape": the port's extra env line is the fix (`VLLM_WORKER_MULTIPROC_METHOD=spawn`;
  upstream defaults to `fork` and `_maybe_force_spawn()` has no NPU branch), and the
  feature the example exercises may still be a coverage gap at that pin.
- Nothing raised, the paste stops on the worker's **constructor** warning
  (`[utils.py:2522] Methods add_lora,cache_config,… not implemented in
  <vllm_ascend.worker.worker_v1.NPUWorker object>`, i.e.
  `warn_for_unimplemented_methods`, `vllm/utils.py:2488-2529`), every line carries a
  `(RayWorkerWrapper pid=…)` prefix, and the engine was built **from a program** —
  `AsyncLLM.from_engine_args` / `LLM(...)` with `distributed_executor_backend="ray"`
  inside an RLHF trainer, so there is no `vllm serve` line — →
  `ray-programmatic-init-hang.md` first. The named frame is the *waiter*:
  `_wait_for_engine_startup` (`vllm/v1/engine/core_client.py:412-443`) raises only when
  a core process exits and logs progress at DEBUG, so a live-but-stalled child is
  silent forever; the constructor warning dates the stall to `init_device` /
  the start of `load_model`, and the actors' existence already refutes the
  `GPU`-vs-`NPU` resource lane.
- Every line is prefixed `(EngineCore_DP0 pid=…)`, `Connected to Ray cluster.`
  printed, and the start dies in the same second with
  `WARNING [ray_utils.py:…] The number of required NPUs exceeds the total number of
  available NPUs in the placement group.` then
  `ValueError: Current node has no NPU available. current_node_resource={… 'GPU': 8.0}`
  (~`ray_utils.py:352/:368` on vllm 0.10.2) → the cluster advertises `GPU` while the
  platform asks for `NPU` (`NPUPlatform.ray_device_key`, `platform.py:52`): read
  `ray status`'s Resources block, then restart each raylet with
  `--resources '{"NPU": 8}'` → `ray-npu-resource-label.md`. One turn decides it; there
  is no card shortage to size and no driver fault behind it.
- The **same** `ValueError: Current node has no NPU available` but the dict reads
  `'NPU': 0` beside `'NPU_group_9_<pgid>': 1.0`, `'NPU_group_10_<pgid>': 1.0` …, on a
  serve with `--data-parallel-size` > 1 → **not** the label case above: the check reads
  the *free* view (`available_resources_per_node`, `v1/executor/ray_utils.py:463`), so a
  live placement group already holds the node's devices and the raising process is
  outside it. Grep their log for `Creating placement groups for data parallel`
  (`v1/engine/utils.py:412`) and `Using the existing placement group`
  (`ray_utils.py:419`): neither, plus `No current placement group found …`, means the
  default `mp` DP backend (`parallel.py:121`) started a sibling engine whose own PACK
  group took the node → `ray-dp-placement-group-npu-exhausted.md` (one read, and no
  `ray start --resources` advice).
- The paste **ends** on `… [parallel_state.py:<n>] Adjusting world_size=<N> rank=<r> distributed_init_method=tcp://127.0.0.1:<port> for DP` with no traceback, on a serve with `--data-parallel-size` > 1 → `dp-world-rendezvous-stall.md` first: that line is the last statement before the world `init_process_group`
  (`vllm/distributed/parallel_state.py:871-895` at vllm v0.8.5), so the ranks have not all
  joined. Compare `PP×TP×DP` (`world_size_across_dp`, `vllm/config.py:1646-1647`, `:1724`)
  with the cards in the same paste before any HCCL theory — DP=16 on one 8-card node can
  never rendezvous; the DP engine cores are spawned locally (`core_client.py:849-851`) and
  each ray worker asks `resources={'NPU': VLLM_RAY_PER_WORKER_GPUS}`
  (`ray_distributed_executor.py:159,205-221`), so the extras stay pending up to
  `PG_WAIT_TIMEOUT = 1800` s without printing. Ask `ray status` + per-rank log count, and
  leave the cause open versus `ray-npu-resource-label.md` only if the paste carries that
  `ValueError`.
- The evidence is a **KubeRay `RayService`/`RayCluster` YAML** (not a log): the service
  sits at `applicationStatus.llms: DEPLOYING`, the reporter says the plugin "起来之后
  没法申请到资源", and the same `vllm serve … --distributed-executor-backend ray` works
  by hand → a three-layer resource question, not an engine one: K8s
  `huawei.com/Ascend910` is not Ray's `NPU`, only `rayStartParams.resources` bridges
  it (a head group with `rayStartParams: null` and no device limit advertises none), the
  Serve replica actor's own `ray_actor_options.resources` consumes a bundle in addition
  to one per engine rank (`pp=2` on two devices = three bundles), and `HCCL_IF_IP` /
  `GLOO_SOCKET_IFNAME` / `TP_SOCKET_IFNAME` must exist **before** `ray start`. One
  `ray status` + `ray list placement-groups` separates the layers; `ray/serve/` is in no
  export → `ray-npu-resource-label.md` §*A KubeRay RayService YAML* (recorded #8156).
- A **model family the pinned release only just added** (a hybrid/GDN or otherwise new
  architecture) hangs or dies at init/load before the first request → read that version's
  own `docs/source/user_guide/release_notes.md` *Highlights + Known issues* and the model's
  own tutorial serve line **before** the source walk: they carry the exclusions and the
  required extra package. At v0.10.2rc1 they say Qwen3-Next support is new and that
  "expert parallel and MTP feature doesn't work with this release", the tutorial requires
  Triton Ascend (`../image-dependency-faults/triton-ascend-pair.md`) and launches with
  `--max-model-len 4096 --gpu-memory-utilization 0.7 --enforce-eager`, while the known issue
  is HBM usage higher than expected (set `max_model_len`/`gpu_memory_utilization`). A
  report that runs the tutorial's model but not its flags is a recipe diff first
  (recorded #3170).
- **Two engines were started on the same host** (a P/D pair, or two deployments) and the
  second one to start dies at worker init with `Bind_Failed(EJ0003)` /
  `hcclCommInitRootInfoConfig … error code is 7` while the first keeps serving → count
  cards per engine before any port variable. `16666` is HCCL's single default comm port,
  so the endpoint is decided by the device IP alone and a collision means the two
  launches were given the **same cards** — the external-DP launcher numbers them from 0
  in every invocation → `../pd-disaggregation/same-host-pd-device-partition.md` (the
  family split: `../hccl-and-hardware/comm-create-errors.md` §code 7).
- Every rank's last line is `Using OOT custom backend for compilation` (nothing
  raised, port refused, HBM resident) → `compile-backend-banner-then-silence.md`
  first: at v0.22.1rc1 that banner is printed **while the model object is built**
  (`vllm/compilation/decorators.py:406` → `wrapper.py:90` → `compilation.py:1066`,
  before `base_loader.py:64` reads a shard), so the phase is the target model's
  weight load and only rank 0 shows read progress (the shard bar is rank-gated).
  `--enforce-eager` cannot separate it — the load runs in both lanes — so rank the
  load-side causes of `weight-load-stall-in-shard-loop.md` before any
  compile/capture theory.
- Startup passes the model load and dies with `KeyError: '<...>.self_attn.attn'`.
  **Split it by the frame, not by the key.** Inside upstream
  `offloading_connector.py register_kv_caches`, right after
  `Creating offloading spec with name: NPUOffloadingSpec` (CPU offload via
  `--kv-transfer-config` on an MLA checkpoint such as GLM-5 / Kimi-K2.5) →
  `../upstream-downstream-drift/kv-offloading-layer-coverage.md`: the connector's
  `Attention`-only layer filter drops `MLAAttention`, so this is a layer-type
  coverage gap, not a worker death. Inside `initialize_kv_cache →
  initialize_attn_backend → get_attn_backends_for_group` with
  `--pipeline-parallel-size` > 1 (typically `ray`, one node per stage) →
  `../pipeline-parallel/kv-config-rank-mismatch.md`: the registry map
  (`get_layers_from_vllm_config`) is fine and the *handed-in layer list* is the
  foreign one (per-worker `KVCacheConfig` index vs Ray's worker re-rank); fingerprint
  the printed layer index with `scripts/pp-layer-partition.sh` before reading any
  executor code.
- Every rank dies in `_reshape_kv_cache_tensors` on a bare
  `AssertionError:  assert sum_page_size_bytes % current_kv_cache_spec.page_size_bytes == 0`,
  on a DSA model with sparse C8 and/or DCP/MTP → not a memory or hardware lane:
  `../dsa-sparse-attention/c8-page-size-reshape-assertion.md` (one script call
  decides whether the arithmetic can even pass).
- The paste's `file:line` names a `vllm_ascend/**` module the pin does not
  contain (e.g. `vllm_ascend/models/<x>.py`) → `environment-correspondence.md`,
  its vendored-file section: the model code is the reporter's, the framework calls
  it makes are still citable.
- `ImportError: cannot import name '<symbol>' from 'vllm…'` raised at a
  `vllm_ascend/**` import while `vllm serve` is still building its parser (frames
  `make_arg_parser` → `pre_register_and_update` → `adapt_patch` → a `patch/platform/`
  module) → `environment-correspondence.md`, its `from vllm.utils import …` section:
  count the unguarded importers of that symbol first (a one-line edit relocates the
  error to worker start), read the monolith-vs-package path in the traceback as the
  upstream generation, and take the symbol's new home from a newer vllm-ascend pin.
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
  The **look-alike** lands on the same page and is not a gate:
  `ERROR [registry.py:<n>] Error in inspecting model architecture '<Arch>'`
  followed by a `Traceback (most recent call last):` and nothing else, on an
  architecture that is registered in both trees — the line is `logger.exception`,
  the real tail was printed into the same log and cut, and inspection runs in a
  `python -m vllm.model_executor.models.registry` subprocess, so it is an
  import-time failure of `vllm_ascend.models.<model>` (wrong triton, or an image
  that is not the tag). Ask for the tail *and* give the reproducing one-liner.
  The same crash also arrives as
  `pydantic_core._pydantic_core.ValidationError: 1 validation error for ModelConfig`
  / `Value error, Model architectures [...] failed to be inspected` with **no**
  `[registry.py]` line (recorded #4206): still not the gate — that wording exists
  only for a registered architecture — and still the same ask.
- Suspicion of "model X is unsupported", wrong image/vllm pairing, or an
  obvious maintainer "works on my machine" reply → `environment-correspondence.md`.
  That page also carries the scope of the question: the launch environment is
  **one environment per worker process**, so on `ray`/multi-node a driver-shell
  `source set_env.sh` proves nothing about a worker on another node — ask for
  `pip show vllm vllm-ascend`, `import triton` and `which bishengir-compile`
  per node (#3098).
- The graph-mode start dies at **config time** — the last line before the
  traceback is `PIECEWISE compilation enabled on NPU. use_inductor not supported -
  using only ACL Graph mode` and `Starting to load model` never appears — on a
  **VL/audio** model, with an `AttributeError` naming the *config object*
  (`'InternVLChatConfig' object has no attribute 'num_hidden_layers'`) or ending in
  the checkpoint's own `configuration_*.py` `to_dict` (`… has no attribute
  'llm_config'. Did you mean: 'sub_configs'?`) → `vl-layer-count-attribute.md`.
  `update_aclgraph_sizes` needs a layer count off a composite config; eager mode
  never reaches the call, so `--enforce-eager` is the workaround and the graph-branch
  call site is the proof. Not the capture-size lane.
- The per-rank traceback is `AssertionError: Current device type:
  AscendDeviceType.A5 does not match the installed version's device type:
  AscendDeviceType.A2, please check your installation package.` (or
  `RuntimeError: Can not support soc_version: <n>`) at
  `vllm_ascend/worker/worker.py:113` → `check_ascend_device_type()`, with no
  `Starting to load model` line: the installed package was **built** for another
  SoC family than the card reports. Two one-liners settle it and the guard must not
  be patched → `../soc-platform-paths/device-type-build-vs-runtime.md`.
- The per-rank traceback shows the **same** first-worker frame (`worker.py` →
  `check_ascend_device_type()` → `utils.py` `torch_npu.npu.get_soc_version()`) but the
  text is `ERR99999 PTA call acl api failed` + `set default device id failed, ret: 107001`
  / `[FUNC:SetDefaultDeviceId]` / `invalid device ID. Check whether the device ID is valid.`
  (+ `ctx is NULL` / `context is a null pointer`, the 107002 cascade): the ACL open
  rejected an **id the container does not expose**, so `ASCEND_RT_VISIBLE_DEVICES` and the
  `docker run`'s `--device /dev/davinciN` are the pair to compare — not the wheel, not a
  flag, not the model. Read the text before routing to the SoC-mismatch bullet above:
  `device-id-invalid-at-worker-init.md` (`ls /dev/davinci*` + `npu-smi info` inside the
  container, then one framework-free `torch_npu.npu.device_count()` line).
- The report's **first** failing command is `npu-smi info` itself ("容器中获取不到NPU信息"), with
  `dcmi model initialized failed, because the device is used. ret is -8020`, `drvRet=87` at
  `halGetDeviceInfo`/`drvGetDevNum` and `rtGetDeviceCount … 507899`, and **no** engine or
  `vllm_ascend` frame anywhere: the DCMI/driver userspace plus the container's device nodes own it,
  and `AscendCheckSoCVersion: cur soc version unknown` is downstream of the failed probe (unlike the
  resolved `PlatformInfoInit … SocVersion <name>` below). Run the same `npu-smi info` on the host
  first — that one comparison splits "container" from "card/driver" — then the device nodes,
  another holder, and the two sides' `driver/version.info` → `no-npu-in-container.md`.
  A `npu-smi info` that **does** answer does not clear the container's mount list: a
  `docker run` missing the pin's `-v /usr/local/dcmi:/usr/local/dcmi` line
  (`docs/source/quick_start.md:30`, `:59` at v0.11.0rc2) surfaces later, at weight load
  (reporter-confirmed #4847) → same page, §*A responding `npu-smi info` is not a mount check*.
- `EngineCore failed to start` / `WorkerProc failed to start` whose deepest frame is
  `torch_npu` device open (`set_device` → `npu_setDevice`) and whose text is
  `RuntimeError: SetPrecisionMode: … aclrtSetCompileOpt(aclCompileOpt::ACL_PRECISION_MODE,
  precision_mode), error code is 500001` + `The internal ACL of the system is incorrect.`
  + `EH9999 … [PLATFORM]PlatformInfoInit runtime platform info failed. SocVersion <name>`
  + `GEInitialize failed`, often ending `ERR99999 UNKNOWN application exception`: ACL/GE
  bring-up inside torch_npu, before any `vllm_ascend` frame. Read the printed `<name>`
  against `npu-smi` before anything else; a `<name>` that matches the card inside a
  container whose `docker run` already matches the docs takes the same command +
  `--privileged=true` as its next step (reporter-confirmed #302), and only then the
  driver/firmware-vs-image-CANN pairing
  → `acl-init-500001-and-soc-version.md` (no precision flag is the cause; cap the C++ walk).
- The **same phase and the same two functions**, but no `500001`: the SoC name resolved and
  matches `npu-smi`, and the failing statement is a file lookup —
  `RealPath:Path '<…>/aarch64-linux/lib64/../data/platform_config' does not exist.` +
  `InitializePlatformInfo:File path[] is not valid.` + `QueryPlatformInfoValue:init runtime
  platform info failed, SocVersion <name>`, then `RuntimeError: Get ACL JitCompile default value
  failed.` / `ERR00100 PTA call acl api failed`. That path is the loaded FE `.so`'s own
  directory plus `../data/platform_config`, so the answer is the container's CANN **tree**, not
  the mount-boundary/`realpath` theory and not `--privileged`: `cann/runtime` is exported and
  `src/platform/platform_info.cpp` prints both lines (two reads) →
  `platform-config-file-missing.md`.
- The startup dies inside `torch.compile` with `AttributeError: '_OpNamespace' '<ns>'
  object has no attribute '<op>'`, deepest frames `vllm_ascend/compilation/passes/**`
  plus `torch/_inductor/pattern_matcher.py` (`check_fn` → `fwd_only` → `make_fx`), e.g.
  `'vllm' object has no attribute 'qkv_rmsnorm_rope'` from
  `qknorm_rope_fusion_pass.py:214`: a triton-backed custom op whose registration is
  `if HAS_TRITON:`-gated, called by an ungated pass that is on by default →
  `missing-triton-op-in-compile-pass.md` (one log grep, then
  `fuse_qknorm_rope: false`).
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
  the loader. The same raise with a **quant aux suffix** in the quoted name
  (`…blocks.0.mlp.fc0.weight_offset`, `…weight_scale`, `…deq_scale`) is the other
  reading: the export quantised a module the model file built without `quant_config`
  (ViT towers, multimodal projectors), and the message's `(`class`)` is the runtime
  CustomOp class, not evidence that the checkpoint named it →
  `weight-scale-keyerror-quant-variant.md` §the `ValueError` form (recorded #7200).
- The loader dies on a **file**, not on a config: `safetensors_rust.SafetensorError:
  Error while deserializing header: HeaderTooLarge` (or `HeaderTooSmall` / `invalid
  JSON`) in the same second as `Loading safetensors checkpoint shards: 0% ... 0/N`, on
  every rank, with `_pickle.PicklingError: Can't pickle <class
  'safetensors_rust.SafetensorError'>` on the non-driver ranks →
  `checkpoint-file-not-safetensors.md`. The header length in the first 8 bytes is
  parsed before any tensor, so the shard is an unresolved Git-LFS pointer, a partial
  copy or the wrong file: no `vllm_ascend` frame below the loader, no device work, and
  the TP size in the title is innocent (one call:
  `../../scripts/checkpoint-header-scan.py <model dir>`). Unlike the row above, the
  message is not about tensor names — do not open the loader.
- `ValueError: Following weights were not initialized from checkpoint: {...}` whose
  missing set is a whole **MTP module** (`model.pre_fc_norm_hidden`,
  `model.pre_fc_norm_embedding`, `model.fc`, `model.layers.0.*`, `model.norm`), on a
  `--speculative-config` `*_mtp` launch and a **fine-tuned export** while the base
  checkpoint starts → `mtp-head-missing-from-checkpoint.md`: the drafter is built from
  `config.json` alone and the raise is the loader's completeness check, not a tensor
  read. One call decides it (`../../scripts/mtp-head-check.py <model dir>`), and the
  quantized checkpoint takes the silent form (`Accepted: 0`, acceptance 1.00) because
  the check is off when `quantization` is set.
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
  A **third** form on the same page is earlier than both and looks like a release
  limit: `vllm serve: error: argument --quantization/-q: invalid choice: 'ascend'
  (choose from 'aqlm', …, None)` — `argparse`, so no `APIServer pid=` line either,
  and the printed choices are upstream's static `QuantizationMethods` `Literal`.
  On a ref whose `NPUPlatform.pre_register_and_update` ignores the `parser` it is
  handed (v0.8.5rc1, and still v0.7.3.post1) the plugin never adds the name to
  `--quantization`, while the platform list at that same ref *would* accept it
  (`supported_quantization = ["ascend"]`) — classify on the message form (§D of
  `quantization-method-gate.md`), and check the reporter's installed
  `pre_register_and_update` before calling the build unquantized. On the 0.7.3 pair
  the flag is not the end either: that tree has no native quantizer and routes w8a8
  through `mindie-turbo` (`quantizer.py:28-44`, `faqs.md:82`, §D-i), so the honest
  route there is the paired upgrade.
  A **fourth** form is §E of the same page and is the one a model named
  `DeepSeek-V3`/`-R1` (not `-w8a8`) hits: the checkpoint is the upstream **FP8/E4M3**
  release, which the 910 does not execute at these refs. It arrives as §A's
  `fp8 quantization is currently not supported in npu` when the plugin never
  registered `ascend`, and as §C's construction-time `KeyError:
  'model.embed_tokens.weight'` when it did — one
  `.../scripts/checkpoint-key-prefix.py --dtypes <model dir>` decides it from the
  safetensors headers, and `config.json` does not (the conversion guide has converters
  delete those fields first).
- A **VL-MoE** quantised serve fails at the same family in
  `vllm/model_executor/models/qwen3_vl_moe.py:279`, reached through
  `models/qwen3_vl.py:2083` → `models/utils.py:335` → `model_loader/online_quantization.py:173`
  (that last frame is the standard `AutoWeightsLoader.load_weights` decorator, not a sign
  of online quantisation): its guard skips `.weight_scale`/`.input_scale` but has no
  `*_offset` member (`:173-184`, `:272-279` at vllm v0.14.1 `d7de043`), so only an
  offset-carrying W8A8 export raises. Recorded #7744: the fix was the newer tag
  (`v0.17.0rc1`, reporter-confirmed), and the pin's matrix leaves the Qwen3-VL-MOE W8A8
  cell empty (`supported_models.md:81` at v0.14.0rc1).
- The loader raises `KeyError: '<prefix>.weight_scale'` (also `.scale` pre-remap,
  and the whole `…mlp.experts….weight_offset` family of a W8A8 export) at
  `param = params_dict[name]`, on every rank but with a **different layer** each
  time, drawn from a few module families (`self_attn.wkv`,
  `self_attn.indexer.wq_b`, `self_attn.compressor.wkv`, or `mlp.experts`) →
  `weight-scale-keyerror-quant-variant.md` first: this is the *mirror* of the
  quantization gate below/above — the checkpoint owns a quant aux tensor the engine
  built no parameter for. Two readings there for the offset form: an index-free
  `experts.w2_weight_offset` is upstream's substring remap of a per-expert
  checkpoint key (not a fused export), and at v0.8.5rc1 only `W8A8_DYNAMIC` allocates
  MoE offsets at all, so a load-phase offset miss means that family resolved to
  `FLOAT` (recorded #986). Census the checkpoint's scale keys **before** reading
  `modelslim_config.py` (`../../scripts/checkpoint-key-prefix.py --scales <dir>`),
  and get the serve command: scales confined to a handful of sub-projections with
  nothing declaring a quantisation mean a full-precision release served instead of
  the documented quantised variant (recorded #12675), not a per-layer code bug.
  The **static-W8A8** spelling `…gate_up_proj.deq_scale` is neither of those: only
  the static `W8A8` linear method allocates `deq_scale`, so a `-W8A8` directory that
  *does* ship `quant_model_description.json` is the variant the release validates and
  the engine was never quantised at all — then check that `--quantization ascend`
  reached the **running** command (`docker inspect … .Config.Cmd`; a folded compose
  `command: >` with an inline `#` silently drops every later flag), §B-i/§B-ii of
  `quantization-method-gate.md` (recorded #5812).
- The quant-method lookup dies before any shard bar with `KeyError: '.weight'` — the
  quoted key **starts with a dot** — deepest frames
  `vllm_ascend/ops/vocab_parallel_embedding.py` → `quant_config.get_quant_method` →
  `is_layer_skipped_ascend`, on a quantised checkpoint whose model file builds the
  embedding with `quant_config=` and **no** `prefix=` (llama at vllm v0.13.0 is the
  recorded one; the grep finds 12 such sites, so never narrow it to one family) →
  `empty-prefix-quant-keyerror.md` first: an upstream model file builds its
  `embed_tokens` without `prefix=`, so the lookup asks for `'' + '.weight'`, and a
  construction-time failure cannot be moved by any serve flag, TP size or graph knob.
  Three calls there: does the reporter's own model file pass a prefix (the sibling
  `qwen2.py` / `deepseek_v2.py` do), does `quant_model_description.json` carry
  `model.embed_tokens.weight: "FLOAT"`, and does `lm_head` already pass one. Not §C of
  `quantization-method-gate.md` (that row's key names a real transformer layer) and no blanket
  `quant_description.get(..., "FLOAT")` guard, because the same function also decides
  the quantised linears. A key that carries a module path but names only `lm_head`
  (`KeyError: 'lm_head.weight'` from `modelslim_config.py` `get_linear_quant_type`, on a
  VL model served with `--quantization ascend`) is the **renamed-prefix** variant of the
  same page, not §C: the model file hard-codes `prefix="lm_head"` while its
  `hf_to_vllm_mapper` renames `lm_head.` → `language_model.lm_head.` in the description,
  so no export spelling can satisfy the lookup and editing the description cannot help
  (recorded #8367: v0.18.0rc1 + vllm v0.18.0). When the key **does** carry another module path, settle provenance
  before reading either chain: a directory from the Ascend-native converter
  (`convert_quant_weight.sh … -type qwen_w8a8`) that MindIE itself serves is §C-ii
  (its per-weight description is in `quant_model_description.json`, which a
  0.7.x-era ref never opens), while a foreign format renamed onto `ascend` is §C —
  same frame, same key, different owner.
- The first MoE forward (usually `profile_run`, so it reads as a startup failure)
  dies with `RuntimeError: Promotion for Float8 Types is not supported, attempted to
  promote Float8_e4m3fn and Bool`, deepest frames `w4a8_mxfp4.py` →
  `moe_comm_method.py` → `token_dispatcher.py:393 topk_weights = topk_weights * mask`
  → `fp8-topk-weights-mask-promotion.md`. W4A8-MXFP activations **plus** flashcomm1
  make `topk_weights` fp8, and the AllGather dispatcher multiplies them by a bool EP
  mask. Three source reads, then the flashcomm1-off ablation — no device, HCCL or
  checkpoint component is involved.
- `AttributeError: '<Model>Config' object has no attribute '<attr>'` raised inside
  `vllm_ascend/distributed/*connector*.py` from `ensure_kv_transfer_initialized`
  (typically a VL/MoE model under PD disaggregation) → `connector-config-attributes.md`.
- The startup `TypeError` is `linear(): argument 'input' (position 1) must be Tensor,
  not tuple` (or `not list`) raised in `profile_run`/`determine_num_available_blocks`,
  every failing frame inside the **pinned upstream model file** → `layers/linear.py
  forward → quant_method.apply → F.linear`, and no `vllm_ascend/**` frame but the worker
  → the model file assigned a fused-residual norm's `(x, residual)` return to one name,
  then passed it to the MLP: `model-file-tuple-unpack.md`. Platform-independent (the
  Ascend `RMSNorm.forward_oot` has the same contract), fails on decoder layer 0, so
  `--enforce-eager`, dtype and TP change nothing; and the unpack is only half the fix —
  the repaired upstream form also restores the layer's running-sum/2-tuple contract.
- Python `AttributeError`/`TypeError` in a `vllm_ascend/{ops,patch,models}` frame
  whose class subclasses or re-implements an upstream vLLM/transformers class
  (e.g. `'AscendQwen2DecoderLayer' object has no attribute 'attention_type'`),
  **or the same error inside an upstream vllm file on a class vllm-ascend
  replaces** (e.g. `'ZeroExpertFusedMoE' object has no attribute
  'select_experts'` — the replacement is keyed by class name)
  → `../upstream-downstream-drift/index.md`; this is drift, not a config value.
  **Unless** the attribute is one the pinned upstream `__init__` assigns on every
  path (even to `None`) and the Ascend subclass calls `super().__init__()` inside a
  `@contextmanager` — then the constructor raised and the wrapper swallowed it
  (`'NPUModelRunner' object has no attribute 'mm_budget'` at v0.13.0rc1), which is
  `../upstream-downstream-drift/swallowed-init-exception.md`: the message's own
  sibling is `AssertionError` from the next `assert`, and the reporter recovers the
  real error with a `raise` in that branch.
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
  `acl-graph-capture.md` first. When the raised text is a *guard on an env value*
  (`Do not support TASK_QUEUE_ENABLE = 2 during NPU graph capture, please export
  TASK_QUEUE_ENABLE=1/0.`), that page's Step 0 owns it: the value comes from the
  worker process's own environment, and the attached launcher script is not that
  environment. If instead the startup was clean and the capture error arrives on a
  **request** (`CUDA graph capturing detected at an
  inappropriate time`, `Worker failed with error …`), that is a descriptor the
  startup never captured →
  `../lora-adapters/graph-capture-descriptor-mismatch.md`. The `ACLgraph has
  insufficient available streams` sentence there is canned wrapper text printed
  for every capture exception, and the capture-size set may already be the
  computed maximum.
- The appended inner error is `Cannot run aclop operators during NPU graph
  capture. Current working aclop is <Op>` (+ `ERR00007 PTA feature not
  supported`) → `aclop-op-during-graph-capture.md` first: torch_npu's own guard
  (`NPUGraphsUtils.h:93-105`, called from `OpCommand::Run` only on the aclop
  path), so the subject is one uncapturable op, not the stream budget; the owning
  Python statement is in the FX trace printed beside it, and the message's
  `allow_internal_format = False` advice is overridden by vllm-ascend's
  import-time `True` (`worker/model_runner_v1.py:138` at v0.11.0rc0, `:154` at
  v0.18.0rc1). The escape is skipping capture — and a reporter's
  `--no-enforce-eager` (vLLM parses boolean pairs) is the opposite direction.
- The appended inner error is `capture_begin: … AclmdlRICaptureBegin(capture_stream_,
  capture_mode), error code is 207005` with `[SqCqManage]Alloc sq cq fail,
  stream_id=1984, retCode=0x7020023` under it, and above it CANN's
  `[Error]: Failed to apply for memory.` / `Check the remaining storage space in the
  hardware environment` (= `207005` `ACL_ERROR_RT_RESOURCE_ALLOC_FAIL`; that sentence is
  canned text, not disk or HBM) → `capture-stream-table-exhaustion.md` first: the device
  stream table ran out at capture, A3's default AICPU HCCL unfolding is where the
  formula under-counts, FAQ #18 is the project's own mitigation, and
  `scripts/aclgraph-size-budget.sh` prices the set for the reporter's layer count. Not
  `acl-graph-capture.md`'s `insufficient available streams` family, and not an
  `NPU out of memory`.
- The appended inner error is a **stream synchronize refusal** —
  `AclrtSynchronizeStreamWithTimeout(copy_stream), error code is 107027`, `EE9999 …
  Not allow to synchronize captured-stream, stream_id=N [FUNC:StreamSynchronize]`,
  wrapped in `ERR00100 PTA call acl api failed` → `captured-stream-host-sync.md`
  first: the owner is a data-dependent host readback (`.item()`, `.cpu().numpy()`,
  `bincount`) executing inside the capture window, so `ERR00100` is not a fabric
  fault, and the cheapest discriminator is the **quantization axis** — same launch,
  unquantized checkpoint or `--quantization None`.
- The capture that failed is the **torchair** one —
  `Capturing torchair graph, this usually takes 1.0~3.0 mins.` followed by
  `Error executing method 'compile_or_warm_up_model'` and a
  `torch._dynamo.exc.Unsupported: call_function DelayGraphBreakVariable() …` /
  `Unsupported function call` — → `torchair-dynamo-compile-unsupported.md` first.
  Torchair compiles with `fullgraph=True`, so a dynamo graph break is a hard
  startup failure; the innermost `vllm_ascend` frame (e.g. `dispose_tensor` →
  `x.set_`) names the owner, and the environment pairing plus the PP/ray/graph
  known issue are the two checks before any patch. On a full-precision 671 B MoE
  the per-rank KV budget on the paste's own `Available memory: <N>, total memory:
  <M>` line is a co-equal cause, not a footnote.
- The same torchair capture instead fails **inside GE shape inference** —
  `RuntimeError: E89999: Inner Error!` + `op[<Node>], attr[perm], has wrong
  value[N], it should be less than input shape size[0][FUNC:TransposeCommonInferShape]`
  + `Call InferShapeAndType for node:<Node>(Transpose) failed` +
  `[Compile][Graph]Compile graph failed … [FUNC:ge_api.cc]` —
  → `torchair-ge-infershape-rejection.md`: a host-side graph-build refusal, so
  nothing device-side belongs to it; the node id is GE's, not a symbol, and the
  `torch_npu/dynamo/torchair` + `cann/ge` sources are both outside the exports, so
  the answer is two ablations (drop the `torchair_graph_config` key — `--enforce-eager`
  cannot be added beside it; then the tutorial tab's own W8A8 checkpoint) before any
  attempt to localise the producing `.permute` from the dump.
- Before any capture line, in the **memory-profiling** dummy run of the same torchair
  serve, every rank dies with `IndexError: index 0 is out of bounds for dimension 0 with
  size 0` at `torchair_attention.py`'s `if kv_cache is not None and kv_cache[0].numel()
  > 0:` → `torchair-graph-profile-run-kv-cache-indexerror.md`: the tensor-index wording
  means upstream's unbound placeholder `torch.tensor([])` was indexed, so no KV budget,
  card, HCCL or checkpoint fact applies; the torchair override hands a real decode
  metadata to a prefill dummy forward where the generic builder returns `None` (and the
  guard lacks the `len(kv_cache) > 0` its sibling has). Rung 1 = drop the
  `torchair_graph_config` key (reporter-confirmed #4847), and do not add
  `--enforce-eager` beside it — `check_ascend_config` refuses that pair.
- The worker raises while the **model forward is being compiled**, not at load
  and not at capture: `torch._dynamo.exc.Unsupported: Attempted to call function
  marked as skipped`, `Explanation: Dynamo does not know how to trace the builtin
  posix.putenv.`, `Developer debug context: module: posix, qualname: putenv`,
  and a `from user code` block whose innermost `vllm_ascend` line is an
  `os.environ[...] = ...` (recorded #8224: `ops/rotary_embedding.py:500`
  `TRITON_ALL_BLOCKS_PARALLEL`, reached from `profile_run` → `_dummy_run`) →
  `env-write-in-compiled-forward.md`: vLLM compiles with `fullgraph=True`, so the
  untraceable Python builtin is a hard error; the launch-side fix (pre-set the
  variable) beats any source patch, and a larger `--max-num-batched-tokens` is
  what selects the branch.
- The engine dies at **weight load** on
  `call aclnnMatmulWeightNz failed, error code is 161002` +
  `AclNN_Parameter_Error(EZ1001): Not support mat2 n = 1 or k = 1 when format is
  FRACTAL_NZ, mat2: [2048, 1]`, deepest frames `ops/linear.py` `unquantized_gemm` →
  `F.linear`, often called from a MoE shared-expert gate
  (`ReplicatedLinear(hidden, 1)`) inside `_validate_shared_expert_consistency` →
  `nz-weight-format-refusal.md`: the NZ gate tests dtype/SoC/mode but never shape, so
  a one-dimension weight cannot be stored in `FRACTAL_NZ`. The lever is
  `weight_nz_mode` / `VLLM_ASCEND_ENABLE_NZ` (`0` off, `1` quantised-only default,
  `2` bf16/fp16 too), and `2` is the wrong direction (`0` is reporter-confirmed on
  #10014 and #11621); the resolved mode is in the paste as the
  `AscendConfig.weight_nz_mode …` log line, so read it before asking or blaming a
  driver. `--enforce-eager` and the compilation config cannot touch a load-phase
  raise. Its dtype sibling (fp32 pooling head) is
  `../pooling-embedding/fp32-head-nz-format-crash.md`.

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
