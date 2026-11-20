# Cheap checks, by track (the second half of the procedure step 3)

Moved out of `SKILL.md` procedure step 3, because that list is read per report
and the always-on rules are not. Read `SKILL.md` steps 1-2 and the Lane A/B/C
bullets first, place the report in a lane/track with the classification table,
then find that track here and take its checks **before** source archaeology.
Every entry ends at a topic page; open that page rather than re-deriving its
rule. Entries are added here, not back into `SKILL.md`, so the always-on file
stays under its length cap.

Links below are skill-root relative (`references/<topic>/…`), as they appear in
`SKILL.md`; prepend `<skill_root>` when running one of the scripts.

Rules that carry over from `SKILL.md`: answer text on every turn (rule 1),
pin every code claim to the reported version (rule 5), label the strength of
each claim (rule 6), keep reads narrow (rule 7), and treat a missing artefact
as the finding (rule 10). One axis per launch when you propose an ablation.

## The checks

- Image / bundled-native-module abort (no vllm-ascend frame, glibc allocator
  text): which copy is imported, then allocator interposition, then the
  installer-vs-upstream dependency diff — steps 1-3 of
  `references/image-dependency-faults/native-import-abort.md`. Skip the
  module's C++.
- Worker init dying on `libatb.so` / `libop_plugin_atb.so`: list
  `<ASCEND_HOME>/nnal/atb/set_env.sh`, then run the worker's own
  `_register_atb_extensions()` probe (step 4 of
  `references/image-dependency-faults/nnal-atb-missing-library.md`). Two calls
  split "not installed" from "installed, not sourced", and the NNAL
  `set_env.sh` is a *separate* file each install root must source itself. Do
  not read `torch_npu`'s loader or hand-copy the `.so`.
- `ImportError: <name>.so: cannot open shared object file` raised while the
  connector module itself imports (`ascend_transport.so` from
  `from mooncake.engine import TransferEngine`), in a deployment a launcher
  started (`args:`/`command:`, compose `entrypoint`, an operator CR): ask for
  the pod spec of **both** roles and `cat /proc/1/environ` inside the running
  container before any image theory. The ENV dump describes a shell, not the
  launched process; the export goes in the launch command (steps 1-2 of
  `references/image-dependency-faults/launch-env-and-library-path.md`).
- Report pasting only a response body with thinking in `content` and
  `reasoning_tokens: 0`: the serve command's `--reasoning-parser` is the
  answer before any model/device theory, and the chat field is `reasoning`,
  not `reasoning_content` (checked at vllm v0.19.1 and v0.22.1 — name the
  field the reporter's version emits)
  (`references/accuracy-benchmarks/reasoning-output-routing.md`).
- Native null-string abort at engine init (`std::logic_error` /
  `basic_string::_S_construct null not valid`, no Python frame): stock-model
  run on the same host/image, then the three `python -c` probes, then
  `gdb --batch -ex 'break std::__throw_logic_error' -ex run -ex bt`. Steps 1-3
  of `references/image-dependency-faults/native-string-abort.md` are the whole
  answer; the checkpoint's own config was already proven innocent.
- `ImportError: …/_ops*.so: undefined symbol: <mangled>` from a prebuilt wheel:
  run `bash <skill_root>/scripts/abi-symbol-check.sh <paste>` first — it demangles,
  says old (`Ss`) or new (`NSt7__cxx11`) `std::string` ABI, and prints the three
  `nm` commands plus the matching `cxx_abi_0`/`cxx_abi_1` export. Reporter-confirmed
  on #1144 ("the env setting works for me"), so this is a three-command lane, not
  a rebuild (`references/image-dependency-faults/abi-undefined-symbol.md`).
- The report **is a pip resolution error** (`Could not find a version that satisfies
  the requirement …`, `No matching distribution found for …`, "unmatched torch
  version between vllm=X and vllm-ascend=Y"): `bash
  <skill_root>/scripts/pip-pairing-check.sh <vllm-ascend-root> [vllm-root]` prints
  both sides' `torch`/`torch-npu`/`torchvision` pins, and the `--pre` /
  `--extra-index-url` lines the wheel metadata drops. Decide mirror-only pin vs
  unsatisfiable pin set vs unreachable index **before** any version survey
  (`references/image-dependency-faults/pip-index-visibility.md`, #1796, #4737).
- `pip install` **from source** in a pod with no public network (the custom-op
  step "pulls dependencies from the public network", `fetch failed`, or a cmake
  `Downloading …` line): name the fetch legs instead of opening `csrc/<op>/` —
  the `build_aclnn` → `build_aclnn.sh` → `build.sh` → `csrc/CMakeLists.txt`
  chain fetches a `catlass` submodule plus json / abseil-cpp / ascend_protobuf /
  makeself from `gitcode.com`, each behind an existence gate under
  `CANN_3RD_LIB_PATH`, and neither `CANN_PKG_SERVER` nor an exported
  `CANN_3RD_LIB_PATH` changes that (seed the directory or pass
  `csrc/build.sh --cann_3rd_lib_path=`). Pin v0.13.0rc3; the chain was
  source-read, not run (`references/image-dependency-faults/offline-source-build.md`).
- Startup `ValueError` whose text names a parallel size inside
  `kv_connector_extra_config` (`KV transfer 'prefill' config has a conflicting
  data parallel size. Expected 1, but got 2.`): one grep for the string decides
  the owner (it is `vllm_ascend/utils.py`, not `arg_utils`), then `is_moe` for the
  model decides whether the process could ever keep DP>1
  (`references/startup-failures/arg-validation-errors.md`, #8345). Do not walk
  `ParallelConfig` or the connector: in that version the extra-config numbers
  have no reader besides the check.
- Custom op that aclnn refuses to *resolve* (`EZ1001`, `Get regInfo failed`,
  `does not support opType`, `binary_info_config.json of socVersion`): registry
  and install, before launch. Do not open `csrc/<op>/`: check the tree's
  per-SoC build branch and `def.cpp` `AddConfig` once to rule the source out,
  then `SOC_VERSION`, `_cann_ops_custom`, `ASCEND_CUSTOM_OPP_PATH` and the
  image tag (`references/custom-ops-runtime-faults/op-not-registered-for-soc.md`).
  A **built-in** op instead (`EZ9999` / `Parse dynamic kernel config fail` /
  `<Op>AiCore ADD_TO_LAUNCHER_LIST_AICORE`): the *image's* CANN has no kernel
  for this card — check the release's supported-hardware lines, then one
  standalone op probe (`references/soc-platform-paths/kernel-package-vs-card.md`).
- Startup traceback carrying `torch/_subclasses/fake_tensor.py` + `torch/_ops.py`
  frames and ending `NotImplementedError: <ns>::<op>: attempted to run this
  operator with Meta tensors` (`_C::rotary_embedding` at #1895): two greps decide it
  — `rg -n 'kMeta|TORCH_LIBRARY_IMPL' csrc/` for an existing Meta impl, and the one
  dtype/shape predicate that selects the branch — then answer `--enforce-eager`
  before proposing a patch, with the dtype that skips the branch as rung 2
  (`references/startup-failures/meta-kernel-missing.md`).
- Startup traceback whose deepest frame is **upstream** and ends
  `AssertionError: Torch not compiled with CUDA enabled`, e.g. a `get_rope`
  branch constructing a device in `__init__` (`dual_chunk_rope.py:37` at vllm
  v0.11.0): one call runs
  `bash <skill_root>/scripts/cuda-only-path-audit.sh <vllm-root> <subpath> <vllm-ascend-root>`,
  which lists the unguarded `torch.cuda` hits, whether the file has any
  `current_platform` guard, the `forward_*` implementations, and whether the
  downstream tree replaces the class at all. The `torch==2.7.1+cpu` line in the
  env dump is normal on NPU and is **not** the cause
  (`references/startup-failures/checkpoint-config-cuda-only-path.md`, #4309).
- Degenerate output whose trigger the reporter names as *token content* ("only when
  these special tokens are in the prompt", identical across several releases, one
  model only): refute or confirm that claim from the paste before any source read —
  same ids on both input routes means the ids are not the mechanism, and the
  checkpoint's own `Using max model len N` is the cheap non-engine check
  (`references/accuracy-benchmarks/short-prompt-corruption.md` step 0, #8590).
- Python attribute error whose frame is an **upstream** vllm file (or the
  reporter asks whether vllm-ascend's ops were ever reached): the Ascend
  replacement is name-keyed, so compare the class name against
  `REGISTERED_ASCEND_OPS` / `CustomOp.op_registry_oot` before theorising about
  upstream (`references/upstream-downstream-drift/oot-class-registration-drift.md`).
  If the class *is* registered, or the missing name is a factory/property the
  patch rebinds, check for a pre-patch alias in the consumer module —
  `consumer.<name> is defining_mod.<name>` settles it in one line
  (`references/upstream-downstream-drift/stale-module-binding.md`, #15815).
- `RuntimeError: Invalid request status: <STATUS>` (e.g. `WAITING_FOR_FSM`)
  raised from `vllm_ascend/core/scheduler.py` while the server was serving: the
  `AscendScheduler` fork lags an upstream request status. One call —
  `grep -n '    def '` on the fork and on `vllm/v1/core/sched/scheduler.py` —
  shows the missing methods (`get_grammar_bitmask`), and one more
  (`grep -n "ascend_scheduler_config.enabled = True" …/platform.py`) says whether
  the fork was force-enabled for a default serve (rc0) or is opt-in — that decides
  between a gate-input flag and dropping the key, and whether the three-site patch
  is even needed
  (`references/upstream-downstream-drift/scheduler-fork-missing-status.md`, #3979).
- PD / EPD transport complaint: **first ask whether a KV pool is configured**
  (`AscendStoreConnector` in a `MultiConnector`, `MOONCAKE_CONFIG_PATH`). A pool
  with **no** PD, or one that initialises and then fails while saving, is not
  this row — take the `references/kv-pool-store/index.md` entry below. With a
  P/D pair, the pool decides before the transport: run
  `bash <skill_root>/scripts/kvpool-segment-check.sh <mooncake.json|log>` — the
  worker log's `register mem … mem type:host, length:N` line carries the
  per-rank `global_segment_size`, and ≥ 20 GB on HDK 25.2.3 is the recorded
  cause of `503900` + 乱码 (#13439), fixed at 2 GB
  (`references/pd-disaggregation/kv-pool-store-segments.md`). Only without a
  pool, or once the segment is small, read the rank table with
  `bash <skill_root>/scripts/ranktable-check.sh <table-or-log>` — a declared
  `server_count > 1` (or more than one distinct `server_id`) decides the rest
  (multi-server view, `device_ip` + RoCE required). Then the plog grep set in
  `references/pd-disaggregation/transport-topology.md`.
- KV pool / MemCache fault in storing or fetching KV rather than in the P/D
  handshake — the recorded shape is a **single node with no PD**, 800I/T A3 +
  DeepSeek-V4 `compress_ratios` + `ock.mmc.local_service.protocol: device_sdma`:
  first request fine and the **second** kills the workers (`HCCL watchdog …`,
  `ERR02005`, `EI0002`, `507001`), or the `host_shm` variant dies at store init
  with a "device id" anomaly. Ask for the **resolved** `mmc-local.conf`
  (`MMC_LOCAL_CONFIG_PATH`) and `pip show memcache_hybrid memfabric_hybrid`
  before any backend code: lazy init here is documented, not the bug (#9975), the
  per-die `dram.size` / `max.dram.size` / 1GB alignment and the non-existent
  `standalone.dram.size` are real config faults, and the HCCL line is downstream
  of a swallowed rank-local save (#14106, family only — the identity is not
  established). CQE / SDMA / device-id text is owned by MemCache/MemFabric, so
  name that owner and ask for its log
  (`references/kv-pool-store/memcache-a3-dsv4-init.md`).
- A pool **put** never lands while the request succeeds (`Failed to invoke
  HcclBatchPut, hccl_result = 4`, `Failed to put N keys out of N`, master
  `PutEnd` 0): one call —
  `bash <skill_root>/scripts/store-put-code-check.sh <paste>`. `4` is
  `HCCL_E_INTERNAL` and the store's `-1` is its generic code, so it is neither a
  parameter error nor capacity (`-200` is the out-of-space code), and the emitting
  string is in the `mooncake-transfer-engine-npu` wheel, not in any export
  (`references/kv-pool-store/hccl-onesided-store-put.md`).
- Report is "the flag did nothing" (logging/verbosity, output printing, a
  missing log stage): name the owner first — upstream vLLM OpenAI frontend,
  not vllm-ascend — then walk the gate checklist in
  `references/observability-and-logs/index.md`: both flags on the serving
  command, `--enable-log-outputs` requires `--enable-log-requests`,
  INFO-vs-DEBUG split, `--max-log-len`, a log-config/`VLLM_CONFIGURE_LOGGING`
  override, and an endpoint that cannot log outputs at all. Grep the two
  warning strings, then answer; do not read `vllm/entrypoints/**` for an
  Ascend-specific cause.
- A startup line `Unknown vLLM environment variable detected: VLLM_ASCEND_<NAME>`
  with the engine otherwise healthy (reporters call it 误报): upstream's
  `envs.validate_environ`, warn-only because `fail_on_environ_validation` defaults
  `False`. The name really is in that process's environment, and vllm-ascend never
  merges its own registry into upstream's, so any `VLLM_ASCEND_*` warns. One
  `env | grep -i vllm_ascend`, one owner sentence (the merge belongs in
  `AscendPlatform.pre_register_and_update`), and say that `collect_env.py` never
  prints `VLLM_ASCEND_*` so its absence proves nothing
  (`references/observability-and-logs/env-var-validation-warning.md`).
- An endpoint that is missing rather than wrong (`POST /start_profile` →
  `404`, or a 500 `Profiling is not enabled`): a route-registration question
  in the serving process. Name `attach_router` (`path:line`) and its condition,
  then ask the live app: `curl -s <host>:<port>/openapi.json` filtered for
  `profile`; check the startup log for the gate's `warning_once` line, and only
  then wonder where the flag went (`references/observability-and-logs/online-profiling-endpoints.md`).
  Model class, quantization and NPU health are irrelevant to a 404 — say so.
- Output that is fluent but stops after tens of tokens ("被截断") with a feature
  flag toggled nearby: `finish_reason` first (`length` = budget, `stop` = early
  EOS), then `rg -n "<key>"` every reader of that key and read the shipped
  support matrix before any mechanism
  (`references/accuracy-benchmarks/early-eos-short-output.md`).
- A spec-decode symptom (acceptance collapse, a stall while drafts keep being
  produced, "works if I remove MTP"): order the ablations MTP off →
  `--async-scheduling` off → `--no-enable-prefix-caching` → `num_speculative_tokens 1`
  → concurrency/`--max-model-len`, and do not lead with the KV arithmetic:
  acceptance dropping to exactly 0.000 everywhere while drafts continue is a
  draft/target context question, and the recorded reporter confirmed the owner by
  removing MTP (`references/spec-decode-mtp/acceptance-collapse.md`).
- A **tool-call** report (client JSON error, "参数被截断", 格式错误, 重复输出, or
  only CoT with `stream: true`): locate the pasted `data:` chunks or ask for one
  `curl -N` capture, then run
  `bash <skill_root>/scripts/sse-toolcall-audit.sh <file|->` **before** any parser
  or sampler theory. Its outcome names the layer: assembly parses = client,
  tail missing = the stream ended early (split with `return_token_ids: true`),
  **balanced braces with an odd quote count and the gap inside a value = the
  parser's delta, keyed on that argument's declared schema type** (run
  `uv run --no-project python <skill_root>/scripts/tool-schema-type-audit.py
  <request-body.json>`; a nullable `anyOf: [string, null]` argument loses its
  opening quote, #12261), other mid-string damage = generation. Say which, with
  the assembled tail, in the first answer
  (`references/tool-call-streaming/streamed-arguments-audit.md`,
  `arg-type-gate-unquoted-value.md`).
- If that report's failing `num_speculative_tokens` is >= 3 while 1 or 2 works,
  name `vllm_ascend/sample/rejection_sampler.py:114 using_block_verify =
  max_spec_len >= 3` (the line is pin-dependent: `:116` at v0.18.0) in the same
  answer with the `n=2` and `temperature: 0`
  ablations; walk the kernel afterwards, never instead
  (`references/tool-call-streaming/mtp-block-verify-acceptance.md`).
- A **first** response with `content: ""`, `completion_tokens: 1` and
  `finish_reason: "stop"`: that is one model-issued blank/stop token. Ask for the
  emitted token id (`return_token_ids: true`), a restart with the identical body
  twice, and a warm-up request row before attributing it to one-shot server state;
  offer `min_tokens: 1` for the case where the id turns out to be EOS. Diff the
  launcher against the family sibling tutorial first (`docs/source/tutorials/models/`),
  and do not re-derive the scheduler padded-rows/null-block theory, it is already
  refuted. The MTP proposer's *padded drafter batch* is a different story and still
  live: `disable_padded_drafter_batch: false` (the default) is what keeps the draft
  model replaying aclgraph, so `"enforce_eager": true` inside `--speculative-config`
  and `"disable_padded_drafter_batch": true` are the two draft-side-only rows
  (`references/spec-decode-mtp/empty-first-response.md`).
- 310P / Atlas 300I report: take the SoC from
  `_build_info.__soc_version__` rather than the image tag, quote the
  documented status (experimental, stable = v0.10.0rc1 at v0.11.0rc3,
  `faqs.md:18`, `single_node_300i.md:3-7`) as the first sentence, and then
  apply the 300I environment (`--enforce-eager`, `--dtype float16`,
  `PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256`) and a smaller
  `--max-model-len` before blaming the platform
  (`references/soc-platform-paths/atlas-300i-310p.md`). Write the allocator value
  explicitly — on 310P `expandable_segments:True` crashed every rank at
  `empty_cache` (`unmapHandles … 507899`) while `:False` deployed
  (`references/runtime-memory/allocator-expandable-segments-unmap-crash.md`) — and
  with TP > 1 quote the graph budget before any HCCL theory
  (`docs/source/tutorials/hardwares/310p.md:20-30`). Then diff the command against
  the release's own validated table (`310p.md:95-176` at v0.22.1rc1: only the
  `Eco-Tech/Qwen3-{8B,14B,32B}-w8a8sc-310-vllm` rows, `--dtype float16
  --quantization ascend --load_format sharded_state --no-enable-prefix-caching
  --max_model_len 16384`) — a VL or fp16 dense serve on 310P is outside it, and
  "run the doc's row on a small model" is rung 1. Two traps when reading that
  paste: the doc's `--additional-config … fuse_norm_quant: false` is inert on 310P
  (`compilation/graph_fusion_pass_manager.py:54`, `and not is_310p()`), so the
  `Enabled custom fusions: …` line is upstream's `pass_config` echo
  (`vllm/config/compilation.py:313`) and not evidence; and under `--enforce-eager`
  no fusion pass runs at all.
- Device allocation failure before the engine is ready
  (`Memory_Allocation_Failure(EL0004)`, `runtime result = 207001`, `507013`
  plus SDMA/SMMU text) at a KV-cache or weight `npu_format_cast`: read
  timestamps, treat the cast frame as the allocation site, and shrink the pool
  (`--max-model-len`, then `--gpu-memory-utilization`) before any
  driver/hardware theory
  (`references/runtime-memory/kv-init-and-format-cast-alloc.md`).
  If instead the frame is `torch.npu.empty_cache()` and the text is
  `unmapHandles … 507899`, nothing failed to allocate: `grep -m1
  'Set PYTORCH_NPU_ALLOC_CONF' <serve log>`, then relaunch with an explicit
  `expandable_segments:False`
  (`references/runtime-memory/allocator-expandable-segments-unmap-crash.md`).
- Runtime memory after a healthy period: take the two-sample split (capacity
  vs growth) and `grep -m1 'Set PYTORCH_NPU_ALLOC_CONF' <serve log>` first, and
  check `additional_config` key liveness (`rg -n '<key>' vllm_ascend/`) before
  any leak theory (`references/runtime-memory/hbm-growth-and-oom.md`).
- Throughput that falls in **one step** at a concurrency number (256 running
  requests fine, 258 collapsed; `Waiting` 0, KV under 90 %, no error text, NPUs
  idle and host cores busy): run
  `bash <skill_root>/scripts/seq-cliff-capture-budget.sh --max-num-seqs N
  --num-spec K` before any graph/compile source — it prints the sequence count
  where `cudagraph_dispatcher` silently returns `NONE`, and the value to put in
  `--compilation-config '{"max_cudagraph_capture_size": N×(1+K)}'`. Then one grep
  for the `max_cudagraph_capture_size … is smaller than the potential max tokens`
  startup line, then the A/B. Ablations that move the baseline
  (`--enforce-eager`) or leave the ceiling unchanged (FlashComm1, DSA-CP,
  `npugraph_ex`, `--async-scheduling`) cannot refute it, so quote them only
  against their own below-cliff N
  (`references/performance-throughput/sequence-count-cliff-capture-size.md`,
  #14440).
- "The machine crashed" / the host dropped / a `top` paste whose python3 ranks
  each hold tens of GiB RES: run
  `bash <skill_root>/scripts/host-ram-check.sh <paste|->` and report the ratio
  before any NPU lane, quoting RES and never the `8326g` VIRT
  (`references/runtime-memory/host-ram-and-per-rank-rss.md`).
- Same crash but **no** `top` to sample and the log stops mid-startup: last lines
  are the per-rank `rank N in world size M is assigned ...` set, no
  `Starting to load model` after them, `--max-model-len` 128K+. Run
  `bash <skill_root>/scripts/attn-mask-budget.sh --log <serve.log> --host-gib N`:
  the eager `max_model_len²` host mask is 9·N² bytes **peak per rank** before any
  weight load, so at 262144 nothing survives
  (`references/runtime-memory/attention-mask-host-allocation.md`).
- A **container** counter climbing for days on a healthy service
  (`container_memory_working_set_bytes` "持续增加"), no crash and `npu-smi` flat:
  run `bash <skill_root>/scripts/working-set-arithmetic.sh --growth <reported>
  --days <window> --rps <rate>` for the bytes/request class, ask for the three
  cAdvisor series at two timestamps plus a per-PID RSS sample, and diff the
  reporter's serve flags against the model tutorial's own recipe before any
  source read (`references/runtime-memory/container-working-set-growth.md`).
- The benchmark/eval client's own traceback is the report ("推理正常、性能测试崩溃",
  `ConnectionRefused [Errno 111]`): treat the client as secondary and locate the
  first in-window server fault — the API process does not listen because it died.
  If the server log shows `!!!!!!! Segfault encountered !!!!!!!` plus
  `died unexpectedly (exit code: None)`, take
  `references/hccl-and-hardware/worker-segfault-no-python-frames.md`: count the
  `vllm_ascend/*.py` frames in the dump (zero = call site unknown, ask for the
  plog and an `--enforce-eager` + `ASCEND_LAUNCH_BLOCKING=1` rerun), and do not
  read the attention backend the top native frames happen to sit under. The same
  marker at **engine start** (`Failed core proc(s): {'EngineCore': -11}`, no
  `Starting to load model`, a trivial CPU `torch.zeros`/`fill_kernel` at the top,
  the `TBE Subprocess … main process disappeared` block after it) is the same lane
  from the child's side: `-11` is SIGSEGV, the fill frame is where damage
  surfaced, and the answer is the image/CANN pairing plus the release's
  doc-validated command — section `## The startup variant` of that page.
- Any pasted plog / RUNTIME excerpt that carries more than one error family: run
  `bash <skill_root>/scripts/plog-timeline.sh --since <report date> <log>` before
  naming a cause. Blocks it flags as older are either the previous crash or this
  process's own `init engine (...)` / graph-capture window recorded at init and
  reported late — decide by pid and log position, not by the gap (rule 13); the
  in-window first error is the one to classify
  (`references/hccl-and-hardware/error-report-timestamps.md`).
- First collective after the weights load dies on `Transport_Init_Error(EI0009)` /
  `[Create][DestLink]Create Dest error!`, with `halShmemOpenHandle … drvRetCode=17` and
  `rtsIpcMemImportByKey … 507899`: run
  `bash <skill_root>/scripts/hccl-link-pair-audit.sh <log>` — it lists the failed
  `rank→rank` edges, the driver codes, and classifies them same-card vs cross-card
  against the `npu-smi` Bus-Ids. All-cross-card with the card-local edges up is a host
  transport/driver pairing question (driver vs the image's CANN, container
  passthrough), and the identical `localIpAddr`/`remote_ip_addr` text is **not** a
  duplicate `device_ip` — the rank table is `scripts/ranktable-check.sh`'s job
  (`references/hccl-and-hardware/transport-destlink-ipc-import.md`, #4148).
- A device fault whose deepest frame is a `torch_npu` scatter/index write in
  `vllm_ascend/attention/sfa_v1.py` (`507035`, `ERR00100`, `EZ9999 … aivec error`,
  `The DDR address of the MTE instruction is out of range`, fault kernel
  `ScatterNdUpdate_…`, one `pc start` on every core): compare the index tensor's row
  count with the source tensor's before anything else. In #4296
  `attn_metadata.slot_mapping` carried the whole batch while `indexer_select` was
  called once per decode/prefill half, so a mixed batch read past `k` — and the
  deployment already ran `--enforce-eager` + `ASCEND_LAUNCH_BLOCKING=1`, which only
  made the attribution exact. The decisive check is an A/B on request concurrency
  (`--max-num-seqs 1` versus the failing value), not a plog
  (`references/dsa-sparse-attention/indexer-cache-scatter-fault.md`).
- Engine-start `AttributeError: 'AscendRowParallelLinear' object has no attribute
  'aclnn_input_scale'` (or `_reciprocal` / `_offset`) at `sfa_v1.py:707` ← `:544`,
  with FLASHCOMM1 / legacy FLASHCOMM / shared-expert-DP on and a DSA model: the
  DSA-CP o_proj TP/Full switch exists only for a W8A8-static o_proj, so the answer
  is the quant method the checkpoint forces (FLASHCOMM1 off, or the tutorial's W8A8
  weights), not an attention or install fault
  (`references/dsa-sparse-attention/o-proj-quant-attribute-crash.md`).
- Degenerate output (`</think>` in `content`, repetition collapse, 乱码) at long
  context **and** concurrency on a model the tree documents, with both `vllm serve`
  commands pasted: open `docs/source/tutorials/models/<model>.md` at the pinned
  commit and diff the reporter's P and D commands against the validated ones plus
  the `Recommended Configurations` tables **before** reading any connector — that
  diff, with the ablation matrix, is the first answer
  (`references/accuracy-benchmarks/garbled-long-context-output.md`). Recorded
  #13002 spent 21 turns and its whole output budget inside a 2600-line
  `mooncake_connector.py` and shipped nothing, though the tutorial's validated
  commands deviated on five flags and were reachable in two calls.
  If instead the *prompt* is short, the *completion* runs long and a few
  responses out of many repeat themselves, that is not the prompt-length lane:
  diff the same tutorial for the decode side and lead with the identity row
  (`--max-num-seqs 1`, `temperature: 0`, same request 3×)
  (`references/accuracy-benchmarks/long-output-repetition.md`).
- A crash under concurrency with **no** error text pasted: do not hunt the mechanism.
  The answer is the gap, the config diff against the tree's own validated command
  (`tests/e2e/**`, `docs/source/tutorials/<model>.md`) and ranked candidates with one
  check each, emitted at the normal gate
  (`references/dsa-sparse-attention/concurrency-crash-with-no-log.md`).
- A "communication timeout" whose only text is a bare `TimeoutError` with
  `multiproc_executor.py` / `shm_broadcast.py` frames: run
  `bash <skill_root>/scripts/executor-timeout-check.sh <paste|->` before any
  HCCL reading. It is the framework's own RPC watchdog (300 s) for a worker that
  stopped answering, so the deliverable is the lane classification plus the list
  of per-rank artefacts — which rank, which op and OOM-vs-hang are *not* in that
  paste (`references/hccl-and-hardware/rpc-watchdog-timeouts.md`, #1865).
- `TypeError: 'NoneType' object is not callable` where the `None` is an upstream
  module-level name (`fused_experts = None` on a non-CUDA platform): read the
  **file** of the `forward_oot` frame. `vllm/model_executor/custom_op.py` there
  means the Ascend replacement was never imported in that process (Ray/`spawn`
  workers; the hook runs only where `VllmConfig` is built) — two greps decide it,
  do not theorise about the model or the checkpoint
  (`references/upstream-downstream-drift/patch-import-side-effect.md`, #253).
- Quantization method rejected before any worker (`<method> quantization is
  currently not supported in npu`, `--quantization awq`): answer from
  `references/startup-failures/quantization-method-gate.md` — three calls, and
  the flag is not the only trigger, since the checkpoint's own `quant_method`
  reaches the same gate. The same page's section B is the inverse report: an
  msmodelslim W8A8 checkpoint whose `config.json` deliberately has **no**
  `quantization_config` started without `--quantization ascend` /
  `quantization="ascend"` and died in a worker (#2015) — the flag is mandatory
  exactly because upstream's auto-detection has nothing to read.
- Deployment sizing / stage layout (does it fit on N cards, why a stage
  layout fails): run `bash <skill_root>/scripts/card-budget.sh` with the
  checkpoint's parameter count, the TP and the per-stage
  `gpu_memory_utilization`/device lists, and report the two numbers it prints
  (weights per card vs card budget; per-device gmu sum). Check the device
  variable first: `ASCEND_RT_VISIBLE_DEVICES`, not `CUDA_VISIBLE_DEVICES`
  (`vllm_ascend/platform.py:101` at v0.19.1rc1;
  `references/deploy-sizing/card-budget-and-device-visibility.md`). A sibling
  project's yaml/CLI absent from `SOURCES.json` (vllm-omni) is a stated gap,
  not something to infer.
- A parallel-topology question about EP **with** DP, or an engine-init crash whose
  deepest frame is the MoE `prepare/finalize` `dist.all_gather` on a 910B DP×TP
  deployment: read `_select_moe_comm_method`
  (`worker/model_runner_v1.py:1373-1432`) and `set_mc2_tokens_capacity`
  (`ascend_forward_context.py:169-183`) first, then answer with that SoC's row,
  the reporter's `world_size_across_dp` arithmetic and the two ablations
  (`references/feature-enable-recipes/moe-expert-parallel-with-dp.md`); say that
  MC2 shares the All2All `finalize`, so the pasted line is not proof MC2 was
  skipped. If instead the paste dies in `profile_run` with `507014` and the
  `[AIC_INFO]` kernel is `MoeDistributeDispatchV2`, the batch is
  `mc2_tokens_capacity` (an extra `_dummy_run` at
  `model_runner_v1.py:2202-2208`), not `--max-num-batched-tokens` — take
  `references/hccl-and-hardware/mc2-dispatch-profile-run-hang.md`.
- Feature-enable / capability question ("how do I correctly enable F on H", no
  log): answer in this order — the validation verdict at the pinned ref (docs
  sentence, the hardware the docs measured, `rg -l
  "prefill_context_parallel_size|prefill-context-parallel-size" tests/` for e2e
  coverage, precedent issues with their versions), then each enforced constraint
  with the error text it raises, then one-axis-at-a-time staging. Grep the flag in
  `vllm_ascend/` for hard errors (`AssertionError`, `NotImplementedError`) versus
  silent overrides; do not read the kernel/code path beyond that
  (`references/feature-enable-recipes/index.md`).
- Code-change / cleanup request (remove a `VLLM_ASCEND_*` switch before a
  release, deprecate a flag, finish a "TODO before 0.x"): one `rg -n '<IDENT>'`
  and one grep for the *mechanism* token at the pinned ref decide answerability —
  absent means the snapshot predates the PR, which is the finding. Then the
  touch-point sweep (definition dict + generated doc page + readers + tests/CI
  env sets + release note), the default-flip note and the reachability check
  (`references/code-change-requests/env-flag-retirement.md`, #1262). Do not open
  the op's code path.
- Community-test / doc-verification task: run the ten checks of
  `references/tutorial-verification/tutorial-command-audit.md` in order
  (flag existence at the pinned vllm, bench argument names, image WORKDIR vs
  the doc's relative paths, relied-on defaults, card-count arithmetic vs
  sibling docs and the CI config, baseline release, one precedent pass,
  package version the doc installs vs the image's own pins, support matrix and
  release notes, `--hf_overrides` architecture string vs the checkpoint family)
  and write the report file as sections become
  true (`references/tutorial-verification/community-test-report.md`). Stop
  auditing after 3-5 tiered defects; never present a doc baseline as a
  measurement.
- Accuracy/corruption: establish whether both numbers came from the same
  server, weights and serve flags before choosing a culprit
  (`references/accuracy-benchmarks/index.md`); for a garbled prompt, get the
  ablation matrix (`references/accuracy-benchmarks/garbled-long-context-output.md`).
- An **embedding / pooling** checkpoint that never starts: read the launch line for
  `--task embed` (`--runner pooling`) and the init line for `pooler_config=None`
  before any source; `ValueError: There is no module or parameter named
  'embed_tokens' in …ForCausalLM` is the missing flag, not the model. One call:
  `../../scripts/checkpoint-key-prefix.py <model dir>`
  (`references/pooling-embedding/embedding-checkpoint-as-generate.md`).
- An **embedding / pooling** serve (`--runner pooling`, `--task embed|score`,
  `/v1/embeddings`): two questions place it before any source read — single
  request or only under concurrency, and is a short prompt correct. Crashes under
  concurrency end in the batch-slot machinery
  (`references/pooling-embedding/batch-index-desync.md`); wrong vectors for long
  inputs take the "which span was pooled" test plus a
  `--max-num-batched-tokens` sweep, not an attention-backend walk
  (`references/pooling-embedding/chunked-prefill-vector-corruption.md`).
- The same serve **never reaches ready**, and the paste carries either a model
  class/HF-config `TypeError` in the APIServer or an `EZ1001` weight-layout line
  at `profile_run`: two greps decide both before any source walk — the
  `architectures` string against the checkpoint's `model_type` family
  (`references/pooling-embedding/seq-cls-arch-routing.md`) and the NZ gate at the
  reporter's ref, `rg -n "def is_enable_nz|_should_trans_nz|maybe_trans_nz"
  vllm_ascend/utils.py`, whose fp32 policy decides whether the fp32 score head is
  converted (`references/pooling-embedding/fp32-head-nz-format-crash.md`). One
  call each: `scripts/nz-format-gate-check.sh <paste> <vllm-ascend-root>` and
  `scripts/seq-cls-arch-check.sh <vllm-root> '<Arch>'`. The `EZ1001` code alone
  does not mean the custom-op package.
- A **VL image/video** request that degenerates into one repeated token
  (`!!!!`, 感叹号) while text-only prompts answer: first row of the ablation is
  `USE_OPTIMIZED_MODEL=0` on the same request, then text-only, TP=1 and an
  official checkpoint (`references/accuracy-benchmarks/vl-vision-tower-degeneracy.md`).
- `patch/**` frame: extract-and-diff first — `sed -n 'A,Bp'` the pinned upstream
  method, dedent with `sed 's/^    //'`, diff the module-level copy, read only the
  non-formatting hunks, then check whether the one added hunk is live for the
  reporter's command (PD `is_kv_consumer`, `use_async_scheduling`). Formatting-only
  means the owner is upstream at that pin: say so with both `path:line`. #8724
  spent 29 proxy turns here and shipped nothing
  (`references/upstream-downstream-drift/patch-copy-vs-upstream.md`).
- Worker `KeyError` in `_update_states` + `NoneType … sampled_token_ids` + a
  negative `Counters …` value on an `--async-scheduling` + MTP launch: one
  provenance (discarded row → count 0 → `num_accepted = -1` → missing
  `prev_req_id_to_index` key). Say that, then async off
  (`references/spec-decode-mtp/async-spec-bookkeeping-desync.md`).
- Device dump whose deepest `vllm_ascend` frame is the drafter's valid-count
  copy (`counts_cpu[: counts.shape[0]].copy_(counts, non_blocking=True)`,
  `model_runner_v1.py:1393`) with `rtMemcpyAsyncEx … model stream execute
  failed` / `aivec … trap`: the copy is the sync point, and landing on it
  **proves** `--async-scheduling` + spec decode + padded drafter batch, so do not
  ask. Resolve the collective by `rankSize` against the DP×PCP×TP group
  arithmetic (a 2-rank group is not `fc3_quant_x` on DP2×TP4), then attribute
  with `--enforce-eager` and `ASCEND_LAUNCH_BLOCKING=1`
  (`references/spec-decode-mtp/drafter-count-copy-device-fault.md`).
- Device fault inside an attention backend on a batch holding decodes **and**
  prefills: run `bash <skill_root>/scripts/batch-split-audit.sh <vllm-ascend-root>`
  before the kernel; its hits are the request tensors sliced by a token count and
  the unrebased `cu_seqlens`, and spec decode is the multiplier that turns a
  silent overshoot into `507015` (`references/mixed-batch-splits/index.md`).
- A **LoRA** serve, either shape: run
  `bash <skill_root>/scripts/lora-graph-descriptor-audit.sh <vllm-ascend-root>
  [<vllm-root>]` first. It names the punica backend the ref binds (that decides
  the slow shape) and labels each `has_lora` producer live-batch vs
  config-constant (that decides the request-time capture crash); the SoC and
  graph-mode facts come from the paste, not from more source
  (`references/lora-adapters/index.md`).
- An operator that **runs** and returns wrong numbers with a framework-free
  `torch_npu` repro (no dump): find the one caller holding that kwargs
  combination before reading the kernel; `cann/ops-nn` is usually absent, so
  the ownership split is the answer
  (`references/custom-ops-runtime-faults/attention-op-numeric-errors.md`).
