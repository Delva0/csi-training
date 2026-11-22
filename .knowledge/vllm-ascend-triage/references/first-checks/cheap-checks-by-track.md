# Cheap checks, by track (the second half of the procedure step 3)

Moved out of `SKILL.md` procedure step 3, because that list is read per report
and the always-on rules are not. Read `SKILL.md` steps 1-2 and the Lane A/B/C
bullets first, place the report in a lane/track with the signature index
(`signature-routes/index.md`), then find that track here and take its checks **before**
source archaeology.
Every entry ends at a topic page; open that page rather than re-deriving its
rule. Entries are added here, not back into `SKILL.md`, so the always-on file
stays under its length cap.

Links below are relative to this directory (`../<topic>/<page>.md`); prepend
`<skill_root>` when running one of the scripts.

Rules that carry over from `SKILL.md`: answer text on every turn (rule 1),
pin every code claim to the reported version (rule 5), label the strength of
each claim (rule 6), keep reads narrow (rule 7), and treat a missing artefact
as the finding (rule 10). One axis per launch when you propose an ablation.

## The checks

- **The paste is a claim; the engine's own dump is the record.** Whenever the
  traceback comes with the `dump_input.py` block (`Dumping input data for V1 LLM
  engine (v…) with config: …`, then the scheduler output and stats), diff it against
  the launcher the reporter pasted before ranking any mechanism: #9359's command said
  `num_speculative_tokens: 1` while the dump said `num_spec_tokens=3`, and its decode
  node claimed `--max-model-len 172032` while the effective cap was 133120 — every
  ladder row written from the script would have tested a launch that never ran. Quote
  the dump's numbers, name the disagreement, and ask for `args: Namespace(...)` /
  `/proc/<pid>/cmdline`. Two further uses: the dump's *field spellings* fingerprint the
  installed wheel against the pin (vllm v0.19.1 `ModelConfig` has `max_model_len`,
  `vllm/config/model.py:182`, not `max_seq_len`, so a `max_seq_len=` line needs
  confirming), and the scheduler-output fields classify the failing step
  (`num_computed_tokens`, `num_scheduled_tokens`, `scheduled_spec_decode_tokens`)
  without a source read — worked rows in
  `../spec-decode-mtp/max-model-len-boundary-fia-crash.md` and
  `../hccl-and-hardware/aclgraph-full-decode-runtime-hang.md`.
- **The log's own process prefixes recover the launch that ran, so a mismatched
  paste is caught for free.** `EngineCore_DP<N>` / `Worker_DP<N>_EP<N>` names exist
  only when that parallel size is > 1 (`vllm/v1/engine/utils.py:136,151` at vllm
  v0.22.1), so `EngineCore_DP2` beside a command carrying `--data-parallel-size 1`
  is two different runs — #11300 pasted exactly that pair. When the disagreement is
  *which hardware tab* the block came from (one page, A2 dp1/tp8+`enable_dsa_cp` vs
  A3 dp4/tp4+`--api-server-count`, and a `-a3` image), one call gives the diff and
  the closest tab: `uv run --no-project python scripts/recipe-tab-diff.py --page
  <page> --section 5.1 --command <reporter script>`. Step 0 of
  `../image-dependency-faults/native-string-abort.md` is where that lands for a
  startup abort; a config-review task can use the same output as the finding.
- Image / bundled-native-module abort (no vllm-ascend frame, glibc allocator
  text): which copy is imported, then allocator interposition, then the
  installer-vs-upstream dependency diff — steps 1-3 of
  `../image-dependency-faults/native-import-abort.md`. Skip the
  module's C++.
- Worker init dying on `libatb.so` / `libop_plugin_atb.so`: list
  `<ASCEND_HOME>/nnal/atb/set_env.sh`, then run the worker's own
  `_register_atb_extensions()` probe (step 4 of
  `../image-dependency-faults/nnal-atb-missing-library.md`). Two calls
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
  `../image-dependency-faults/launch-env-and-library-path.md`).
- An `ImportError` naming an attribute of **`vllm` itself** — `cannot import name
  '<X>' from 'vllm'`, `(unknown location)` included, from `from vllm import …` in
  an upstream frame — is answered by reading the message's location text and the
  **paths** of the traceback's own frames, before any version survey: a frame
  resolving `vllm` from a checkout (`$HOME/…/vllm/vllm/…`) plus `(unknown location)`
  says a bare `vllm` directory was imported and no regular package was reachable,
  so the fix is deleting/renaming that directory; the `…+empty` version in the env
  dump is the docs' own `VLLM_TARGET_DEVICE=empty` marker, not a cause. One
  `python -c "import vllm, sys; print(sys.executable, vllm.__file__, list(vllm.__path__))"`
  in the reporter's shell and launch directory decides it
  (`../image-dependency-faults/wrong-vllm-module-shadowed.md`).
  The **submodule** form is the other case and takes the opposite reading:
  `cannot import name '<X>' from 'vllm.v1.worker.utils'` where that file *exists*
  means the symbol moved between vLLM releases (`bind_kv_cache` is at
  `vllm/v1/utils.py:278` in vllm 0.9.2 and `vllm/v1/worker/utils.py:120` in vllm
  0.10.0), i.e. the two packages are a mismatched **pair** — pair them, do not edit
  the import and do not assume the plugin build is stale
  (`../startup-failures/environment-correspondence.md`, its nearest-tag section).
- Log stops after `Loading model weights took X GB` with the port still refused
  and nothing raised: bound the phase before theorising — at v0.8.4rc1 the next
  code is `worker.py:240-256` `determine_num_available_blocks` → `profile_run()`
  ending in `torch_npu.npu.synchronize()` (`model_runner.py:1078`), `initialize_cache`
  is next, and `_warm_up_model` (`worker.py:325-326`) captures nothing at that ref.
  Then ask for the exact `vllm serve` line, the container exit code, `dmesg` OOM
  lines and `npu-smi` **during** the stall (`../startup-failures/silent-stall-after-weight-load.md`).
  The log's `model_runner.py:<line>` vs `model_runner_v1.py:<line>` fingerprints
  V0 vs V1; use it as evidence, not proof.
- `NPU out of memory` raised **while building the model** (`get_model` →
  `quantization/methods/w4a16.py:get_weight` → `torch.empty`), a 384-512-expert MoE,
  `--enable-expert-parallel` and a `--data-parallel-size` in the launch dict: the
  cheapest check is two greps of the worker log (`[EP Rank r/N] … Local/global number
  of experts`, `rank … in world size … assigned as … EP rank …`), because the expert
  shard width is `dp × pcp × tp` only when the DP ranks share a DP group and a TP-only
  shard is 2× the per-card bytes. Do not decode the refusal's byte count into an
  expert count — `../deploy-sizing/moe-load-oom-expert-shard.md`.
- A DP>1 MoE deployment whose dummy/`profile_run` batch passes but a **real
  request** hangs a layer or two in, or whose layer output is garbage of the right
  shape right after a `(dp_size × max_tokens_across_dp, hidden)` tensor: the pad
  target must be the DP-wide max on every rank and the owner slice is per-rank —
  print `dp_rank`/pad target/local `num_tokens`/slice offset on **all** ranks,
  recompute `dp_size × max_tokens_across_dp` against the gathered row count, and
  read `get_fused_moe_state` before blaming the gather (EP at `ep_size < 16` is
  `All2All`, so the DP `all_gather` does not even run)
  (`../feature-enable-recipes/dp-moe-padding-and-owner-slice.md`).
- Report pasting only a response body with thinking in `content` and
  `reasoning_tokens: 0`: the serve command's `--reasoning-parser` is the
  answer before any model/device theory, and the chat field is `reasoning`,
  not `reasoning_content` (checked at vllm v0.19.1 and v0.22.1 — name the
  field the reporter's version emits)
  (`../accuracy-benchmarks/reasoning-output-routing.md`).
- Native null-string abort at engine init (`std::logic_error` /
  `basic_string::_S_construct null not valid`, no Python frame): stock-model
  run on the same host/image, then the three `python -c` probes, then
  `gdb --batch -ex 'break std::__throw_logic_error' -ex run -ex bt`. Steps 1-3
  of `../image-dependency-faults/native-string-abort.md` are the whole
  answer; the checkpoint's own config was already proven innocent.
- `ImportError: …/_ops*.so: undefined symbol: <mangled>` from a prebuilt wheel:
  run `bash <skill_root>/scripts/abi-symbol-check.sh <paste>` first — it demangles,
  says old (`Ss`) or new (`NSt7__cxx11`) `std::string` ABI, and prints the three
  `nm` commands plus the matching `cxx_abi_0`/`cxx_abi_1` export. Reporter-confirmed
  on #1144 ("the env setting works for me"), so this is a three-command lane, not
  a rebuild (`../image-dependency-faults/abi-undefined-symbol.md`).
- The report **is a pip resolution error** (`Could not find a version that satisfies
  the requirement …`, `No matching distribution found for …`, "unmatched torch
  version between vllm=X and vllm-ascend=Y"): `bash
  <skill_root>/scripts/pip-pairing-check.sh <vllm-ascend-root> [vllm-root]` prints
  both sides' `torch`/`torch-npu`/`torchvision` pins, and the `--pre` /
  `--extra-index-url` lines the wheel metadata drops. Decide mirror-only pin vs
  unsatisfiable pin set vs unreachable index **before** any version survey
  (`../image-dependency-faults/pip-index-visibility.md`, #1796, #4737). When the error is
  the `Cannot install <a> and <b> … conflicting dependencies` one, the line under it decides
  the owner: `pip subprocess to install build dependencies did not run successfully` →
  `pyproject.toml [build-system] requires` resolved into an empty prefix, so the reporter's
  installed torch/torch-npu is irrelevant and `--no-build-isolation` is rung 1 (§D, #446);
  no build-deps subprocess → the `+cpu` local-label row (§C, #7089). With **no pip
  message pasted** — 「安装 vllm 0.9.1 提示卸载 torch 2.5.1 装 2.7.x」 on a box whose CANN and
  torch came preinstalled, then vllm-ascend "could not find torch 2.5.1" (#1363) — the same
  script plus the release's pairing row decide it, and the rung is vllm from source with
  `VLLM_TARGET_DEVICE=empty` (`Dockerfile:40-44`), not a pip command
  (`../image-dependency-faults/pip-index-visibility.md` §A0). When the same
  call is asked as a **dependency bump** ("能否把 torch-npu 升级为 2.8.0", no pip error),
  the same script's matrix block is the answer, not a version survey
  (`../image-dependency-faults/dependency-pin-bump-request.md`, #3266).
- `pip install` **from source** in a pod with no public network (the custom-op
  step "pulls dependencies from the public network", `fetch failed`, or a cmake
  `Downloading …` line): name the fetch legs instead of opening `csrc/<op>/` —
  the `build_aclnn` → `build_aclnn.sh` → `build.sh` → `csrc/CMakeLists.txt`
  chain fetches a `catlass` submodule plus json / abseil-cpp / ascend_protobuf /
  makeself from `gitcode.com`, each behind an existence gate under
  `CANN_3RD_LIB_PATH`, and neither `CANN_PKG_SERVER` nor an exported
  `CANN_3RD_LIB_PATH` changes that (seed the directory or pass
  `csrc/build.sh --cann_3rd_lib_path=`). Pin v0.13.0rc3; the chain was
  source-read, not run (`../image-dependency-faults/offline-source-build.md`).
- The same source build fails at **compile or link** (`multiple definition`,
  `collect2: error: ld returned 1`, `CalledProcessError` from CANN's
  `recompile_binary.py`): read the paste's last frames as the **stage** (python
  metadata / cmake configure / AscendC compile-pack / CANN relink / host stub
  link), check the reported CANN/torch/torch-npu trio against the pin's own
  `installation.md` table, and read a `multiple definition` off the `recompile:`
  line's object list — one source under two absolute prefixes is one tree compiled
  under two path spellings, so `rm -rf build` from `pwd -P` (or install without
  `COMPILE_CUSTOM_KERNELS`) is rung 1
  (`../image-dependency-faults/custom-op-build-stage.md`, #1774).
- Startup `ValueError` whose text names a parallel size inside
  `kv_connector_extra_config` (`KV transfer 'prefill' config has a conflicting
  data parallel size. Expected 1, but got 2.`): one grep for the string decides
  the owner (it is `vllm_ascend/utils.py`, not `arg_utils`), then `is_moe` for the
  model decides whether the process could ever keep DP>1
  (`../startup-failures/arg-validation-errors.md`, #8345). Do not walk
  `ParallelConfig` or the connector: in that version the extra-config numbers
  have no reader besides the check.
- A `--distributed-executor-backend ray` start that dies in the same second as
  `Connected to Ray cluster.` with `ValueError: Current node has no NPU available.
  current_node_resource={… 'GPU': 8.0}` (and, just above it, `The number of
  required NPUs exceeds the total number of available NPUs in the placement
  group`): one `ray status` on each node decides it — the Resources block must list
  `NPU`, and `GPU` means the raylets were started without
  `--resources '{"NPU": 8}'`, which is the whole fix. No sizing arithmetic, no
  driver read, and the world size already matches the card count
  (`../startup-failures/ray-npu-resource-label.md`, #3114).
  **Read the dict before acting on that**: if it shows `'NPU': 0` beside
  `NPU_group_<i>_<pgid>` labels on a `--data-parallel-size` > 1 serve, the key is
  right and a live placement group holds the node — the same `ray status` then shows
  `NPU: 0.0/16.0`, and the levers are `--data-parallel-backend ray` (its own log line,
  `Creating placement groups for data parallel`) or a stale group, not `--resources`
  (`../startup-failures/ray-dp-placement-group-npu-exhausted.md`, #10910).
- Custom op that aclnn refuses to *resolve* (`EZ1001`, `Get regInfo failed`,
  `does not support opType`, `binary_info_config.json of socVersion`): registry
  and install, before launch. Do not open `csrc/<op>/`: check the tree's
  per-SoC build branch and `def.cpp` `AddConfig` once to rule the source out,
  then `SOC_VERSION`, `_cann_ops_custom`, `ASCEND_CUSTOM_OPP_PATH` and the
  image tag (`../custom-ops-runtime-faults/op-not-registered-for-soc.md`).
  A **built-in** op instead (`EZ9999` / `Parse dynamic kernel config fail` /
  `<Op>AiCore ADD_TO_LAUNCHER_LIST_AICORE`): the *image's* CANN has no kernel
  for this card — check the release's supported-hardware lines, then one
  standalone op probe (`../soc-platform-paths/kernel-package-vs-card.md`).
  A **`aclnn<Op> or aclnn<Op>GetWorkspaceSize not in libopapi.so`** at a
  `torch.ops._C_ascend.<op>` call is neither: `EXEC_NPU_CMD` searches the
  vendored custom-op package and `ASCEND_CUSTOM_OPP_PATH` **before** the toolkit
  `libopapi.so`, so read `_cann_ops_custom/vendors/*/op_api/lib/libcust_opapi.so`
  and the import warning first (`../custom-ops-runtime-faults/opapi-symbol-absent.md`).
- Startup traceback carrying `torch/_subclasses/fake_tensor.py` + `torch/_ops.py`
  frames and ending `NotImplementedError: <ns>::<op>: attempted to run this
  operator with Meta tensors` (`_C::rotary_embedding` at #1895): two greps decide it
  — `rg -n 'kMeta|TORCH_LIBRARY_IMPL' csrc/` for an existing Meta impl, and the one
  dtype/shape predicate that selects the branch — then answer `--enforce-eager`
  before proposing a patch, with the dtype that skips the branch as rung 2
  (`../startup-failures/meta-kernel-missing.md`).
- Startup traceback whose deepest frame is **upstream** and ends
  `AssertionError: Torch not compiled with CUDA enabled`, e.g. a `get_rope`
  branch constructing a device in `__init__` (`dual_chunk_rope.py:37` at vllm
  v0.11.0): one call runs
  `bash <skill_root>/scripts/cuda-only-path-audit.sh <vllm-root> <subpath> <vllm-ascend-root>`,
  which lists the unguarded `torch.cuda` hits, whether the file has any
  `current_platform` guard, the `forward_*` implementations, and whether the
  downstream tree replaces the class at all. The `torch==2.7.1+cpu` line in the
  env dump is normal on NPU and is **not** the cause
  (`../startup-failures/checkpoint-config-cuda-only-path.md`, #4309).
- Degenerate output whose trigger the reporter names as *token content* ("only when
  these special tokens are in the prompt", identical across several releases, one
  model only): refute or confirm that claim from the paste before any source read —
  same ids on both input routes means the ids are not the mechanism, and the
  checkpoint's own `Using max model len N` is the cheap non-engine check
  (`../accuracy-benchmarks/short-prompt-corruption.md` step 0, #8590).
- Python attribute error whose frame is an **upstream** vllm file (or the
  reporter asks whether vllm-ascend's ops were ever reached): the Ascend
  replacement is name-keyed, so compare the class name against
  `REGISTERED_ASCEND_OPS` / `CustomOp.op_registry_oot` before theorising about
  upstream (`../upstream-downstream-drift/oot-class-registration-drift.md`).
  If the class *is* registered, or the missing name is a factory/property the
  patch rebinds, check for a pre-patch alias in the consumer module —
  `consumer.<name> is defining_mod.<name>` settles it in one line
  (`../upstream-downstream-drift/stale-module-binding.md`, #15815).
- Python `AttributeError` on a **per-object metadata name** (`attention_type`,
  `layer_type`) raised inside a downstream copy of a library `forward`: one call in the
  reporter's interpreter —
  `uv run --no-project python <skill_root>/scripts/qwen2-body-drift-probe.py --config <model dir>`
  — prints the installed body and says whether the copy mirrors it. Answer with the
  delegation (`super().forward()`), never a `getattr` default: recorded #11991's guard
  started the service and broke image recognition, so name the acceptance check as one
  real image request (`../upstream-downstream-drift/patched-forward-api-drift.md`).
- `RuntimeError: Invalid request status: <STATUS>` (e.g. `WAITING_FOR_FSM`)
  raised from `vllm_ascend/core/scheduler.py` while the server was serving: the
  `AscendScheduler` fork lags an upstream request status. One call —
  `grep -n '    def '` on the fork and on `vllm/v1/core/sched/scheduler.py` —
  shows the missing methods (`get_grammar_bitmask`), and one more
  (`grep -n "ascend_scheduler_config.enabled = True" …/platform.py`) says whether
  the fork was force-enabled for a default serve (rc0) or is opt-in — that decides
  between a gate-input flag and dropping the key, and whether the three-site patch
  is even needed
  (`../upstream-downstream-drift/scheduler-fork-missing-status.md`, #3979).
- PD / EPD transport complaint: **first ask whether a KV pool is configured**
  (`AscendStoreConnector` in a `MultiConnector`, `MOONCAKE_CONFIG_PATH`). A pool
  with **no** PD, or one that initialises and then fails while saving, is not
  this row — take the `../kv-pool-store/index.md` entry below. With a
  P/D pair, the pool decides before the transport: run
  `bash <skill_root>/scripts/kvpool-segment-check.sh <mooncake.json|log>` — the
  worker log's `register mem … mem type:host, length:N` line carries the
  per-rank `global_segment_size`, and ≥ 20 GB on HDK 25.2.3 is the recorded
  cause of `503900` + 乱码 (#13439), fixed at 2 GB
  (`../pd-disaggregation/kv-pool-store-segments.md`). Only without a
  pool, or once the segment is small, read the rank table with
  `bash <skill_root>/scripts/ranktable-check.sh <table-or-log>` — a declared
  `server_count > 1` (or more than one distinct `server_id`) decides the rest
  (multi-server view, `device_ip` + RoCE required). Then the plog grep set in
  `../pd-disaggregation/transport-topology.md`. If the connector is Mooncake and the
  lines are `ascend_direct_transport.cpp:820 Failed to connect to target … 103900`
  (`Mooncake transfer failed, ret: -1`), skip the port/`hostNetwork` tests and ask for
  the decode node's plog in the first message: the recorded #4866 answer carried them
  as the cheapest action and the plog the reporter then sent showed
  `'device_ip' is not set correctly` → `HcclCommInitClusterInfo failed, rankNum[0]`
  (`../pd-disaggregation/mooncake-adxl-connect-103900.md`).
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
  (`../kv-pool-store/memcache-a3-dsv4-init.md`).
- A pool **put** never lands while the request succeeds (`Failed to invoke
  HcclBatchPut, hccl_result = 4`, `Failed to put N keys out of N`, master
  `PutEnd` 0): one call —
  `bash <skill_root>/scripts/store-put-code-check.sh <paste>`. `4` is
  `HCCL_E_INTERNAL` and the store's `-1` is its generic code, so it is neither a
  parameter error nor capacity (`-200` is the out-of-space code), and the emitting
  string is in the `mooncake-transfer-engine-npu` wheel, not in any export
  (`../kv-pool-store/hccl-onesided-store-put.md`).
- Report is "the flag did nothing" (logging/verbosity, output printing, a
  missing log stage): name the owner first — upstream vLLM OpenAI frontend,
  not vllm-ascend — then walk the gate checklist in
  `../observability-and-logs/index.md`: both flags on the serving
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
  (`../observability-and-logs/env-var-validation-warning.md`). One more call decides
  the advice: names in the pin's *Migration Guide* table (`additional_config.md:6-24`
  at v0.22.1rc1) are being retired, so answer with their `--additional-config` key
  (`VLLM_ASCEND_ENABLE_NZ` → `weight_nz_mode`), never with `unset` alone.
- An endpoint that is missing rather than wrong (`POST /start_profile` →
  `404`, or a 500 `Profiling is not enabled`): a route-registration question
  in the serving process. Name `attach_router` (`path:line`) and its condition,
  then ask the live app: `curl -s <host>:<port>/openapi.json` filtered for
  `profile`; check the startup log for the gate's `warning_once` line, and only
  then wonder where the flag went (`../observability-and-logs/online-profiling-endpoints.md`).
  Model class, quantization and NPU health are irrelevant to a 404 — say so.
- Output that is fluent but stops after tens of tokens ("被截断") with a feature
  flag toggled nearby: `finish_reason` first (`length` = budget, `stop` = early
  EOS), then `rg -n "<key>"` every reader of that key and read the shipped
  support matrix before any mechanism
  (`../accuracy-benchmarks/early-eos-short-output.md`).
- A spec-decode symptom (acceptance collapse, a stall while drafts keep being
  produced, "works if I remove MTP"): order the ablations MTP off →
  `--async-scheduling` off → `--no-enable-prefix-caching` → `num_speculative_tokens 1`
  → concurrency/`--max-model-len`, and do not lead with the KV arithmetic:
  acceptance dropping to exactly 0.000 everywhere while drafts continue is a
  draft/target context question, and the recorded reporter confirmed the owner by
  removing MTP (`../spec-decode-mtp/acceptance-collapse.md`). One exception to that
  order: when the report's own A/B **is** a feature env (`VLLM_ASCEND_ENABLE_FLASHCOMM1`
  on vs off, TTFT improving as promised, acceptance halving, no error), the ablation is
  FC1 off and the reads are four lines long —
  `../feature-enable-recipes/sequence-parallel-flashcomm.md` §*FC1 with a MoE MTP
  drafter*, which also refutes the `pad_size` / double-shard story such a report usually
  proposes.
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
  (`../tool-call-streaming/streamed-arguments-audit.md`,
  `arg-type-gate-unquoted-value.md`).
- If that report's failing `num_speculative_tokens` is >= 3 while 1 or 2 works,
  name `vllm_ascend/sample/rejection_sampler.py:114 using_block_verify =
  max_spec_len >= 3` (the line is pin-dependent: `:116` at v0.18.0) in the same
  answer with the `n=2` and `temperature: 0`
  ablations; walk the kernel afterwards, never instead
  (`../tool-call-streaming/mtp-block-verify-acceptance.md`).
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
  (`../spec-decode-mtp/empty-first-response.md`).
- 310P / Atlas 300I report: take the SoC from
  `_build_info.__soc_version__` rather than the image tag, quote the
  documented status (experimental, stable = v0.10.0rc1 at v0.11.0rc3,
  `faqs.md:18`, `single_node_300i.md:3-7`) as the first sentence, and then
  apply the 300I environment (`--enforce-eager`, `--dtype float16`,
  `PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256`) and a smaller
  `--max-model-len` before blaming the platform
  (`../soc-platform-paths/atlas-300i-310p.md`). Write the allocator value
  explicitly — on 310P `expandable_segments:True` crashed every rank at
  `empty_cache` (`unmapHandles … 507899`) while `:False` deployed
  (`../runtime-memory/allocator-expandable-segments-unmap-crash.md`) — and
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
  (`../runtime-memory/kv-init-and-format-cast-alloc.md`).
  If instead the frame is `torch.npu.empty_cache()` and the text is
  `unmapHandles … 507899`, nothing failed to allocate: `grep -m1
  'Set PYTORCH_NPU_ALLOC_CONF' <serve log>`, then relaunch with an explicit
  `expandable_segments:False`
  (`../runtime-memory/allocator-expandable-segments-unmap-crash.md`).
- Runtime memory after a healthy period, or any "memory leak / 内存持续增长"
  title: `rg -n -i 'leak' <pin>/docs/source/user_guide/release_notes.md` (at
  v0.13.0rc1 `:197` is the tree's own "memory may be leaked … a bug from
  torch-npu") and read `torch_npu==` in the env dump **first** — `2.8.0` leaks and
  `2.8.0.post1`/`.post2` + a vllm-ascend rebuild is the fix that ends the report
  (reporter-confirmed #6299; the ref's own `requirements.txt:31` pins it, and
  `--no-enable-prefix-cache` does not help)
  (`../runtime-memory/host-ram-and-per-rank-rss.md` §Step 0). Then take the
  two-sample split (capacity vs growth) and
  `grep -m1 'Set PYTORCH_NPU_ALLOC_CONF' <serve log>`, and check
  `additional_config` key liveness before any leak theory: grep the whole repo,
  not `vllm_ascend/` alone — a key that survives **only** in `tests/` (`e2e` launch
  lines, a MagicMock'd UT attribute) is the project's own leftover, not a live
  knob, and a key with no hit anywhere is a typo
  (`../runtime-memory/hbm-growth-and-oom.md`, `../runtime-memory/index.md` rule 5).
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
  (`../performance-throughput/sequence-count-cliff-capture-size.md`,
  #14440).
- The body compares **two runs** and gives one number each ("12s vs 4s" between two
  releases, "吞吐高了一倍" against a GPU run, an EvalScope/AISBench table or a
  screenshot of one): run
  `uv run --no-project python <skill_root>/scripts/benchmark-table-normalize.py --run 'NAME OUT_TOK_S
  WALL_S REQS CONC' --run … --total-tok-s NAME=V --ttft NAME=S --compare A B` before
  any mechanism. It derives the per-request token counts, the decode step
  (`concurrency / output tok/s`), the prefill cost (`TTFT / input tokens`) and names
  the axis that moved — quote those instead of the table's own averages, which a
  screenshot OCR transposes (#2833: `4094/100` came through as `4994/199`). A
  same-command, two-release ratio also divides out the output token count and, on a
  spec-decode decode side, `Mean acceptance length`
  (`../performance-throughput/cross-run-ratio-normalization.md`, #2833, #14291).
- A DP×TP **MoE** serve reports low throughput while its own command omits
  `--enable-expert-parallel` (or has it on one node only, often because "开 EP 会胡言乱语"):
  read the selector once — `ep_size` is the EP group's world size when enabled and `1`
  otherwise, and `1` means every MoE layer all-gathers instead of MC2 — before any
  tuning; graph mode and spec decode may not exist for the family at that ref either
  (`../performance-throughput/moe-comm-path-when-ep-off.md`, #2833).
- The evidence is a **profiler table** rather than a metric block or a log — "`<Op>`
  被调用了 98 万次", an `msprof`/MindStudio operator-statistics screenshot, "NPU
  空闲时间 95%", an NPU-vs-GPU operator-count ratio: run
  `uv run --no-project python <skill_root>/scripts/opcount-reconcile.py --count NAME=N … [--steps N]
  [--elapsed-s S --avg-us NAME=US]` first (it prints the one step count every row's
  quotient agrees on, per-step factors, and the total-time reconciliation), ask
  whether the collection itself exited 0 (a `libruntime_v100.so` bug and a window
  that mostly covered idle time are the reporter-confirmed failure family), and let
  an unchanged same-request TPOT delete the amplification claim. Do not open
  `ops/linear_op.py` or `quantization/w8a8_dynamic.py` for a count
  (`../performance-throughput/profiler-opcount-evidence.md`, #7351).
- "The machine crashed" / the host dropped / a `top` paste whose python3 ranks
  each hold tens of GiB RES: run
  `bash <skill_root>/scripts/host-ram-check.sh <paste|->` and report the ratio
  before any NPU lane, quoting RES and never the `8326g` VIRT
  (`../runtime-memory/host-ram-and-per-rank-rss.md`).
- Same crash but **no** `top` to sample and the log stops mid-startup: last lines
  are the per-rank `rank N in world size M is assigned ...` set, no
  `Starting to load model` after them, `--max-model-len` 128K+. Run
  `bash <skill_root>/scripts/attn-mask-budget.sh --log <serve.log> --host-gib N`:
  the eager `max_model_len²` host mask is 9·N² bytes **peak per rank** before any
  weight load, so at 262144 nothing survives
  (`../runtime-memory/attention-mask-host-allocation.md`).
- A **container** counter climbing for days on a healthy service
  (`container_memory_working_set_bytes` "持续增加"), no crash and `npu-smi` flat:
  run `bash <skill_root>/scripts/working-set-arithmetic.sh --growth <reported>
  --days <window> --rps <rate>` for the bytes/request class, ask for the three
  cAdvisor series at two timestamps plus a per-PID RSS sample, and diff the
  reporter's serve flags against the model tutorial's own recipe before any
  source read (`../runtime-memory/container-working-set-growth.md`).
- The benchmark/eval client's own traceback is the report ("推理正常、性能测试崩溃",
  `ConnectionRefused [Errno 111]`): treat the client as secondary and locate the
  first in-window server fault — the API process does not listen because it died.
  If the server log shows `!!!!!!! Segfault encountered !!!!!!!` plus
  `died unexpectedly (exit code: None)`, take
  `../hccl-and-hardware/worker-segfault-no-python-frames.md`: count the
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
  (`../hccl-and-hardware/error-report-timestamps.md`).
- First collective after the weights load dies on `Transport_Init_Error(EI0009)` /
  `[Create][DestLink]Create Dest error!`, with `halShmemOpenHandle … drvRetCode=17` and
  `rtsIpcMemImportByKey … 507899`: run
  `bash <skill_root>/scripts/hccl-link-pair-audit.sh <log>` — it lists the failed
  `rank→rank` edges, the driver codes, and classifies them same-card vs cross-card
  against the `npu-smi` Bus-Ids. All-cross-card with the card-local edges up is a host
  transport/driver pairing question (driver vs the image's CANN, container
  passthrough), and the identical `localIpAddr`/`remote_ip_addr` text is **not** a
  duplicate `device_ip` — the rank table is `scripts/ranktable-check.sh`'s job
  (`../hccl-and-hardware/transport-destlink-ipc-import.md`, #4148).
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
  (`../dsa-sparse-attention/indexer-cache-scatter-fault.md`).
- Engine-start `AttributeError: 'AscendRowParallelLinear' object has no attribute
  'aclnn_input_scale'` (or `_reciprocal` / `_offset`) at `sfa_v1.py:707` ← `:544`,
  with FLASHCOMM1 / legacy FLASHCOMM / shared-expert-DP on and a DSA model: the
  DSA-CP o_proj TP/Full switch exists only for a W8A8-static o_proj, so the answer
  is the quant method the checkpoint forces (FLASHCOMM1 off, or the tutorial's W8A8
  weights), not an attention or install fault
  (`../dsa-sparse-attention/o-proj-quant-attribute-crash.md`).
- A MoE deployment dying at the first real batch with `call aclnn<Op> failed,
  error code is <6 digits>` (`aclnnMoeInitRoutingV3 … 561103`,
  `Kernel Run failed … MoeInitRoutingV2`): the number is a **failed tiling check on
  one input**, identical across unrelated ops, so ask for the EZ9999 body
  (`CheckTokenCount`, `moe_init_routing_v2_tiling`) before any kernel reading, and
  settle which parameters reach the op with the selector
  (`ascend_forward_context.py:246-255`) plus
  `ops/fused_moe/token_dispatcher.py:324-350`. A "DP2 TP4 有没有限定 / 能否启动时校验
  拦截" question is answered by that arithmetic and the documented A2 block
  (`docs/source/tutorials/DeepSeek-V4.md:149-167` = DP1×TP8 + EP), never by a
  DP×TP matrix (`../custom-ops-runtime-faults/moe-init-routing-tiling.md`).
- Degenerate output (`</think>` in `content`, repetition collapse, 乱码) at long
  context **and** concurrency on a model the tree documents, with both `vllm serve`
  commands pasted: open `docs/source/tutorials/models/<model>.md` at the pinned
  commit and diff the reporter's P and D commands against the validated ones plus
  the `Recommended Configurations` tables **before** reading any connector — that
  diff, with the ablation matrix, is the first answer
  (`../accuracy-benchmarks/garbled-long-context-output.md`). Recorded
  #13002 spent 21 turns and its whole output budget inside a 2600-line
  `mooncake_connector.py` and shipped nothing, though the tutorial's validated
  commands deviated on five flags and were reachable in two calls — the replay that
  had `garbled-long-context-output.md` answered the same paste in 11 turns, which is
  what those two calls buy.
  If instead the *prompt* is short, the *completion* runs long and a few
  responses out of many repeat themselves, that is not the prompt-length lane:
  diff the same tutorial for the decode side and lead with the identity row
  (`--max-num-seqs 1`, `temperature: 0`, same request 3×)
  (`../accuracy-benchmarks/long-output-repetition.md`).
- The same prompt or body run **several times** gives different lengths or contents
  ("nv 卡多次一致，npu 多次差异大", "`temperature=0` 结果不确定"): this is a
  reproducibility question, so the first two reads are the request body's
  `temperature`/`top_k`/`seed` and the *pattern* of which runs differ — not a kernel
  walk and not a long-output hunt. A stable NV baseline is a control that refutes
  sampling randomness, the tree's FAQ #16 (`docs/source/faqs.md:167-203`, four env
  vars) is the documented recipe, and `VLLM_BATCH_INVARIANT=1` is the beta mitigation
  with 910B/custom-op/coverage gates
  (`../accuracy-benchmarks/repeated-run-nondeterminism.md`). Recorded #7180 ranked
  "sampling not pinned" first; the reporter's next comment pasted a greedy body.
  When the body is **stable** and only the **concurrency** changes the answer, it is the
  batch-composition sibling: probe `python -c "import batch_invariant_ops"` before
  anything else, because the batch-invariance INFO log line prints with attention
  still unpatched (`../accuracy-benchmarks/batch-invariant-coverage.md`).
- A crash under concurrency with **no** error text pasted: do not hunt the mechanism.
  The answer is the gap, the config diff against the tree's own validated command
  (`tests/e2e/**`, `docs/source/tutorials/<model>.md`) and ranked candidates with one
  check each, emitted at the normal gate
  (`../dsa-sparse-attention/concurrency-crash-with-no-log.md`).
- A "communication timeout" whose only text is a bare `TimeoutError` with
  `multiproc_executor.py` / `shm_broadcast.py` frames: run
  `bash <skill_root>/scripts/executor-timeout-check.sh <paste|->` before any
  HCCL reading. It is the framework's own RPC watchdog (300 s) for a worker that
  stopped answering, so the deliverable is the lane classification plus the list
  of per-rank artefacts — which rank, which op and OOM-vs-hang are *not* in that
  paste (`../hccl-and-hardware/rpc-watchdog-timeouts.md`, #1865).
- A **PD deployment with `--decode-context-parallel-size` > 1** whose engines
  started and whose **first request** killed one side's EngineCore, the APIServer
  showing only `EngineDeadError: EngineCore encountered an issue. See stack trace
  (above)`: before any source walk, put the two serve commands side by side and
  compare `--tensor-parallel-size`, `--decode-context-parallel-size`,
  `--prefill-context-parallel-size`, `--cp-kv-cache-interleave-size` and
  `--block-size`. DCP is a *matched* launch parameter — the prefill side
  advertises its `dcp_size` and the peer asserts divisibility on the first
  transfer — so a dcp that differs between the sides (or a 1M recipe's `8`
  reduced on one side) is the answer, and the MemCache pool is not the suspect
  (`../feature-enable-recipes/context-parallel-pcp-dcp.md` §3,
  reporter-confirmed #12677).
- A **multi-node start** (two nodes, DP/EP across them) whose paste carries each
  node's own command: put the two blocks side by side before ranking anything. Some
  flags are *matched* inputs to a collective, not local tuning —
  `cudagraph_capture_sizes` feeds `mc2_tokens_capacity = ceil(max_capture/tp)*tp` and
  the MoE dispatch's `global_bs = ceil(max_capture/tp) * ep_world_size`, both derived
  **per rank from that node's own value** — so a list that differs between the nodes
  makes the two halves of the EP group declare different buffer sizes to one all-to-all
  (recorded #8294: `[1..64]` vs `[1..512]` → capacity 64 vs 512, `global_bs` 128 vs
  1024, `ep_world_size` 16). One call:
  `scripts/mc2-capacity-asymmetry.sh --capture-a 1,2,4,8,16,32,64 --capture-b
  1,2,4,8,16,32,64,128,256,512 --tp 8 --dp 2`; the mechanism and its open status are
  `../hccl-and-hardware/mc2-dispatch-profile-run-hang.md` Step 1b.
- A **long-run hang** in a multi-node or co-located deployment: `hccl fftsplus
  task timeout ... stuck notify num:1, timeout=<N>s` plus
  `[TaskExecStage][HeartbeatAbnormal] ... Heartbeat Lost Occurred`. Subtract
  before ordering: stall start = report time − `<N>` (read `<N>` off the line: at
  v0.23.0 the armed default is `1836 s = 27*68`, not the `1800 s` of the stale
  `error_code.cc` text, and a reporter-set `HCCL_EXEC_TIMEOUT` appears verbatim),
  and check the last `200 OK` of that instance
  against it; the heartbeat lines arrive ~5 min later and the `[shutdown] MPClient`
  / `ApiServer ... died` lines are teardown, not the cause
  (`../hccl-and-hardware/error-report-timestamps.md`, #16652).
- `TypeError: 'NoneType' object is not callable` where the `None` is an upstream
  module-level name (`fused_experts = None` on a non-CUDA platform): read the
  **file** of the `forward_oot` frame. `vllm/model_executor/custom_op.py` there
  means the Ascend replacement was never imported in that process (Ray/`spawn`
  workers; the hook runs only where `VllmConfig` is built) — two greps decide it,
  do not theorise about the model or the checkpoint
  (`../upstream-downstream-drift/patch-import-side-effect.md`, #253).
- Quantization method rejected before any worker (`<method> quantization is
  currently not supported in npu`, `--quantization awq`): answer from
  `../startup-failures/quantization-method-gate.md` — three calls, and
  the flag is not the only trigger, since the checkpoint's own `quant_method`
  reaches the same gate. The same page's section B is the inverse report: an
  msmodelslim W8A8 checkpoint whose `config.json` deliberately has **no**
  `quantization_config` started without `--quantization ascend` /
  `quantization="ascend"` and died in a worker (#2015) — the flag is mandatory
  exactly because upstream's auto-detection has nothing to read.
- A serve that dies on a *quantised-lineage* model whose name carries no quant suffix
  (`DeepSeek-V3`, `DeepSeek-R1`, `-0324`) with `quantization=ascend` in the engine line
  and no `--quantization` in the command, or with `fp8 quantization is currently not
  supported in npu` beside such a model (the plugin's `ascend` never registered, so the
  checkpoint's own `fp8` reached the platform gate): one call decides whether the
  *weights* are the limit before any quant-config walk —
  `uv run --no-project python <skill_root>/scripts/checkpoint-key-prefix.py --dtypes <model dir>`.
  `F8_E4M3` weight tensors = the upstream FP8 release and no flag/config edit runs it
  (dequantize to bf16, or a msmodelslim W8A8 export); int8 + `quant_model_description.json`
  = a converted export whose method was not selected; bf16 with a leftover
  `quantization_config` = the declaration is the bug, one config edit
  (`../startup-failures/quantization-method-gate.md` §E — and its §E-i when the release
  has the name but no implementation, e.g. v0.7.1rc1).
- A load-phase `KeyError: '<prefix>.weight_scale'` or `'…experts.w2_weight_offset'`
  (a quant **aux** tensor, not a weight): one call censuses the checkpoint before any
  `modelslim_config.py` read —
  `uv run --no-project python <skill_root>/scripts/checkpoint-key-prefix.py --scales <model dir>`
  — and the serve command decides the rest
  (`../startup-failures/weight-scale-keyerror-quant-variant.md`). Scales on the
  main linear/MoE families = a quantised export missing its method (the bullet
  above); scales confined to `self_attn.wkv` / `self_attn.indexer.wq_b` = a
  full-precision release served instead of the documented quantised variant; an
  expert offset name with no expert index (`…mlp.experts.w2_weight_offset`) is
  upstream's remap of a per-expert key, and at v0.8.5rc1 only `W8A8_DYNAMIC`
  allocates MoE offsets — so the description in effect marked the experts `FLOAT`
  (ask for that entry; recorded #986). The same family's upstream
  `ValueError: There is no module or parameter named '<module>.<aux>'` is the same
  census plus one constructor read: the export quantised a module the model file
  built without `quant_config` (ViT tower, projector), and its `(`class`)` is the
  runtime CustomOp name (recorded #7200).
- A `RuntimeError: Promotion for Float8 Types is not supported … Float8_e4m3fn and
  Bool` in `ops/fused_moe/token_dispatcher.py` on the first MoE forward: three
  `sed` reads and one ablation, never a device theory
  (`../startup-failures/fp8-topk-weights-mask-promotion.md`).
- Deployment sizing / stage layout (does it fit on N cards, why a stage
  layout fails): run `bash <skill_root>/scripts/card-budget.sh` with the
  checkpoint's parameter count, the TP and the per-stage
  `gpu_memory_utilization`/device lists, and report the two numbers it prints
  (weights per card vs card budget; per-device gmu sum). Check the device
  variable first: `ASCEND_RT_VISIBLE_DEVICES`, not `CUDA_VISIBLE_DEVICES`
  (`vllm_ascend/platform.py:101` at v0.19.1rc1;
  `../deploy-sizing/card-budget-and-device-visibility.md`). A sibling
  project's yaml/CLI absent from `SOURCES.json` (vllm-omni) is a stated gap,
  not something to infer.
- A first-worker device open returning `ERR99999` + `set default device id failed,
  ret: 107001` / `invalid device ID` (frames `worker.py` → `check_ascend_device_type()`
  → `get_soc_version()`): two host calls settle it — `ls /dev/davinci*` + `npu-smi info`
  inside the container, and one `python -c "import torch, torch_npu;
  print(torch_npu.npu.device_count(), torch_npu.npu.get_soc_version())"`. Before ranking
  any serve flag, grep the flag set in the tree: a line copied verbatim from
  `tests/e2e/**` is exonerated in one call
  (`../startup-failures/device-id-invalid-at-worker-init.md`).
- A parallel-topology question about EP **with** DP, or an engine-init crash whose
  deepest frame is the MoE `prepare/finalize` `dist.all_gather` on a 910B DP×TP
  deployment: read `_select_moe_comm_method`
  (`worker/model_runner_v1.py:1373-1432`) and `set_mc2_tokens_capacity`
  (`ascend_forward_context.py:169-183`) first, then answer with that SoC's row,
  the reporter's `world_size_across_dp` arithmetic and the two ablations
  (`../feature-enable-recipes/moe-expert-parallel-with-dp.md`); say that
  MC2 shares the All2All `finalize`, so the pasted line is not proof MC2 was
  skipped. If instead the paste dies in `profile_run` with `507014` and the
  `[AIC_INFO]` kernel is `MoeDistributeDispatchV2`, the batch is
  `mc2_tokens_capacity` (an extra `_dummy_run` at
  `model_runner_v1.py:2202-2208`), not `--max-num-batched-tokens` — take
  `../hccl-and-hardware/mc2-dispatch-profile-run-hang.md`.
- Feature-enable / capability question ("how do I correctly enable F on H", no
  log): answer in this order — the validation verdict at the pinned ref (docs
  sentence, the hardware the docs measured, `rg -l
  "prefill_context_parallel_size|prefill-context-parallel-size" tests/` for e2e
  coverage, precedent issues with their versions), then each enforced constraint
  with the error text it raises, then one-axis-at-a-time staging. Grep the flag in
  `vllm_ascend/` for hard errors (`AssertionError`, `NotImplementedError`) versus
  silent overrides; do not read the kernel/code path beyond that
  (`../feature-enable-recipes/index.md`).
- Code-change / cleanup request (remove a `VLLM_ASCEND_*` switch before a
  release, deprecate a flag, finish a "TODO before 0.x"): one `rg -n '<IDENT>'`
  and one grep for the *mechanism* token at the pinned ref decide answerability —
  absent means the snapshot predates the PR, which is the finding. Then the
  touch-point sweep (definition dict + generated doc page + readers + tests/CI
  env sets + release note), the default-flip note and the reachability check
  (`../code-change-requests/env-flag-retirement.md`, #1262). Do not open
  the op's code path.
- Community-test / doc-verification task: run the twelve checks of
  `../tutorial-verification/tutorial-command-audit.md` in order
  (flag existence at the pinned vllm, bench argument names, image WORKDIR vs
  the doc's relative paths including any `bash <script>.sh` helper, relied-on
  defaults, card-count arithmetic vs sibling docs and the CI config, baseline
  release, one precedent pass, package version the doc installs vs the image's
  own pins, support matrix and release notes, `--hf_overrides` architecture
  string vs the checkpoint family, tab-set block vs its own prose and its own
  `kv_connector`, printed sample body vs the pinned response models) — plus,
  when the 任务书's own two items are "bench 报 404" and "加载要 3 小时", the
  served-name contract and the weight pre-download of
  `../tutorial-verification/serve-contract-and-weight-predownload.md`, which are
  one page grep each. Then write the report file as sections become
  true (`../tutorial-verification/community-test-report.md`). Stop
  auditing after 3-5 tiered defects; never present a doc baseline as a
  measurement.
- Accuracy/corruption: establish whether both numbers came from the same
  server, weights and serve flags before choosing a culprit
  (`../accuracy-benchmarks/index.md`); for a garbled prompt, get the
  ablation matrix (`../accuracy-benchmarks/garbled-long-context-output.md`).
  When the two numbers are **two quantized checkpoints** of one family, the
  same-model row in the reporter's table is the control and the two formats'
  pinned C8 recipes differ with the width too
  (`../accuracy-benchmarks/quantized-sibling-score-gap.md`); when the request
  carries **audio**, `usage.prompt_tokens` places the modality and the
  offline-vs-online row owns it
  (`../accuracy-benchmarks/omni-audio-chat-degeneration.md`).
- An **embedding / pooling** checkpoint that never starts: read the launch line for
  `--task embed` (`--runner pooling`) and the init line for `pooler_config=None`
  before any source; `ValueError: There is no module or parameter named
  'embed_tokens' in …ForCausalLM` is the missing flag, not the model. One call:
  `../../scripts/checkpoint-key-prefix.py <model dir>`
  (`../pooling-embedding/embedding-checkpoint-as-generate.md`).
- An **embedding / pooling** serve (`--runner pooling`, `--task embed|score`,
  `/v1/embeddings`): two questions place it before any source read — single
  request or only under concurrency, and is a short prompt correct. Crashes under
  concurrency end in the batch-slot machinery
  (`../pooling-embedding/batch-index-desync.md`); wrong vectors for long
  inputs take the "which span was pooled" test plus a
  `--max-num-batched-tokens` sweep, not an attention-backend walk
  (`../pooling-embedding/chunked-prefill-vector-corruption.md`).
- The same serve **never reaches ready**, and the paste carries either a model
  class/HF-config `TypeError` in the APIServer or an `EZ1001` weight-layout line
  at `profile_run`: two greps decide both before any source walk — the
  `architectures` string against the checkpoint's `model_type` family
  (`../pooling-embedding/seq-cls-arch-routing.md`) and the NZ gate at the
  reporter's ref, `rg -n "def is_enable_nz|_should_trans_nz|maybe_trans_nz"
  vllm_ascend/utils.py`, whose fp32 policy decides whether the fp32 score head is
  converted (`../pooling-embedding/fp32-head-nz-format-crash.md`). The same grep
  also answers the decoder-shaped sibling — an `EZ1001` whose text is `Not support
  mat2 n = 1 or k = 1 when format is FRACTAL_NZ` is the **shape** family, and the
  lever is `weight_nz_mode` (`0`/`1` ok, `2` wrong) for a MoE shared-expert gate
  (`../startup-failures/nz-weight-format-refusal.md`). One
  call each: `scripts/nz-format-gate-check.sh <paste> <vllm-ascend-root>` and
  `scripts/seq-cls-arch-check.sh <vllm-root> '<Arch>'`. The `EZ1001` code alone
  does not mean the custom-op package.
- A **VL image/video** request that degenerates into one repeated token
  (`!!!!`, 感叹号) while text-only prompts answer: first row of the ablation is
  `USE_OPTIMIZED_MODEL=0` on the same request, then text-only, TP=1 and an
  official checkpoint (`../accuracy-benchmarks/vl-vision-tower-degeneracy.md`).
- `patch/**` frame: extract-and-diff first — `sed -n 'A,Bp'` the pinned upstream
  method, dedent with `sed 's/^    //'`, diff the module-level copy, read only the
  non-formatting hunks, then check whether the one added hunk is live for the
  reporter's command (PD `is_kv_consumer`, `use_async_scheduling`). Formatting-only
  means the owner is upstream at that pin: say so with both `path:line`. #8724
  spent 29 proxy turns here and shipped nothing
  (`../upstream-downstream-drift/patch-copy-vs-upstream.md`).
- Worker `KeyError` in `_update_states` + `NoneType … sampled_token_ids` + a
  negative `Counters …` value on an `--async-scheduling` + MTP launch: one
  provenance (discarded row → count 0 → `num_accepted = -1` → missing
  `prev_req_id_to_index` key). Say that, then async off
  (`../spec-decode-mtp/async-spec-bookkeeping-desync.md`).
- Device dump whose deepest `vllm_ascend` frame is the drafter's valid-count
  copy (`counts_cpu[: counts.shape[0]].copy_(counts, non_blocking=True)`,
  `model_runner_v1.py:1393`) with `rtMemcpyAsyncEx … model stream execute
  failed` / `aivec … trap`: the copy is the sync point, and landing on it
  **proves** `--async-scheduling` + spec decode + padded drafter batch, so do not
  ask. Resolve the collective by `rankSize` against the DP×PCP×TP group
  arithmetic (a 2-rank group is not `fc3_quant_x` on DP2×TP4), then attribute
  with `--enforce-eager` and `ASCEND_LAUNCH_BLOCKING=1`
  (`../spec-decode-mtp/drafter-count-copy-device-fault.md`).
- Device fault inside an attention backend on a batch holding decodes **and**
  prefills: run `bash <skill_root>/scripts/batch-split-audit.sh <vllm-ascend-root>`
  before the kernel; its hits are the request tensors sliced by a token count and
  the unrebased `cu_seqlens`, and spec decode is the multiplier that turns a
  silent overshoot into `507015` (`../mixed-batch-splits/index.md`). The same
  script covers the **host-side** twin — `could not broadcast input array` /
  `np.repeat` length error in `worker/pcp_utils.py` with MTP + PCP > 1 — where
  the printed lengths alone name the batch
  (`../mixed-batch-splits/pcp-schedule-index-units.md`).
- A **LoRA** serve, either shape: run
  `bash <skill_root>/scripts/lora-graph-descriptor-audit.sh <vllm-ascend-root>
  [<vllm-root>]` first. It names the punica backend the ref binds (that decides
  the slow shape) and labels each `has_lora` producer live-batch vs
  config-constant (that decides the request-time capture crash); the SoC and
  graph-mode facts come from the paste, not from more source
  (`../lora-adapters/index.md`).
- An operator that **runs** and returns wrong numbers with a framework-free
  `torch_npu` repro (no dump): find the one caller holding that kwargs
  combination before reading the kernel; `cann/ops-nn` is usually absent, so
  the ownership split is the answer
  (`../custom-ops-runtime-faults/attention-op-numeric-errors.md`).
