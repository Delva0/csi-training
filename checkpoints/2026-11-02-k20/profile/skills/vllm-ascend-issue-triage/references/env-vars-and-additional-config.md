# vLLM Ascend env vars, the "Unknown environment variable" log, and `additional_config`

Scope: reports that quote a `VLLM_ASCEND_*` variable, an "Unknown vLLM environment
variable detected" log line, or an `--additional-config` key. Verified by reading tagged
source and docs in the local clones on 2026-11-01; nothing was executed on an NPU.

## The warning is upstream-side and cosmetic for vllm-ascend-only names

- Upstream `vllm/envs.py` keeps a closed dict `environment_variables` (v0.22.1: line 577)
  and `validate_environ(hard_fail)` (line 2051) walks `os.environ`, warning — or raising
  `ValueError` when `hard_fail` is true — for every name starting `VLLM_` that is not a key
  of that dict. It is called once from `ArgUtils.create_engine_config`
  (`vllm/engine/arg_utils.py:1711` at v0.22.1); the switch is `fail_on_environ_validation`
  (same file, line 714, default `False`).
- vllm-ascend declares its own variables in `vllm_ascend/envs.py` under a different dict
  name, `env_variables` (v0.22.1rc1: line 30; `VLLM_ASCEND_ENABLE_NZ` line 92,
  `VLLM_ASCEND_ENABLE_FLASHCOMM1` line 75), and never inserts them into upstream's dict.
  `git grep -n 'environment_variables' <ref> -- vllm_ascend` finds only `collect_env.py`
  reading it at v0.22.1rc1, v0.23.0, `origin/releases/v0.30.0` and `main`.
- So every set `VLLM_ASCEND_*` name produces the warning while still being honored, and
  `--fail-on-environ-validation` turns it into a startup failure. Names registered upstream
  (`VLLM_USE_MODELSCOPE` at v0.22.1 `envs.py:677`, `VLLM_TARGET_DEVICE`) do not warn;
  `VLLM_VERSION` is declared only by vllm-ascend, so it does.
- Maintainer-confirmed in issue #11624: the line is "printed by the upstream vLLM. This log
  can be ignored because it does not detect the environment variables exclusive to
  vllm-ascend". Do not report it as a version mismatch or as ignored configuration.

## Which value applies: `additional_config` > env var > default

`AscendConfig._get_config_value(additional_config, key, env_key, env_value)` (v0.22.1rc1
`vllm_ascend/ascend_config.py:289`) returns `additional_config[key]` when present (logs
"set from additional_config"); otherwise, if `env_key in os.environ`, it logs "falls back
to environment variable ... will be removed in the next release" and returns the
env-derived default. The env var only supplies the default; `--additional-config` wins.

Read the tag's `docs/source/user_guide/configuration/additional_config.md` before grepping
the config: its migration table lists each env var beside its key, expected type and
`"1"`→`true` conversion (at v0.22.1rc1: `enable_balance_scheduling`, `enable_flashcomm1`,
`enable_matmul_allreduce`, `flashcomm2_parallel_size`, `msmonitor_use_daemon`,
`enable_mlapo`, `weight_nz_mode`, `enable_fused_mc2`,
`enable_transpose_kv_cache_by_block`). `docs/source/user_guide/configuration/env_vars.md`
is a `literalinclude` of the `begin-env-vars-definition` block of `vllm_ascend/envs.py`, so
that list shrinks as variables are dropped.

A misspelled *flat* key in `additional_config` is silently ignored at v0.22.1rc1–v0.26.0rc1
(flat keys are read with `additional_config.get(...)`); unknown-key rejection arrives with
the later pydantic-dataclass configs on `main` (`extra="forbid"`). A "flag has no effect"
report can therefore be a typo, and the same typo is not silent on newer releases.

## `PYTORCH_NPU_ALLOC_CONF` is set by the plugin itself, so "I unset it and it started" is a no-op delta

- Both tags checked apply the allocator default themselves: `v0.26.0rc1`
  `vllm_ascend/platform.py:767-785` (inline) and `v0.27.1rc1`
  `vllm_ascend/platform.py:1339-1358 _set_pytorch_npu_alloc_env` (called at `:496`) read
  `os.getenv("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")` and append
  `,expandable_segments:True` unless the value already contains `expandable_segments`,
  `max_split_size_mb` or `garbage_collection_threshold`, then log
  `Set PYTORCH_NPU_ALLOC_CONF=...`. A launcher that stops exporting it therefore gets the same
  effective config; only `enable_sleep_mode` skips the block, and the RL config path adds the
  mirror `_disable_expandable_segments()` (`v0.27.1rc1:1361-1375`, called from
  `ascend_config.py:215-217` when `rl_config.enabled`) so an RL/rollout process runs without
  expandable segments either way.
- So an "unsetting `PYTORCH_NPU_ALLOC_CONF` fixed it" claim on a serving run needs another
  explanation; quote the `Set PYTORCH_NPU_ALLOC_CONF=` log line as the evidence of what took
  effect instead of reasoning from the launcher, and A/B one variable at a time when the report
  removed two exports together.
- **A launcher export can suppress the default, and that is the mirror of "unsetting it is a
  no-op".** The append is skipped when the existing value already contains `expandable_segments`,
  `max_split_size_mb` or `garbage_collection_threshold` — the same three names at `v0.23.0rc1`
  `platform.py:761-779` (inline, `if model_config and not model_config.enable_sleep_mode` at `:765`,
  the three-name guard at `:772-777`), `v0.26.0rc1:767-785` and `v0.27.1rc1:1339-1358`. A launcher passing
  `-e PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256` therefore runs without expandable segments and
  logs `Set PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256`, the opposite of the tag's own OOM advice
  (`docs/source/faqs.md:149` at `v0.23.0rc1`: "use `export
  PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` to enable virtual memory ... to mitigate memory
  fragmentation"). For a "memory keeps growing / is never released" report whose launcher sets
  `max_split_size_mb`, propose dropping that single export as the A/B before any allocator-internals
  theory; it is code-read here, not measured, and it does not establish a leak by itself.
- **Recorded instance (#16022, `v0.23.0rc1` image, A2 TP4 dense Qwen3.6-27B, open, 2 comments):**
  "VRAM is not released after running a while, then it crashes", screenshot-only, no exception to
  read. Its serve line sets `max_split_size_mb:256` and carries `torchair_graph_config` +
  `ascend_scheduler_config`, i.e. the ACLGraph run with the suppression above; the thread's own
  repro on `v0.26.0rc2` held per-device memory at ~46 GB across two 32-concurrent rounds and KV usage
  returned to 0%. Answer with the export A/B plus the upgrade, labelled unreproduced — do not invent
  a leak site for a report that has no log.

## `HCCL_OP_EXPANSION_MODE`: what the plugin reads per tag, and where the tag's docs disagree

- At `v0.27.1rc1` the plugin reads it in exactly two places after the platform refactor: a CANN
  comm-mode switch in `vllm_ascend/device/device_op.py:946` (`CCU_SCHED` ->
  `comm_mode="ccu"` for `npu_mm_reduce_scatter_base`) and the MiniMax-M2 patch
  (`vllm_ascend/patch/platform/patch_minimax_m2_config.py:110-127`), which sets `AIV` when the
  model is `minimax_m2`, graph capture is on and the user left it unset, and warns that any other
  value "may reduce ACL graph shape coverage". The ACL-graph capture-size formula that also read
  it at `v0.18.0rc1` ([full-decode-only-graph-mode.md](full-decode-only-graph-mode.md)) is gone
  from `utils.py` at `v0.27.1rc1` (`update_cudagraph_capture_sizes` no longer consults it).
- The tag's own docs treat AIV as a per-topology choice, not a default: the GLM-5.3 A2
  multi-node recipe exports `HCCL_OP_EXPANSION_MODE="AIV"`, while `tutorials/models/Kimi-K3.md`
  says "Do not set `HCCL_OP_EXPANSION_MODE=AIV` for this baseline" for one large DP baseline and
  "Prefill uses AICPU by default, so leave `HCCL_OP_EXPANSION_MODE` unset in the Prefill command.
  Decode uses AIV". A report whose workaround is "remove the AIV export" is therefore inside
  documented variation; ask for the one-variable A/B before calling it a plugin defect.

## `VLLM_VERSION` overrides the detected upstream version, so a tutorial that exports it subverts every version gate

- `vllm_version_is(target)` prefers the environment over the installed tree: at `v0.23.0rc1`
  `vllm_ascend/utils.py:610-618` reads `envs_ascend.VLLM_VERSION` when it is set and only then
  falls back to `import vllm; vllm.__version__` (the same function is `functools.cache`d above, so
  one value decides every gate in the process). `docs/source/user_guide/configuration/env_vars.md`
  is a `literalinclude` of `vllm_ascend/envs.py`, so the name is not upstream's.
- The value has to be compared with the tag's own `Dockerfile` `ARG VLLM_TAG`, and a shipped
  tutorial can disagree with it. At `v0.23.0rc1` `docs/source/tutorials/models/GLM5.2.md` exports
  `VLLM_VERSION=0.21.0` in every deployment block (`:321`, `:378`, `:440`, `:499`, `:1009`,
  `:1092`) while `Dockerfile:42` is `ARG VLLM_TAG=v0.23.0`. The same literal is in that file at
  `v0.22.1rc1` (`ARG VLLM_TAG=v0.22.1`) and `v0.24.0rc1` (`ARG VLLM_TAG=v0.24.0`), and it is gone
  from `v0.23.0`, `v0.23.0.post1` and `v0.21.0rc1` — so check the tag, not the model.
- Consequence, counted at each tag: `v0.23.0rc1` has 48 `vllm_version_is("0.23.0")` call sites in
  23 files (including `worker/model_runner_v1.py`, `ops/fused_moe/fused_moe.py`,
  `patch/platform/__init__.py`, `patch/worker/__init__.py`); `v0.22.1rc1` has 20 `("0.22.1")` plus
  2 `("0.21.0")`; `v0.24.0rc1` has 24 `("0.24.0")`. An exported older value therefore flips the
  whole set at once, in both directions: `patch/platform/__init__.py:34` (0.23.0-only GLM/MiniMax
  tool-call and usage-accounting patches) is skipped, `:53-55` (`patch_fused_moe`,
  `patch_dp_device_ids`) is imported instead, and `patch/worker/__init__.py:28-31` sets
  `_V2_MODEL_RUNNER_SUPPORTED = True` with the file's own comment saying "the v2 worker patches
  target main-only APIs" — which then imports `patch_v2.patch_uva` / `patch_input_batch` /
  `patch_model_state` (`:80-83`) against the release tree.
- Rule: when a report's launcher exports `VLLM_VERSION`, compare it with the reported tag's
  `ARG VLLM_TAG` before reading any version-gated code, and have the reporter unset it — the tag's
  FAQ (`docs/source/faqs.md:137`) describes the variable as a way to state an installed vLLM
  version when a dev/editable build hides it, which a release image does not need. Only after
  unsetting it does a start-up `ImportError`/`AttributeError` at plugin import belong to the
  plugin/upstream seam.
- Evidence boundary: source- and doc-verified at the tags named above; no run was made. The
  consequence is stated as "these gate flips happen", not as "this report's crash was caused by
  it" — for the recorded GLM-5.2 report the line was commented out, so there it is only the first
  thing to rule out.

## `ascend_scheduler_config` and `torchair_graph_config`: 0.10-0.11-only keys, silently inert afterwards

`git grep -c 'ascend_scheduler_config' <ref> -- 'vllm_ascend/*'` hits `ascend_config.py`,
`core/schedule_config.py` and `platform.py` at `v0.10.0rc1` and `v0.11.0`, and returns nothing at
`v0.12.0rc1` or `v0.13.0rc1`: `f10acddb7` / PR #4498 "drop ascend scheduler" removed the
implementation, first tag `v0.12.0rc1`. Quote that commit when a report on 0.10/0.11 turns chunked
prefill on through this block — it states the reason the scheduler existed ("Ascend scheduler was
added for non chunk prefill case before, since that the npu ops didn't work well with chunked
prefill. Now the ops with chunked prefill work better, it's time to remove the ascend scheduler to
use vLLM default scheduler"). Because flat `additional_config` keys are read with `.get()` in that
period, a serve line copied onto 0.12+ still starts and the block is silently ignored, so the
scheduler the reporter actually ran is vLLM's default, not the Ascend one.

The sibling graph key went the same way in the same release: `v0.11.0`'s notes announce "Torchair
graph has been deprecated, it'll be removed in `v0.12.0rc1`", and `v0.12.0rc1`'s say "Torchair graph
mode is removed. `--additional-config {"torchair_graph_config": {"enabled": true}}` doesn't work
anymore. Please use aclgraph instead." (`release_notes.md:1232` and `:1275-1276` as carried at
`v0.23.0rc1`; the same note says the removed scheduler key was spelled `ascend_scheduler` there).
So on any tag from `v0.12.0rc1` a serve line carrying `torchair_graph_config` — with
`use_cached_graph`, `mode`, `enable_multistream` — starts normally in ACLGraph mode and those keys
are dropped by the same per-key `additional_config.get(...)` reads. Do not build a cached-torchair-graph
explanation (static graph pool per shape, graph-cache memory never returned) for such a report: the
mode named in the report never ran. Find the note with
`git grep -n '"torchair_graph_config"' <tag> -- docs` and read its enclosing `## vX.Y` header — the
removal sentence sits many releases back in a file that accumulates every older section.

## `SLO_limits_for_dynamic_batch`: the key installs a scheduler subclass that lags upstream

Reported shape: `TypeError: SchedulerDynamicBatch.__init__() got an unexpected keyword argument
'hash_block_size'` at engine start, one line after the generic
"Using custom scheduler class vllm_ascend.core.scheduler_dynamic_batch.SchedulerDynamicBatch.
This scheduler interface is not public and compatibility may not be maintained." Verified at
vllm-ascend `v0.23.0`, whose `Dockerfile:42 ARG VLLM_TAG=v0.23.0` pairs it with vLLM `v0.23.0`.

1. Key and doc surface: `--additional-config '{"SLO_limits_for_dynamic_batch": N}'`, read at
   `vllm_ascend/ascend_config.py:238` (default `-1`); the tag's own
   `docs/source/user_guide/configuration/additional_config.md:83` and
   `user_guide/feature_guide/dynamic_batch.md` document `-1`/`0` as "disabled" and `>0` as
   enabled, integer type.
2. The guard is `!= -1`, not `> 0` (`vllm_ascend/platform.py:713-718`): the documented `0`
   still installs `SchedulerDynamicBatch`, and a non-integer value such as `1.0` passes too.
   The same block forces `enable_chunked_prefill = True`.
3. At engine start upstream resolves the configured class and calls it with the *whole* upstream
   signature: `Scheduler = vllm_config.scheduler_config.get_scheduler_cls()` in
   `vllm/v1/engine/core.py:136`, then `Scheduler(..., block_size=scheduler_block_size,
   hash_block_size=hash_block_size)` at `:149-158` (`resolve_kv_cache_block_sizes` `:145`).
   Upstream `Scheduler.__init__` has carried `hash_block_size` since vLLM `v0.20.0` (absent at
   `v0.19.1`).
4. The plugin subclass enumerates its arguments and never gained the kwarg:
   `vllm_ascend/core/scheduler_dynamic_batch.py:126-144` at `v0.23.0`, with the same keyword list
   (through `block_size: int | None = None`) at `v0.18.0`, `v0.20.2rc1`, `v0.21.0rc1` and
   `v0.22.1rc1`. So this is a constructor mismatch,
   not a model, KV-cache or NPU fault — the reporter's `LLM(...)` loads weights and builds the KV
   cache before the scheduler is constructed.
5. Two working shapes exist in the same tag, and they are the fix template:
   explicit optional keyword with the reason in a comment —
   `core/scheduler_profiling_chunk.py:57-86` ("`hash_block_size` was added in vLLM #40946; keep it
   optional so the subclass works on both pinned vllm and main") and
   `patch/platform/patch_balance_schedule.py:56-73`; and `*args, **kwargs` pass-through —
   `core/recompute_scheduler.py:95-98` and `:1124-1125`, which is why
   `RecomputeScheduler`/`AsyncRecomputeScheduler` are immune to this class of breakage.
6. Lifecycle: the feature was deleted rather than patched. `2b115d02d` "[Refactor] Remove dynamic
   batch slo (#11933)" (2026-07-15) drops the 595-line scheduler, the `ascend_config.py` field,
   the `platform.py` block, both docs pages and the mkdocs nav entry; the first release tag
   containing it is `v0.24.0rc1`, and it is absent from `v0.23.0`, `v0.23.0.post1` and
   `v0.23.0rc1`. `git log --all -S'hash_block_size' -- vllm_ascend/core/scheduler_dynamic_batch.py`
   is empty: no fix ever landed on that file.

Answer shape: wherever the paired vLLM already passes `hash_block_size` (from vLLM `v0.20.0`)
against a plugin tag whose `SchedulerDynamicBatch` still has the old ctor — checked at
`v0.20.2rc1`-`v0.23.x` — the option is unusable: drop the key (the default `-1` leaves vLLM's own
`Scheduler` in place), add `hash_block_size: int | None = None` to the subclass, or move to
`v0.24.0rc1`+ where the option no longer exists. For any plugin-installed `scheduler_cls`, the cheap general check is one pair of
commands at the paired tags: list every `Scheduler` subclass under `vllm_ascend/`
(`git grep -n 'class .*(Scheduler)' <ref> -- vllm_ascend`) and compare each `__init__` keyword set
with `vllm/v1/core/sched/scheduler.py`'s.

Boundary: read from the pinned trees, nothing run; the reported reproducer is an offline
`LLM(...)` call (HTTP serving not exercised), and the "`0` also installs the class" statement is
code-read, not executed.

## `weight_nz_mode` lifecycle (the #11624 variable)

- Meaning: `0` disabled, `1` only quantized tensors use FRACTAL_NZ (default at every tag
  checked), `2` also use NZ for float/BF16/FP16. `xlite_config.weight_nz` is set from
  `weight_nz_mode == 2`.
- RL interaction differs by ref: at v0.22.1rc1 `worker.py:234-240` `wake_up()` raises
  `ValueError` with "FRACTAL_NZ mode is enabled. This may cause model parameter precision
  issues in the RL scenarios. Please set weight_nz_mode=0 via --additional-config." when the
  mode is nonzero, while `main`'s RL path logs a warning and overrides
  `ascend_config.weight_nz_mode = 0`. So a sleep/wake RL deployment can need the switch
  explicitly, and the failure mode is an exception, not a silent override.
- `VLLM_ASCEND_ENABLE_NZ` exists at v0.22.1rc1 (`envs.py:92`), v0.23.0, and on
  `origin/releases/v0.24.0rc`, `v0.25.1rc`, `v0.26.0rc`. It is removed by `0c4d84e7f`
  "[Refactor][Config] Remove weight NZ environment variable (#14830)", whose first
  containing tag is `v0.27.1rc1`; `main` keeps only the field `weight_nz_mode: int = 1`
  with a 0/1/2 `ValueError` validation and the RL override. Check the reported tag before
  quoting the variable.
- The maintainer answer for #11624: the variable is being phased out, use
  `--additional-config '{"weight_nz_mode": 0}'`. Model tutorials at v0.22.1rc1/v0.23.0
  still print `export VLLM_ASCEND_ENABLE_NZ=0` (`docs/source/tutorials/models/DeepSeekOCR2.md:138`,
  GLM5.2 docs) although the switch is deprecated — a doc/code disagreement, not a second
  supported spelling.
- **NZ conversion had no singleton-dimension guard before `v0.28.0.rc1`, and that is the
  `161002`/`EZ1001` start-up failure.** `_should_trans_nz` (`vllm_ascend/utils.py:209-236` at
  `v0.22.1rc1`; `:243`-area at other tags) converted every eligible weight to FRACTAL_NZ with no
  shape check, and `aclnnMatmulWeightNz` rejects a one-length output or reduction dim:
  `AclNN_Parameter_Error(EZ1001): Not support mat2 n = 1 or k = 1 when foramt is FRACTAL_NZ,
  mat2: [2048, 1]` surfacing as `RuntimeError: matmul_implement_npu ... call aclnnMatmulWeightNz
  failed, error code is 161002`. The weights that hit it are the 1-output gate linears of a MoE's
  shared experts — upstream `qwen2_moe.py:149-155` builds `ReplicatedLinear(config.hidden_size, 1)`
  and `:120` applies it as `F.sigmoid(self.expert_gate(x)[0]) * out` — because
  `AscendUnquantizedLinearMethod.process_weights_after_loading` NZ-casts them
  (`vllm_ascend/ops/linear.py:87` at `v0.22.1rc1`) and `apply` runs `unquantized_gemm` → `F.linear`
  (`:52`, `:89`). Fix `d43f2475a` (#16058, 2026-09-17) returns `False` for `shape[-1] == 1 or
  shape[-2] == 1` before the FORCE_NZ branch, so 310P too; **first containing tag `v0.28.0.rc1`**.
  Workaround before it: the default `weight_nz_mode` is `1` (quantized weights only), so removing
  the launcher's `VLLM_ASCEND_ENABLE_NZ=2` (or `--additional-config '{"weight_nz_mode": 1}'` / `0`)
  suffices — mode `2` is exactly what NZ-casts BF16/FP16. `multistream_overlap_shared_expert: true`
  decides only *when* it surfaces: the load-time self-check runs the shared experts
  (`ops/fused_moe/fused_moe.py:478-491`, `:501` at `v0.22.1rc1`), so without the flag the same
  dispatch is reached at the first forward — inference from the code, not measured.
- The same deprecation shape applies to the other migrated switches; `vllm_ascend/envs.py`
  comments `VLLM_ASCEND_ENABLE_FLASHCOMM1` as "DEPRECATED: use
  additional_config.enable_flashcomm1 instead" at v0.22.1rc1.

## `enable_mlapo`: default `true` in the docs, inert through three separate gates

At `v0.21.0rc1`/`v0.22.1rc1` the tag's `configuration/additional_config.md` lists
`enable_mlapo` default **`True`** (`:97`) beside `enable_flashcomm1` default `False` (`:93`), but the
flag value is only the first gate:

- `enabling_mlapo(vllm_config)` (`vllm_ascend/attention/utils.py:381-391`; `:511` at `v0.23.0`)
  returns the value unchanged on A5 and, on every other SoC, `config_val and is_decode_instance`
  where `is_decode_instance` needs `kv_transfer_config is not None and is_kv_consumer and not
  is_kv_producer`. On A2/A3 a single instance, a PD **prefill** node or a plain non-PD serve has
  MLAPO off whatever the flag says.
- Per layer, non-W8A8 weights disable it again: `mla_v1.py:962-978` requires
  `fused_qkv_a_proj.quant_method` to be `AscendW8A8LinearMethod`/`AscendW8A8MXFP8DynamicLinearMethod`
  and otherwise logs `warning_once` "mlapo only supports W8A8 quantization in MLA. Some layers not
  W8A8 quantized, mlapo disabled for these layers."; `sfa_v1.py:576-597` does the same for SFA
  ("Currently mlapo only supports W8A8 quantization in SFA scenario on non-A5 devices.").
- DSA-CP excludes it: the same `sfa_v1.py` reason list adds "Currently mlapo does not support SFA
  with CP" whenever `enable_dsa_cp` is on, so `{"enable_dsa_cp": true, "enable_mlapo": true}` is a
  silent conflict (one `warning_once`, not an error) at least through `v0.23.0`.

Consequence for triage: "I set `enable_mlapo` and nothing changed" is expected on a non-PD or
non-W8A8 run, and a garbled-output fix that bundles `enable_mlapo` with something else cannot be
attributed to MLAPO without first showing the run was a W8A8 PD decode node. The Kimi tutorials and
the Kimi-K2.7 e2e config set `VLLM_ASCEND_ENABLE_MLAPO=1` **and**
`VLLM_ASCEND_ENABLE_FLASHCOMM1=1` together, so read the pair; when the aim is the SP/reduce-scatter
path, the effective switch is `enable_flashcomm1`.

## `enable_fused_mc2` and "MC2": two comm types behind one word

Reports say "MC2" (or 开MC2) for either of two code paths. Decide which one the serve line enables
before reading any MoE code; they have different knobs, prerequisites and memory behaviour.

- **`MoECommType.MC2` is automatic, not a flag.** `select_moe_comm_method`
  (`vllm_ascend/ascend_forward_context.py:233`, docstring `:236-252`, rules `:270-281` at
  `v0.22.1rc1`) returns
  `ALLGATHER` when `enable_expert_parallel` is off or `get_ep_group().world_size == 1`; on A2 it
  returns `MC2` only for `num_experts_per_device <= 24` and `ep_world_size >= 16` and
  `num_tokens <= get_mc2_tokens_capacity()`, else `ALLGATHER`; A3/A5 have their own rules. So "MC2
  enabled" on an EP MoE run is usually this default path with nothing in `--additional-config`.
- **`MoECommType.FUSED_MC2` needs `enable_fused_mc2`** (deprecated env alias
  `VLLM_ASCEND_ENABLE_FUSED_MC2`; `_get_config_value` at `ascend_config.py:150-154` makes the
  `--additional-config` key win). The env dict comment (`envs.py:95-102`) is the contract:
  `0`/unset = plain ALLTOALL + MC2; `1` = `dispatch_ffn_combine`, only for W8A8 MoE layers with
  `EP <= 32`, non-MTP, non-dynamic-EPLB; `2` = `dispatch_gmm_combine_decode`, **decode node only**,
  W8A8, and the MTP layer must be W8A8 too.
- **Per-step selection on A3** (`ascend_forward_context.py:286-305`):
  `dispatch_ffn_combine_enable = get_ep_group().world_size <= 32 and (not is_draft_model)`; below
  `mc2_tokens_capacity` the decode path is `FUSED_MC2`/`MC2`, above it `FUSED_MC2`/`ALLTOALL`; value
  `2` turns the prefill branch off outright. The op calls are
  `ops/fused_moe/moe_comm_method.py:309` (`dispatch_ffn_combine`) and `:329`
  (`dispatch_gmm_combine_decode`) inside `FusedMC2CommImpl.fused_experts` (`:285`).
- **BF16 (or any non-`w8a8_dynamic`) MoE cannot reach `dispatch_ffn_combine` from a flag — a
  report that names the op on BF16 has patched `vllm_ascend/`.** At `v0.15.0rc1` the A3 branch is
  `ascend_forward_context.py:252` `fused_mc2_enable = VLLM_ASCEND_ENABLE_FUSED_MC2 and
  quant_type == "w8a8_dynamic"` plus `:253` `dispatch_ffn_combine_enable = get_ep_group().world_size
  <= 32 and (not is_draft_model) and (not dynamic_eplb)`; the BF16 kernel does exist in the tree
  (`csrc/dispatch_ffn_combine_bf16/`), but nothing in Python selects it, and
  `FusedMC2CommImpl.fused_experts` asserts `w1_scale`/`w2_scale` are not `None`
  (`ops/fused_moe/moe_comm_method.py:321`), of which a BF16 MoE has none. Issue #7154
  (`v0.15.0rc1`, BF16 Qwen3-235B-A22B, CANN 8.3.RC2) says in its body that the reporter modified
  the related files. Report the gate and that the run is a local hack; do not read it as a
  supported-flag defect, and do not look for a `VLLM_ASCEND_*` switch that allows BF16.
- **That case's observable shape and the reporter-verified workaround:** start-up, after model
  loading and graph compilation, a 60-s `shm_broadcast.py:542` repeat loop and no traceback
  (the capture-stall section of [log-signatures.md](log-signatures.md)). The reporter followed up
  that the op OOMs once its M dimension exceeds 2048 (prompt > 2048 at batch 1, or 2048/n at batch
  n) and that the hang disappears with the input length below 2048 — so the actionable cap is
  `--max-num-batched-tokens` and the profiling dummy run's token count, not the graph config. The
  2048 threshold is the reporter's measurement, no maintainer root cause: label it as such.
  Forward locators on the op, none of which establishes the fix for this hang:
  `44ef9a36a` (#7198, bf16 precision + "remove redundant sync", first tag `v0.18.0rc1`),
  `88f7fe410` (#11258, hardcoded `max_output_size` → configurable `mega_moe_max_tokens`, first tag
  `v0.23.0`, the constant is `65536` at `v0.15.0rc1` `moe_comm_method.py:344`), `6ae53224f`
  (#12896, removes unused custom ops, first tag `v0.27.1rc1`).
- **Start-up conflict whose message is the workaround:** `platform.py:761-765` raises
  `ValueError("fused mc2 op cannot be used with hierarchy communication.Please disable
  VLLM_ASCEND_ENABLE_FUSED_MC2 by setting it to 0.")` when both `enable_mc2_hierarchy_comm`
  (`ascend_config.py:267`, `additional_config` only) and `enable_fused_mc2` are set.
- **Memory lead, candidate only.** The value-`1` call passes a hardcoded
  `max_output_size=131072` (`moe_comm_method.py:320` at `v0.22.1rc1`; `65536` at the
  `v0.14.0rc1` `:327` cited in [log-signatures.md](log-signatures.md), so read the tag, not the
  constant) next to
  `out = torch.empty_like(hidden_states)`, so that dispatch buffer is sized by a token cap rather
  than by the request — a plausible reading of "OOM on the single first curl" on a large-hidden
  MoE, and nothing about it was measured. A/B it (`enable_fused_mc2: 0`, then `npu-smi` during the
  first request) before presenting it as a cause.
- **Which CANN-9.0 known issue to quote at `v0.22.1rc1`:** the MC2 accuracy/start-up cluster is in
  the `v0.21.0rc1` section (`release_notes.md:180`, "GLM5/GLM5.1 W4A8 ... CANN 9.0 with MC2 can
  return inaccurate output, FlashComm can fail during model startup, and MTP weight loading can
  fail in 1P1D A3 deployments", #9395/#9658/#9655); the GLM5.1 1P1D 140K hang + `Kimi-K2.5` MC2 OOM
  pair is the tag's own Known Issues entry at `:44` (#10963). Match the symptom, do not merge them.

## `compilation_config`: `custom_ops` is overwritten, and `pass_config` only switches the sequence-parallel passes

A report asking "I passed `custom_ops: []` / `disabled_custom_ops: ['all']` and the EngineCore
config still shows `custom_ops: ['all']` — is that expected?" is answered yes, from two places:

- `AscendPlatform.check_and_update_config` overwrites the field for every device except 310P:
  "Activate custom ops for v1, except on 310P" → `if get_ascend_device_type() !=
  AscendDeviceType._310P: compilation_config.custom_ops = ["all"]`
  (`vllm_ascend/platform.py:487` at `v0.20.2rc1`; `:356` at `v0.15.0rc1`, `:675` at `v0.23.0`,
  `:1377` at `v0.30.0`, `:1482` on `main`), so a user-supplied `custom_ops` list is discarded
  before the engine config is logged.
- `enabled_custom_ops` / `disabled_custom_ops` are not inputs: both are
  `field(default_factory=Counter, init=False)` on vLLM's `CompilationConfig`
  (`vllm/config/compilation.py:716-718` at `v0.20.2`) and are filled in during compilation, so
  passing them in the `compilation_config` dict does nothing.

**The two fusion switchboards are different, and a "I disabled the fusion passes and it still
fails" experiment is usually inert.** Upstream `PassConfig` (`vllm/config/compilation.py:107-140`
at `v0.20.2`) carries `fuse_norm_quant`/`fuse_act_quant`/`fuse_allreduce_rms`/`enable_sp`, but at
`v0.20.2rc1` the plugin reads only `pass_config.enable_sp` (and `sp_min_token_num`) from it —
`platform.py:203-208`, `:473`, and `compilation/graph_fusion_pass_manager.py:72`, and every other
`pass_config` reference in the plugin tree (`ascend_config.py:265`, the `platform.py:356` comment,
`worker/model_runner_v1.py:4474-4480`) is that same `enable_sp` field. The quant/norm/rope/muls-add
passes are gated on `additional_config["ascend_compilation_config"]` instead — but the
**sequence-parallel pair is not**: `graph_fusion_pass_manager.py:72-77` appends
`SequenceParallelismPass` + `SequenceParallelismMoePass` when
`config.compilation_config.pass_config.enable_sp` is true (identical at `v0.18.0` and at
`v0.20.2rc1`), so `pass_config.enable_sp` is a live switch, not an inert one. For the other passes,
`graph_fusion_pass_manager.py:50-70` does `self.ascend_compilation_config =
config.additional_config.get("ascend_compilation_config", {})` and then
`.get("fuse_norm_quant", True)`, `.get("fuse_qknorm_rope", True)`,
`.get("fuse_allreduce_rms", True)`, `.get("fuse_muls_add", True)`; `AscendConfig` builds
`AscendCompilationConfig(**additional_config["ascend_compilation_config"])`
(`ascend_config.py:39-40`, fields at `:485-521`). So the working command is
`--additional-config '{"ascend_compilation_config": {"fuse_norm_quant": false}}'` — the form the
tag's own `docs/source/user_guide/configuration/additional_config.md` documents and
`tutorials/hardwares/310p.md:122` uses. A report listing `pass_config: {fuse_norm_quant: false}`
has not turned that pass off; say so rather than accepting the failed experiment as evidence
about the fusion pass.

## `pass_config.enable_sp: true` without FlashComm1: output that stops mid-sentence on `finish_reason: "stop"`

`#12278` (`vllm-ascend:v0.18.0-a3`, Qwen3.5-397B-A17B-w4a8-mtp, 910C 4 cards/8 dies, reported as
"v0.18.0-a3 and later") pastes a chat completion whose body is cut mid-phrase ("...基于 Q") with
`completion_tokens: 97`, `finish_reason: "stop"`, `stop_reason: 248044`. The thread, not the body,
carries the useful half: output was normal on `v0.17.0rc1-a3`, the failing launcher had added
`"pass_config": {"enable_sp": true}` to `--compilation-config` for a ~5% TTFT gain, and a second
910C host (8 cards) with the same model and flags did not show it. Reporter-verified remediation:
delete that entry, or `export VLLM_ASCEND_ENABLE_FLASHCOMM1=1`. The `finish_reason` discriminator
that keeps this report out of the context-length class is in
[served-context-length.md](served-context-length.md).

At the bare `v0.18.0` tag the `-a3` image is built from
([model-tutorial-recipes.md](model-tutorial-recipes.md)), the flag is **read**, so do not repeat the
thread's "this parameter is not supported from v0.18.0" (their inference from the compute team; the
tag says otherwise):

- `AscendConfig.enable_sp_by_pass = model_config is not None and not enforce_eager and
  compilation_config.pass_config.enable_sp` (`ascend_config.py:163-167`).
- `AscendPlatform.apply_config_platform_defaults` fills `sp_min_token_num` from it — docstring
  "Set sp_min_token_num=1 when enable_sp and not set" (`platform.py:197-208`).
- `GraphFusionPassManager.configure` appends the two SP passes (`graph_fusion_pass_manager.py:72-77`).
- Runtime gates read it beside the env switch: `ops/fused_moe/prepare_finalize.py:327`, `:436`
  (`enable_sp() or enable_sp_by_pass()`); `ops/register_custom_ops.py:46`
  (`_EXTRA_CTX.flash_comm_v1_enabled or (enable_sp_by_pass() and is_ep_comm)`), `:80`, `:93`, `:118`;
  `worker/model_runner_v1.py:1848` pads tokens to TP for SP on the same pair.
- `worker/model_runner_v1.py:3435-3444 update_pass_config` temporarily replaces the field with the
  env-gated `enable_sp(vllm_config)` around `_check_and_update_cudagraph_mode` (`:3318`), so it is
  not one value: upstream `vllm/compilation/backends.py:847 configure_post_pass` and the runtime
  gates can read different inputs.

**Do not forward "SP fusion without the comm" as the mechanism.** The runtime honours
`enable_sp_by_pass()` on the MoE prepare/finalize, the EP-comm all-gather and the token padding
above, so a pure fusion-vs-comm mismatch is not what these sites show — that was the tempting
explanation and it does not survive the grep. What is established is the reporter's A/B (flag off, or
FLASHCOMM1 on, fixes it) plus the two-host delta that the flag alone does not break every host.
Which host-side input decides it (the SP passes, `sp_min_token_num`, or a compile/capture difference
between the two machines) is open. The separating evidence is the reporter's A/B plus a logits or
`prompt_logprobs` comparison on one prompt with the entry removed vs `FLASHCOMM1=1`; the thread is
still open with 3 comments (last one a ping), so no maintainer analysis exists.

The same `pass_config.enable_sp` form is what the VL path redirects to when `platform.py` refuses
Flash Comm V1 ([vision-tower-accuracy.md](vision-tower-accuracy.md)), so this report class is not
confined to text checkpoints.

## Boundary

Source- and thread-verified: the validation mechanism, the resolution order, the lifecycle
per tag, and the maintainer's stated answers. For the `weight_nz_mode` singleton-dim failure
(#11621) the mapping is now source-verified rather than reporter-claimed — the reported tag has no
guard, the log's op and shape (`aclnnMatmulWeightNz`, `mat2: [2048, 1]`) match a 1-output gate
linear, and the later fix states the same CANN restriction — but nothing here was run, so keep
"not reproduced on hardware" in the answer and do not cite the workaround as confirmed.

## Triage rules for this topic

These rules are kept with the topic they decide; SKILL.md carries the routing line and only
cross-cutting rules.

- **Config switches are non-monotonic across releases.** A flag can exist at the reported
  tag, be absent on `main`, be re-added later and be removed again, and a report naming two
  versions must be checked at both tags. Absence on `main` proves nothing about the
  reported release — test the reported tag itself, as the `c8_enable_reshape_optim`
  lifecycle shows. A knob also migrates between config surfaces: `VLLM_ASCEND_BALANCE_SCHEDULING`
  is read from `envs.py` at `v0.23.0`/`v0.26.0rc1`, but the env fallback is gone at `v0.27.1rc1`
  and later, where only `--additional-config '{"enable_balance_scheduling": true}'` arms the
  patch. Before crediting a suspected flag with any effect, confirm that the surface the reporter
  used is still wired at that tag.

- **A vllm-ascend patch frame can be a disabled pass-through.** At
  `origin/releases/v0.21.0rc`, `patch_balance_schedule.py:73` is `return super().schedule()`
  under `if not self._balance_enabled:` — the frame proves the patch is off and the stock
  scheduler is executing, not that the patch is at fault.

- **A documented exclusion can answer the report faster than the dispatch code.**
  `VLLM_ASCEND_BALANCE_SCHEDULING` (later `--additional-config '{"enable_balance_scheduling":
  true}'`) is documented as "not recommended in scenarios where PD is separated", so a PD report
  that arms it is outside the documented topology and the doc sentence is the finding. Grep the
  tag's docs for any maintainer-named flag before reading its guard:
  `git grep -n '<FLAG>' <tag> -- 'docs/*' | grep -v locale`.

- **A pasted vendor/performance env bundle is filtered against the reported tag before any of it is
  suspected.** A `docker run -e ...` block copied from a Huawei performance guide can name knobs
  from a different lineage than the tag under test — some newer, some retired before it
  (`VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE` is gone from `v0.9.2rc1` on, per
  [sampling-and-output-quality.md](sampling-and-output-quality.md)). At `v0.18.0rc1`,
  `git grep -n <NAME> v0.18.0rc1 -- vllm_ascend` returns zero hits for
  `VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE`, `VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE` and
  `VLLM_ASCEND_ENABLE_PREFETCH_MLP`, while `VLLM_ASCEND_ENABLE_NZ` (`envs.py:91`, weight-format
  only) and `VLLM_ASCEND_BALANCE_SCHEDULING` (`envs.py:105`) are live. Scope the inference by
  namespace: the plugin reads its knobs through `envs.py`, so a `VLLM_ASCEND_*` name that appears
  nowhere in `vllm_ascend/` has no plugin reader and can be dropped from the suspect set — the
  upstream "Unknown vLLM environment variable detected" line for it is cosmetic (section above),
  not evidence that the knob was applied. `HCCL_*`, `ATB_LLM_*`, `MIES_*`, `TASK_QUEUE_ENABLE`,
  `CPU_AFFINITY_CONF` and `LD_PRELOAD` are owned by CANN/torch_npu/ATB/MindIE — a plugin-tree
  miss says nothing about them, so either check
  their own tree under `/repos/Ascend/<repo>` or label them unexamined. One surviving name is
  usually the whole candidate set; the working deployment's value for it is the workaround and a
  one-variable rerun is the confirming check.
