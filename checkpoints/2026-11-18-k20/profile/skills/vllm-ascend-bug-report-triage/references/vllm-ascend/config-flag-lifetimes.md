# `--additional-config` flags: parsing traps and the sparse-C8 / StoreKVBlock chain

A serving flag can change meaning three ways across releases: a new key replaces an old one, the
gating expression changes, or the whole implementation is deleted. All three happened to the
sparse-C8 family, which is why a C8 report needs the tag checked before any interpretation. The same
applies to the `VLLM_ASCEND_*` environment variables §6–7 cover: honoured locally, unknown upstream,
and on 310P one of them does nothing at all.

Verified against the local clone (`main` = `1bd23a0a9`, 2026-10-30); tags `v0.20.2rc1` …
`v0.30.0`. Recheck with `git -C "$REPO" grep -n -F '<key>' <ref> -- '*.py'` before quoting line
numbers.

## 1. Parsing and the two traps

`vllm_ascend/ascend_config.py` → `AscendConfig(vllm_config)` reads `vllm_config.additional_config`
into attributes; some values also fall back to `VLLM_ASCEND_*` environment variables through a
`_get_config_value(additional_config, key, env_key, env_default)` helper.

**Trap A — a switch is not evidence that the switch did something.** In releases up to at least
`v0.23.0rc1` the parser reads keys with `additional_config.get(...)`, so a **renamed or removed key
is silently ignored** and the feature stays off. Later `main` enforces unknown-key rejection with
pydantic `extra="forbid"` for the sub-configs (with a compatibility shim for `vllm_omni`), so
behaviour differs by release. Consequence for triage: when a reporter shows a "working" config and
a "failing" config, verify that every flag in the working one exists at their ref — a deprecated
key makes the working config equivalent to "feature disabled", which is often the whole answer.

**Trap B — flags gated at runtime.** Configuration is often ANDed with a model capability, so
setting the flag does nothing:

```python
# v0.23.0rc1 ascend_config.py
use_sparse = model_uses_sfa_sparse(vllm_config.model_config)
self.enable_sparse_sfa_c8 = additional_config.get("enable_sparse_sfa_c8", False) and use_sparse
self.enable_sparse_li_c8  = additional_config.get("enable_sparse_li_c8", False)  and use_sparse
self.c8_enable_reshape_optim = self.enable_sparse_li_c8 and additional_config.get(
    "c8_enable_reshape_optim", False
)
```

Also cross-checked at config time: `_check_mooncake_c8_kv_cache_quant()` validates the C8 KV-cache
+ GQA backend + Mooncake connector combination, i.e. that combination has explicit constraints
worth quoting in a report about MemCache + C8.

**Trap C — the flag name in a report may be the *model's* quantization, not a flag.**
`enable_sparse_sfa_c8` / `enable_sparse_li_c8` / `enable_sparse_c8` describe **KV-cache** (fp8/int8)
paths, derived in places from `cache_config.cache_dtype` or `attention_config.indexer_kv_dtype`
rather than from `additional_config` (e.g. later `worker/v1/model_runner.py`,
`worker/v2/attn_utils.py`). Newer releases unify the dtype through `--kv-quant-dtype` (#15514).
Check which mechanism is in force at the reporter's ref before blaming a flag.

**Trap D — a value the user sets can be rewritten by the platform.** Not ignored: overridden. At
`v0.11.0rc0`, `vllm_ascend/platform.py` (`check_and_update_config`) raises
`scheduler_config.max_num_batched_tokens` to `max_model_len` for non-MLA models whenever the Ascend
scheduler is active, `async_scheduling` is off, and the ascend scheduler's own chunked-prefill switch
is off — so `--max-num-batched-tokens 8192` is a no-op and a long prompt is prefilled in one step.
The head of that section logs *"Non-MLA LLMs forcibly disable the chunked prefill feature"*, and
`AscendSchedulerConfig.__post_init__` (`vllm_ascend/core/schedule_config.py`) raises instead when
`max_num_batched_tokens < max_model_len` without chunked prefill. Consequence for triage: a
workaround built on a numeric flag must name the switch that lets the number survive, e.g.
`--additional-config '{"ascend_scheduler_config": {"enable_chunked_prefill": true}}'`. Grep the flag
at the ref (`git show <tag>:vllm_ascend/platform.py | grep -n <FLAG>`) before promising it, and read
whether it is *derived*, *ignored* or *rewritten*.

**Trap D, the profile-run consequence.** That rewritten `max_num_batched_tokens` also sizes the worker's
memory profile: `NPUModelRunner.__init__` sets `self.max_num_tokens = scheduler_config.max_num_batched_tokens`
(`worker/model_runner_v1.py:256` at `v0.11.0`) and `profile_run` dummy-runs exactly that many tokens
(`:2567-2571`), after which `determine_available_memory` subtracts the measured peak from
`total × gpu_memory_utilization` (`worker/worker_v1.py:219-259`). On a model whose `max_model_len` is the
checkpoint's 262 144, the rewrite therefore makes startup refuse with `Available memory: 0` and
`No available memory for the cache blocks` **at any** `--gpu-memory-utilization` — the flag the message
recommends. Lever: `--max-model-len`, or the `enable_chunked_prefill: true` key above (it skips the rewrite),
or `--async-scheduling`; `../symptoms/startup-and-stall.md` §3c, #3843. The block is release-scoped: present
at `v0.11.0rc0` (`platform.py:139-163`), removed by #3967 (`a1558b99c`) on main and #4094 (`ebd45b659`) on
the release branch (in `v0.11.0rc1`+, not `rc0`), so a `v0.11.0` image whose log prints `platform.py:N Non-MLA LLMs forcibly disable the
chunked prefill feature` is running pre-removal code whatever its tag says.

**Trap D, second member — a parallelism key replaced by the platform's own algebra.** At
`v0.9.2rc1`, `vllm_ascend/platform.py:135-149` rewrites `parallel_config` unconditionally:

```python
parallel_config.expert_tensor_parallel_size = parallel_config.tensor_parallel_size   # :135
if parallel_config.enable_expert_parallel:
    parallel_config.expert_tensor_parallel_size = 1                                   # :140
elif ascend_config.expert_tensor_parallel_size > 0:                                   # :143
    parallel_config.expert_tensor_parallel_size = ascend_config.expert_tensor_parallel_size
parallel_config.expert_parallel_size = (
    parallel_config.world_size_across_dp // parallel_config.expert_tensor_parallel_size)
```

So `additional_config.expert_tensor_parallel_size` is read **only when `--enable-expert-parallel` is
off**; with EP on, the key is dead and `ep_size = world_size_across_dp`. `distributed/parallel_state.py:28-65`
then builds the EP group as `range(i, world_size, etp)` over the whole world (DP replicas included).
Read this pair before explaining any "I set etp=N" report, and before treating an EP+DP accuracy delta as
a kernel problem (`../symptoms/benchmark-accuracy.md` §2).

Per-tag presence of a flag, its release-note mentions and the pickaxe commits that added/removed it:
`bash scripts/flag_lifecycle.sh <flag>`. The table is **not monotonic across release branches**
(they are cut and cherry-picked independently), so read the row for the reporter's exact tag.

## 2. `enable_sparse_c8` was split

- Original single key documented in #7600.
- Split into `enable_sparse_sfa_c8` and `enable_sparse_li_c8` by **#12351** (`70768c876`,
  `[v0.23.0]`, 2026-07-19) on the release branch and **#12470** (`60cba9393`, main, 2026-07-28).
  Release note: `docs/source/user_guide/release_notes.md:458` and `:583`.
- `v0.23.0rc1` **already** contains the split. A report against it that writes
  `"enable_sparse_c8": false` passes a key nothing reads: at that tag the string survives only as a
  local variable in `vllm_ascend/platform.py:725`. Treat such a report's "I turned C8 off" as "C8
  was never on for that node", and look for the key that *was* on.

## 3. `enable_sparse_sfa_c8` / `enable_sparse_li_c8` derivation moved again

| ref | how the two booleans are derived |
| --- | --- |
| `v0.23.0rc1` (`ascend_config.py:280-281`) | `additional_config.get("enable_sparse_sfa_c8"/"enable_sparse_li_c8", False) and use_sparse` |
| `v0.26.0rc1`, `v0.26.0rc2` | same expression, still `additional_config`-driven |
| `main` (`ascend_config.py:879-881`) | `enable_sparse_sfa_c8 = cache_config.cache_dtype in ("fp8","int8") and use_sparse_sfa`; `enable_sparse_li_c8 = attention_config.indexer_kv_dtype in ("fp8","int8") and use_sparse`; `enable_sparse_li_c4 = ... == "mxfp4"`. The `additional_config` keys remain only in the accepted-keys defaults table (`:450-452`) and in `quantization/methods/kv_cache/turboquant/config.py:31-36`, which *raises* if `enable_sparse_sfa_c8` is combined with `turboquant_4bit_nc`. `worker/v2/attn_utils.py:230` derives LI C8 the same way. |

So the same JSON enables C8 on the rc tags and is ignored on HEAD. Never conclude "the flag does
nothing" from a HEAD-only grep; the reporter's tag is what counts. At HEAD the documented switches
are `--kv-cache-dtype` (SFA C8) and `--attention_config.indexer_kv_dtype int8` (LI C8), as in
`docs/source/tutorials/models/GLM5.md:1519-1520`. The migration is recorded as "Unified the KV cache
quantization dtype configuration through the `--kv-quant-dtype` parameter" (#15514) in the same
release-note section as the StoreKVBlock removal.

## 4. `c8_enable_reshape_optim` / StoreKVBlock: the switch behind P-node hangs

The switch gates a grouped cache-write optimisation for the LightningIndexer C8 cache, executed by
custom AscendC/AICPU operators `store_kv_block` + `store_kv_block_metadata`
(`torch.ops._C_ascend.*`, sources under `csrc/attention/store_kv_block{,_metadata}/`, 20 files).

| when | commit / PR | state |
| --- | --- | --- |
| op added | `a53f8a023` #10292 | `csrc` + Python call sites |
| flag, A5 enablement, UTs | `b417b95e4` #11451 | opt-in key |
| `v0.23.0rc1`, `v0.26.0rc1/rc2` | `ascend_config.py:282-284` / `:288-289` | `c8_enable_reshape_optim = enable_sparse_li_c8 and additional_config.get("c8_enable_reshape_optim", False)` — needs **both** LI C8 and the explicit key; docs table lists default `False` |
| `v0.27.1rc1` | dataclass field `ascend_config.py:409`, default `False`, still ANDed (`:696`) | call site `attention/sfa_v1.py:361` |
| **`v0.28.0.rc1`** | **user key removed**: `_c8_reshape_optim_enabled = enable_sparse_li_c8 and is_prefill_node`, so there is **no opt-out** | call site `attention/indexer.py:277` |
| `v0.29.0.rc1` … `v0.30.0rc2` | field back, default `True`, ANDed with LI C8 and prefill node (`ascend_config.py:786` at `v0.29.0.rc1`, `:933` at `v0.30.0rc2`) | call site `attention/indexer.py` |
| auto-enable | `c30092741` #15752, 2026-09-07 | user key removed; enabled when SFA + LI C8 + PD **prefill-only** (`kv_producer`) node |
| switch restored, default **true** | `1e0742722` #16952, 2026-09-20 | gated by SFA + LI C8 + prefill node; code comment links issue **#15896**; PR body: "on HDK 0.26.0 or earlier this optimization may conflict with pooling models using AICPU operators, and can be disabled as a workaround" |
| removed end to end | `ac5324b2a` #18128, 2026-10-10 (main) and `c604a5c86` #18301 (`releases/v0.30.0`) | `AscendSFAIndexerBackend.write_cache` keeps only `scatter_cache`; csrc ops, `group_len`/`group_key_idx`/`group_key_cache_idx` plumbing, UTs and doc entries deleted. Release note: `release_notes.md:118` |

Membership: #15752 and #16952 are in `v0.30.0` and `main` but **not** in `v0.26.0rc1`/`rc2`
(`merge-base --is-ancestor`); #18128 is in `main` only until the #18301 cherry-pick, and both are
outside `v0.30.0rc1/rc2` while inside `v0.30.0`/HEAD. The release-notes bullet announcing the
removal is already present in `v0.30.0rc1` while the code still has the flag — release notes can run
ahead of a branch, so trust the tag's code over its prose.

Where it is used at the rc tags: metadata build `vllm_ascend/attention/sfa_v1.py:474`
(`v0.23.0rc1`), `:460-461` (`v0.26.0rc1`, calls `store_kv_block_metadata`), write path
`:1837`/`:1855` (`v0.23.0rc1`), `:1983-2006` (`v0.26.0rc1`, calls `store_kv_block`), helper
`_use_li_c8_reshape_optim` at `:1547-1549`. UTs that encode the gating:
`tests/ut/attention/a2/test_sfa_v1.py:397/635` (`v0.23.0rc1`),
`test_li_c8_reshape_optim_requires_layer_li_c8` (`v0.26.0rc1:258-269`).

Documented defaults to quote, from `docs/source/user_guide/configuration/additional_config.md`:
`enable_sparse_sfa_c8` / `enable_sparse_li_c8` default `False` at `v0.26.0rc1:81-82`, and
`c8_enable_reshape_optim` `:83` "`enable_sparse_li_c8` must also be enabled. In the PD separation
scenario, only the P node is enabled." At `v0.30.0rc1/rc2:64` the default is `True` and `:88`
documents a concrete conflict: with HDK 0.26.0 or earlier the flag "may conflict with pooling models
that use AICPU operators"; setting it to `false` is the documented remedy (issue #15896). Those doc
lines are the authoritative statement for the release they ship in.

### What this means for a report

- Reporters copying the official GLM-5.2 PD command got the key switched on: at `v0.23.0rc1`
  `docs/source/tutorials/models/GLM5.2.md:683` and `:752` contain `"c8_enable_reshape_optim": true`;
  at `v0.26.0rc2` `:505` does; `main` has no such key.
- P-node hang with MemCache pooling + sparse C8 on HDK 25.1/0.26.0: the AICPU operator is launched
  in the same process as pooling, which is the hang described in #15896. Workarounds, in order: set
  `"c8_enable_reshape_optim": false`; drop the sparse-C8 keys entirely; upgrade past #18128
  (v0.30.0+) where the path no longer exists.
- On versions where #16952 applies the switch defaults to **true**, so "it started hanging without
  a config change" is expected: the *default*, not the user's JSON, changed.
- `v0.28.0.rc1` has no key at all: there the only lever is turning `enable_sparse_li_c8` off, and an
  "opt-out did nothing" report against 0.28 is explained by the removed key rather than by a broken
  config.
- The flag is ANDed with `enable_sparse_li_c8` on every line that has it, so it is already off when
  LI C8 is off — a report where turning it off changes behaviour proves LI C8 is on.

## 5. Other documented renames and removals (`release_notes.md` at main)

- `VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL` removed; DSA-CP is now `additional_config.enable_dsa_cp`
  (#9668, #9697, #9910). FlashComm1: `VLLM_ASCEND_ENABLE_FLASHCOMM1` (env only, alongside the legacy
  `VLLM_ASCEND_ENABLE_FLASHCOMM`) is the whole switch through `v0.19.1rc1`; the
  `additional_config.enable_flashcomm1` key appears at `v0.20.2rc1` (`ascend_config.py:83`, from
  `_get_config_value`, commit 958daf832 / #9064, first tag `v0.20.2rc1`). Both spellings feed
  `enable_sp()` (`utils.py:799-816` at `v0.19.1rc1`, memoised in `_ENABLE_SP`), and
  `enable_shared_expert_dp` forces SP on — so "FLASHCOMM1 is off" is a claim to check in code, not in
  the env dump (`../symptoms/flashcomm1-and-moe-reduce.md`).
- `additional_config.expert_tensor_parallel_size` — parsed only through `v0.9.2rc1`
  (`ascend_config.py:48-49`), and even there overridden whenever `--enable-expert-parallel` is set (the
  algebra is Trap D's second member above). Removed by `8cfd25799` "[Dist][EP] Remove ETP/EP maintained in
  vllm-ascend" (#1681), first tag `v0.10.0rc1` — from there the key is Trap A (never read) and the parallel
  config comes from vLLM's own `ParallelConfig`. A command copied onto a newer tag therefore still starts,
  but with whatever EP semantics vLLM's `ParallelConfig` applies — not the ones the reporter's
  `additional_config` block implies (`../symptoms/benchmark-accuracy.md` §2).
- `additional_config.torchair_graph_config` (with its `enabled` / `use_cached_graph` /
  `enable_graph_mode` sub-keys) and `additional_config.ascend_scheduler_config` — read at `v0.9.2rc1`
  (`ascend_config.py:39`, class `:57-66`) and at `v0.11.0`/`v0.11.0rc0` (`:18-20`, `:43-46`), gone from the
  tree at `v0.12.0rc1` (`835b4c8f1` "Drop torchair (#4814)", 2025-12-10; `git ls-tree -r --name-only <ref> |
  grep -c -i torchair` 37 → 0, last tag that reads it `v0.11.0`). The removal is in the tag's own
  `docs/source/user_guide/release_notes.md`, section `## v0.12.0rc1` → `### Other`: "Torchair graph mode is
  removed. `--additional-config {"torchair_graph_config": {"enabled": true}}` doesn't work anymore. Please
  use aclgraph instead." (the Ascend scheduler and the `llmdatadist` connector are retired in the same
  block). **Nothing validates the leftover keys** — `AscendConfig.__init__` reads a fixed list with
  `additional_config.get(...)` and never warns — so a `v0.13`–`v0.23` argv that says "torchair graph
  enabled, `use_cached_graph`" starts and **silently runs aclgraph** under `cudagraph_mode`, describing
  neither mode, and no traceback reflects the dead key. Say that in the reply before reading a graph path.
  Do not date the removal from where the sentence sits in a modern tag's release notes
  (`release_notes.md:1276` when read at `v0.23.0rc1` is the **v0.12.0** section); the TorchAir timeline and
  its `LoadGraph failed` symptoms are `../hccl/init-failures.md`.
- `ASCEND_BUFFER_POOL` removed (#13834); sequence parallelism marked unavailable in the v0.23.0
  support matrix (#12860).
- Known issue recorded for v0.23.0: PP + PCP combination unsupported.
- Operator names are not stable either: custom operators whose names collided with official CANN
  names were renamed with a `_vllm` suffix — KDA gate cumsum, GDN FwdH, sparse attention, quant
  lightning indexer (#17998, #17997, #17937, #17209; `release_notes.md:115`). An operator name
  quoted in a log from an older build (for example
  `csrc/attention/kv_quant_sparse_flash_attention_vllm/`) may not match the name at another ref.
  When grepping operator names, search the bare stem (`kv_quant_sparse_flash_attention`) rather
  than the suffixed name, and check `csrc/*/build_aclnn.sh` and `csrc/torch_binding_meta.cpp` for
  the registration actually in force at that ref.

## 6. `VLLM_ASCEND_*` env vars: the false "Unknown vLLM environment variable" warning

Signature in the report: `WARNING ... [envs.py:2057] Unknown vLLM environment variable detected:
VLLM_ASCEND_ENABLE_NZ`, immediately followed by the feature working anyway — the reporter's own
observation in #11624.

Verified chain (upstream `vllm-project/vllm` `v0.22.1`, `vllm/envs.py:2051-2057`):

```python
def validate_environ(hard_fail: bool) -> None:
    for env in os.environ:
        if env.startswith("VLLM_") and env not in environment_variables:
            if hard_fail: raise ValueError(f"Unknown vLLM environment variable detected: {env}")
            else:         logger.warning("Unknown vLLM environment variable detected: %s", env)
```

It is called once from `vllm/engine/arg_utils.py:1711` with `self.fail_on_environ_validation`. The scan
matches the `VLLM_` **prefix** and compares against *upstream vLLM's* registry, so every
`VLLM_ASCEND_*` name trips it; vllm-ascend reads its own names from `vllm_ascend/envs.py` (e.g.
`VLLM_ASCEND_ENABLE_NZ` at `:92` of `v0.22.1rc1`), which is why the variable does take effect. The
warning is cosmetic — but with `--fail-on-environ-validation` (or a build that hard-fails) the same
scan raises. That CLI flag is `arg_utils.py:1513`, dataclass field `:714`, default `False`; do **not**
recommend enabling it to "silence" the line, because it turns the plugin's own variable into a startup
`ValueError`. Say that in the answer instead of telling the reporter to drop the variable.

Later refs moved the check behind a platform hook — HEAD: `vllm/engine/arg_utils.py:2267` calls
`current_platform.validate_environ(...)`, and `vllm/platforms/interface.py:1463-1476` still contains
the same prefix scan and warning. `git grep -n validate_environ main` in the vllm-ascend clone returns
**nothing**: vllm-ascend does not override the hook, so the false warning persists at HEAD and the
extension point for a fix is `AscendPlatform.validate_environ` whitelisting the keys of
`vllm_ascend/envs.py`. Treat "silence the warning" as a workaround needing a code change, not a flag.

Worked case #11624 (vLLM `0.22.1` / vllm-ascend `0.22.1rc1`, 910B3 A2, log `[envs.py:2057]`): the
reporter set `VLLM_ASCEND_ENABLE_NZ=0`, got the "unknown environment variable" warning, and the
variable still took effect (it fixed the #11621 runtime error); the issue was closed as `resolved`.
Both halves are correct and expected: the value is consumed by `vllm_ascend/ascend_config.py`
(`:220-224`, deprecation notice `:295`, "… will be removed in the next release. Please use
additional_config.weight_nz_mode"), and the upstream warning is cosmetic.

Triage rule: an "unknown environment variable" line is never the bug; treat the *effect* as the
signal. The documented migration is `additional_config.<key>` on any tag where `_get_config_value`
supplies the fallback, and the warning disappears only when the plugin stops using a `VLLM_` prefix
(or a platform override lands).

### A vendor tuning bundle: which names does the package even read?

A "deployment A hangs, deployment B works" report whose argv is identical usually differs by a **block of
env vars copied from vendor guidance**. Treat the block as candidates and check liveness at the reporter's
ref before ranking them (`git -C <vllm-ascend> grep -n '<NAME>' <ref> -- vllm_ascend`), because a name the
package never reads cannot be the cause, however performance-shaped it looks.

Verified at `v0.18.0rc1` for #8328's failing deployment (15 host/Ascend names + `--mm-processor-cache-type
shm` against an otherwise identical working deployment):

| name in the delta | reader at `v0.18.0rc1` |
| --- | --- |
| `VLLM_ASCEND_BALANCE_SCHEDULING` | yes — `envs.py:105`, gate `patch/platform/__init__.py:38` (scheduler replacement, see below) |
| `VLLM_ASCEND_ENABLE_NZ` (`=2`) | yes — `envs.py:91`, `utils.py:138-156` (`2` also converts bf16/fp16 weights) |
| `VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE` | **removed** at that tag — `docs/source/user_guide/release_notes.md:318`, `:494` ("removed and `VLLM_ASCEND_ENABLE_PREFETCH_MLP` is recommended") |
| `VLLM_ASCEND_ENABLE_PREFETCH_MLP`, `VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE` | docs/tests only in that tree (release notes, `tutorials/models/GLM4.x.md`, nightly YAMLs) — no reader under `vllm_ascend/` |
| `MIES_USE_MB_SWAPPER` | no hit anywhere in the repo (a MindIE/vendor variable) |
| `--mm-processor-cache-type` | upstream vLLM only (`vllm/config/multimodal.py`, `vllm/envs.py`, `vllm/multimodal/registry.py` at `v0.18.0`); the plugin reads `shm_broadcast` only through `patch/platform/patch_multiproc_executor.py`/`patch/worker/patch_distributed.py`, and `patch_shm_broadcast.py` does not exist at that tag |
| `TASK_QUEUE_ENABLE`, `HCCL_*`, `ATB_*`, `CPU_AFFINITY_CONF`, `LD_PRELOAD`, `OMP_*` | not the plugin's: torch_npu / HCCL / ATB / libc, i.e. **outside every clone**. The plugin's own binding switch is `additional_config.enable_cpu_binding` (`ascend_config.py:90`, `worker/worker.py:302`), and that block was in *both* deployments |

So the answer to that class is a **bisect matrix plus a declared gap**, not a plugin mechanism: unset the
whole bundle (confirming the working deployment), add it back in halves, and if none of the two live plugin
switches reproduces it, say the remaining delta acts in torch_npu/CANN/ATB where no clone can localise it.
Do not answer "the switch did nothing" from the `main` tree — the same name can be live in another release.

## 7. `VLLM_ASCEND_ENABLE_NZ` → `additional_config.weight_nz_mode` (and why it is a no-op on 310P)

Same rename shape as the C8 family, and it changes what code actually runs on the weight path.

| ref | how the NZ mode is read |
| --- | --- |
| `v0.18.0rc1` (last env-only) | read straight from `vllm_ascend/envs.py` (`VLLM_ASCEND_ENABLE_NZ`, default `1`) by `vllm_ascend/utils.py`; `ascend_config.py` has no `weight_nz_mode` |
| `v0.20.2rc1`, `v0.21.0rc1`, `v0.22.1rc1` | `weight_nz_mode` present in `ascend_config.py`, env var kept as the deprecated fallback |
| `v0.22.1rc1` (line anchors) | `vllm_ascend/ascend_config.py:220-225` — `self.weight_nz_mode = self._get_config_value(additional_config, "weight_nz_mode", "VLLM_ASCEND_ENABLE_NZ", ascend_envs.VLLM_ASCEND_ENABLE_NZ)`; the env var path logs "falls back … will be removed in the next release". Consumed at `utils.py:222-235`, docs table `docs/source/user_guide/configuration/additional_config.md:25` (env→field rename) and `:98` (default `1`). |

Semantics at `v0.22.1rc1` (`ascend_config.py:218-219` comment, `utils.py:222-235`): `0` disables NZ,
`1` converts quantised weights only (default), `2` also converts bf16/fp16 weights; fp32 and meta
tensors are never converted, and **310P always converts** regardless of the mode. So
`VLLM_ASCEND_ENABLE_NZ=0` and `--additional-config '{"weight_nz_mode": 0}'` are the same request, and
on 310P neither disables the conversion. The maintainer reply on #11624 worded this as "You can enable
it by using `--additional-config '{"weight_nz_mode": 0}'`" — quote the mapping, not the verb: `0` is the
*disable* value.

Why the 310P claim holds, checked in the code rather than inferred from the mode table:
`vllm_ascend/utils.py:261` `_should_trans_nz(weight)` returns `True` **before** it reads
`ascend_config.weight_nz_mode` (`:276-287` at `v0.23.0rc1`) when `is_310p()` (`:271-272`), so on 310P
the FRACTAL_NZ weight layout is unconditional; on A2/A3 the mode *is* honoured (`nz_mode == 0` → no
conversion, bf16/fp16 only at mode 2), which is why #11624's workaround worked on a 910B3. A 310P
report whose workaround hinges on this key is wrong — see `../symptoms/device-kernel-faults.md`.

### `VLLM_ASCEND_BALANCE_SCHEDULING` / `enable_balance_scheduling`

The switch behind the PD decode hang in `../symptoms/pd-collective-hang.md` moved the other way —
env-only first, config key later:

| ref | how `BalanceScheduler` is installed |
| --- | --- |
| `v0.18.0rc1`, `v0.19.1rc1` | `patch/platform/__init__.py:38` — `if envs.VLLM_ASCEND_BALANCE_SCHEDULING: import ...patch_balance_schedule`. Env var only; `ascend_config.py` has no such key, so a JSON opt-out is silently ignored |
| `v0.20.2rc1` … `v0.22.1rc1` | key appears in `ascend_config.py` (`enable_balance_scheduling`, `_get_config_value` with the env var as deprecated fallback) |
| HEAD | the gate is `get_ascend_config().scheduler_config.enable_balance_scheduling` (`patch/platform/patch_balance_schedule.py:91`), with a dict/`additional_config` fallback (`:94-99`); documented form `--additional-config '{"enable_balance_scheduling": true}'`, env var deprecated (`patch/__init__.py:62-63`). `ascend_config.py:737` rejects it together with `profiling_chunk_config` |

Docs state the PD caveat at the reported tags: `docs/source/tutorials/models/DeepSeek-R1.md:123` in
`v0.18.0rc1` — "enabling this feature is not recommended in scenarios where PD is separated" (same
sentence in `DeepSeek-V3.1.md:134`, `Kimi-K2.5.md:131`).

Two more facts, both read at `v0.18.0rc1`, that decide whether this switch can be a report's cause:

- **It replaces more than `Scheduler.schedule`.** `patch/platform/patch_balance_schedule.py:705-706` sets
  `vllm.v1.core.sched.scheduler.Scheduler = BalanceScheduler` **and** `EngineCoreProc.run_engine_core =
  run_engine_core`; the copied `schedule()` is a pre-0.18 upstream loop (a per-step
  `skipped_waiting_requests` queue prepended back into `self.waiting`, `:263-303`, including the
  `WAITING_FOR_FSM` skip at `:297-304`), while upstream `v0.18.0` keeps a persistent `self.skipped_waiting`
  and promotes blocked statuses through `_try_promote_blocked_waiting_request`
  (`vllm/v1/core/sched/scheduler.py:557-578`, `:2050-2072`). The copied `run_engine_core` (`:652-705`) also
  lacks that release's `SignalCallback`/`wakeup_engine`, `engine_index=dp_rank`, `data_parallel_index` and
  KV-transfer engine-id rewrite (`vllm/v1/engine/core.py:1029-1110`). A stale copy of a patched entry point
  is a first-class suspect for any *post-patch* behaviour change, structured output included.
- **`--async-scheduling` makes "is the patched class even used?" an import-order question, not a flag
  question.** `SchedulerConfig.get_scheduler_cls()` (`vllm/config/scheduler.py:160-168` at `v0.18.0`)
  returns `AsyncScheduler`, which subclasses whatever `Scheduler` object was bound when
  `vllm/v1/core/sched/async_scheduler.py` was imported (`:6` `from …scheduler import Scheduler`) and
  overrides only `__init__`, `_update_after_schedule` and `_update_request_with_output`. So with the balance
  patch imported first, the async class can inherit the stale loop; if upstream's module was imported first,
  it cannot. Neither "balance scheduling cannot matter here" nor "balance scheduling is the cause" is
  answerable from the argv — the reply gives the A/B (same command with `VLLM_ASCEND_BALANCE_SCHEDULING`
  unset, or `--async-scheduling` off) and keeps the mechanism as `hypothesis` with that falsifier.

## 8. Related documented sparse-C8 caveats (`release_notes.md`)

- `:483` — for SFA models (DeepSeek-V3.2, GLM-5.1, GLM-5.2), DCP must be enabled or disabled on
  **both** prefiller and decoder; asymmetric DCP causes accuracy issues (#14322).
- `:491` — v0.23.0: GLM-5.2 DCP + Sparse Flash Attention C8 is not recommended, incl. performance
  degradation (#14011).
- `:499` — GLM-5.2-W4A8C8 PD serving with `enable_sparse_sfa_c8` shows ~2–3 point GPQA variation
  across runs (#14378).

These are pre-existing, documented accuracy caveats. When a report is "accuracy loss with sparse C8
+ PD", check whether it is one of these before hunting for a new bug.
