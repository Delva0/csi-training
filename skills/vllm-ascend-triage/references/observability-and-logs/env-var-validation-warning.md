# "Unknown vLLM environment variable detected: VLLM_ASCEND_…"

Scope: a startup log line, usually one per variable, of the form
`(APIServer pid=N) WARNING <ts> [envs.py:1717] Unknown vLLM environment variable
detected: VLLM_ASCEND_ENABLE_FLASHCOMM1`, while the engine then starts normally.
Reporters file it as 误报 / "false alarm" and often add "the export is commented
out in my script" — or the inverse, "the log calls it unknown but the variable
does work and fixes my error". It is a registry/ownership question about one log
line, not a FlashComm, quantization or card question.

Verified against vllm **v0.18.0** (`bcf2be9612`) / vllm-ascend **v0.18.0rc1**
(`99e1ea0fe6`) — the pair in #9986 — and re-verified against vllm **v0.22.1**
(`0decac0d96c4`) / vllm-ascend **v0.22.1rc1** (`5f6faa0cb883`), the pair in
#11624. Line numbers move between releases (v0.17.0rc1 printed the warning from
`[envs.py:1710]`, v0.18.0 from `:1717`, v0.22.1 from `:2051-2057`); the mechanism
does not.

## Answer rule

The warning is **warn-only and cosmetic**, and it proves exactly one thing: that
name was in *that process's* environment. It says nothing about whether
vllm-ascend honours the name, and a report usually carries both halves at once
("the log calls it unknown, but setting it fixes my error"). So split on one
question — **does the name still do something?** — and deliver:

- **Inert, stale or typo'd name**: not the cause of any failure
  (`fail_on_environ_validation` defaults `False`). Show it with
  `env | grep -i vllm_ascend` (running server:
  `tr '\0' '\n' < /proc/<pid>/environ | grep -i vllm_ascend`), then `unset …` in a
  fresh shell or stop injecting it.
- **A live knob being retired by migration**: one of the ten names in the pin's own
  *Migration Guide* (`docs/source/user_guide/configuration/additional_config.md:6-24`
  at v0.22.1rc1 — `VLLM_ASCEND_BALANCE_SCHEDULING`, `…ENABLE_FLASHCOMM1`,
  `…ENABLE_MATMUL_ALLREDUCE`, `…FLASHCOMM2_PARALLEL_SIZE`, `MSMONITOR_USE_DAEMON`,
  `…ENABLE_MLAPO`, `…ENABLE_NZ`, `…ENABLE_CONTEXT_PARALLEL`,
  `…ENABLE_FUSED_MC2`, `…FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK`). Here the line is
  a side effect of the migration and the answer is the supported
  `--additional-config` key — **not** "unset it", because the reporter is using the
  flag deliberately. Quote the key's own row (`:90-101`, each says "can also be
  configured via `VLLM_ASCEND_<NAME>` environment variable (deprecated)") and keep
  their value. #11624, maintainer-confirmed and accepted by the reporter:
  `VLLM_ASCEND_ENABLE_NZ=0` both works and is deprecated →
  `--additional-config '{"weight_nz_mode": 0}'` (0 = off, 1 = NZ for quant weights
  only, the default, 2 = BF16/FP16 also; `ascend_config.py:221-226,289-300`).

Then the project-side registry gap (below) in one sentence. Two or three calls,
then answer.

## The mechanism at v0.18.0 / v0.18.0rc1

| # | Site | Fact |
|---|---|---|
| 1 | `vllm/envs.py:1711-1717` (vllm v0.18.0) | `validate_environ(hard_fail)`: for every `os.environ` key, `if env.startswith("VLLM_") and env not in environment_variables` → `raise ValueError(...)` when `hard_fail`, else `logger.warning("Unknown vLLM environment variable detected: %s", env)` |
| 2 | `vllm/engine/arg_utils.py:1488-1490` | the call site is `create_engine_config`: `current_platform.pre_register_and_update()` then `envs.validate_environ(self.fail_on_environ_validation)` — it runs in whichever process builds the engine config, before model loading; read the log prefix to name it (`APIServer` in every recorded paste) |
| 3 | `vllm/engine/arg_utils.py:617` | `fail_on_environ_validation: bool = False` — the flag is off unless the user passes it, so the line can never be the failure by itself |
| 4 | `vllm_ascend/envs.py:30,74,117-123` | vllm-ascend keeps its **own** registry `env_variables` (`"VLLM_ASCEND_ENABLE_FLASHCOMM1"` at `:74`), reached through `__getattr__`/`__dir__`. At this ref nothing merges it into `vllm.envs.environment_variables` (only `collect_env.py:26,311` reads that dict), so **every** `VLLM_ASCEND_*` name is unknown to upstream by construction |
| 5 | `vllm_ascend/platform.py:134-154` | the merge point already exists and already runs one line before the check: `AscendPlatform.pre_register_and_update` registers the `ascend` quantization choice for the parser. Registering `env_variables` there is the downstream fix — the gap is vllm-ascend's, not upstream's |
| 6 | `vllm_ascend/batch_invariant.py:79-81` (v0.18.0rc1 only) | vllm-ascend also *writes* `VLLM_ASCEND_ENABLE_NZ=0` and `VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE=0` itself, which warn for the same reason — so at that ref a line can appear with the reporter having exported nothing (batch-invariant runs). **Re-grep per pin**: at v0.22.1rc1 that file contains no `VLLM_ASCEND_*` write |
| 7 | `collect_env.py:304-316` (vllm-ascend) | `get_env_vars()` prints a variable only if it is in upstream's `environment_variables` **or** starts with `TORCH`/`PYTORCH`/`ASCEND_`/`ATB_`. `VLLM_ASCEND_*` therefore never appears in the pasted `collect_env.py` block — that is why the reporter cannot see it, not evidence they did not set it |

## Version drift rechecked at v0.22.1 / v0.22.1rc1 (#11624)

- `vllm/envs.py:2051-2057` — the same three lines; caller
  `vllm/engine/arg_utils.py:1711` in `create_engine_config`, flag default `:714`.
- Registry: `vllm_ascend/envs.py:30` with `__getattr__`/`__dir__` at `:120-126`;
  `VLLM_ASCEND_ENABLE_NZ` registered at `:92` (default `1`).
  `rg -n "environment_variables" vllm_ascend/` still returns nothing, so **no
  merge** at this ref either, and `AscendPlatform.pre_register_and_update`
  (`platform.py:182-202`) still only adds the `ascend` quantization choice (it now
  ends with `config_deprecated_logging()`, which only installs the one-line warning
  format — not a list of deprecated names).
- The retired-knob half is new at this ref: `ascend_config.py:289-300`
  `_get_config_value(additional_config, "<key>", "VLLM_ASCEND_<NAME>", env_value)`
  prefers the `additional_config` key, else logs "…falls back to environment
  variable … Please use additional_config.<key> instead, because VLLM_ASCEND_<NAME>
  will be removed in the next release". So honouring the variable and deprecating
  it are the same code path — which is exactly the "log says unknown, flag works"
  pair in the report.
- `collect_env.py:304-316` unchanged.

## Procedure

1. Name the emitter and the process from the log prefix (`(APIServer pid=…)`,
   `(Worker pid=…)`); state that the variable really was exported in it.
2. Rule out the flag as a cause: `fail_on_environ_validation` default (row 3).
   If the report also asserts "FlashComm is on", check the flag's documented
   scope before believing it: the pinned doc says the optimisation is MoE-only
   with `tp_size > 1` (`docs/source/tutorials/features/long_sequence_context_parallel_single_node.md:129`
   at v0.18.0rc1), so on a single-card dense run the flag is a no-op and the
   warning is not evidence that FlashComm engaged.
3. Decide the branch: check the reported name against the pin's Migration Guide
   table (`rg -n "Migration Guide" -A22
   <vllm-ascend>/docs/source/user_guide/configuration/additional_config.md`).
   - not in it → cheap reporter action: a fresh shell, or `unset <name>` before
     `vllm serve`. Commenting an `export` in a script does not unset a variable an
     earlier command already exported in the same shell/container; other injection
     paths worth one line: `docker run -e …`, a scheduler/launcher env block, an
     image `.bashrc`.
   - in it → name the `additional_config` key, its default and the values, and
     keep the reporter's current value (`{"weight_nz_mode": 0}` for the NZ case).
4. State the ownership split in one sentence: benign for the user; the registry fix
   is `AscendPlatform.pre_register_and_update` merging
   `vllm_ascend.envs.env_variables` into `vllm.envs.environment_variables`
   (row 5), or upstream accepting downstream names. Deprecating the env name is a
   separate, deliberate removal — do not conflate the two.

## Precedents

- **#7876** v0.17.0rc1, Kimi-K2.5-w4a8, TP8: two lines at `[envs.py:1710]`
  (`VLLM_ASCEND_BALANCE_SCHEDULING`, `VLLM_ASCEND_ENABLE_MLAPO`); the engine then
  ran on and failed later in an unrelated multimodal prompt-replacement
  `AssertionError`. The warning was never the fault.
- **#9986** v0.18.0rc1 image, single 910 card, Qwen3-0.6B dense TP=1, FLASHCOMM1
  export commented out: the only evidence was a screenshot; its OCR line reads
  `(APIServer pid=231) … [envs.py:1717] Unknown vLLM environment variable
  detected: …FLASHCOMM1` a few seconds after the `non-default args` dump.
- **#11624** v0.22.1rc1 + vllm 0.22.1, 2×910B3, CANN 9.0.0, `[Usage]` title: the
  reporter exports `VLLM_ASCEND_ENABLE_NZ=0` to work around a runtime error,
  gets the warning, and reports the contradiction. Both halves are true and the
  maintainer confirmed both: upstream prints the line and it can be ignored; the
  NZ variable is being phased out, so the supported form is
  `--additional-config '{"weight_nz_mode": 0}'`. The reporter accepted ("Got it,
  thanks!"). The recorded run reached the same two halves from the pinned tree in
  ~10 tool calls by re-deriving rows 1-5 plus the migration table — the branch in
  the answer rule is what the maintainer's reply consisted of.

## Do not claim

- which shell, script or image exported the variable — that is the reporter's
  machine and the snapshot cannot show it;
- that upstream raises here, or that the run will fail because of it;
- that "vllm-ascend sets this variable itself" for `VLLM_ASCEND_ENABLE_FLASHCOMM1`:
  at both refs the tree only *reads* it (`vllm_ascend/envs.py:74`;
  `vllm_ascend/utils.py:780` at v0.18.0rc1, `:806-809` at v0.19.1rc1); the
  self-written names were the batch-invariant pair in row 6, and even that is
  version-bounded;
- that the flag must be unset when the pin's migration table still honours it — the
  supported key is the deliverable, and the old name keeps working "during the
  transition period" (that doc says so explicitly).

## Recheck

`rg -n "def validate_environ" -A7 <vllm>/vllm/envs.py` and
`rg -n "environment_variables" <vllm-ascend>/vllm_ascend/` for the reported pair;
`rg -n "Migration Guide" -A22
<vllm-ascend>/docs/source/user_guide/configuration/additional_config.md` for what
is being retired at that ref, and `rg -n VLLM_ASCEND_
<vllm-ascend>/vllm_ascend/batch_invariant.py` before reusing row 6. If a release
merges the registries (or adds `VLLM_ASCEND_*` to upstream), the warning
disappears and this page is stale.
