# "Unknown vLLM environment variable detected: VLLM_ASCEND_…"

Scope: a startup log line, usually one per variable, of the form
`(APIServer pid=N) WARNING <ts> [envs.py:1717] Unknown vLLM environment variable
detected: VLLM_ASCEND_ENABLE_FLASHCOMM1`, while the engine then starts normally.
Reporters file it as 误报 / "false alarm" and often add "the export is commented
out in my script". It is a registry/ownership question about one log line, not a
FlashComm, quantization or card question.

Verified against vllm **v0.18.0** (`bcf2be9612`) and vllm-ascend **v0.18.0rc1**
(`99e1ea0fe6`), the pair in #9986. Line numbers move between releases (v0.17.0rc1
emitted the same warning at `[envs.py:1710]`); the mechanism does not.

## Answer rule

The warning is **warn-only and cosmetic**, and it proves exactly one thing: that
name was in *that process's* environment. Deliver, in this order: (1) not the
cause of any failure, `fail_on_environ_validation` defaults `False`; (2) the one
command that shows the variable — `env | grep -i vllm_ascend`, or for a running
server `tr '\0' '\n' < /proc/<pid>/environ | grep -i vllm_ascend`; (3) the fix for
the reporter (`unset …` in a fresh shell, or stop injecting it) *and* the separate
registry gap in the project (below). Two or three calls, then answer.

## The mechanism at v0.18.0 / v0.18.0rc1

| # | Site | Fact |
|---|---|---|
| 1 | `vllm/envs.py:1711-1717` (vllm v0.18.0) | `validate_environ(hard_fail)`: for every `os.environ` key, `if env.startswith("VLLM_") and env not in environment_variables` → `raise ValueError(...)` when `hard_fail`, else `logger.warning("Unknown vLLM environment variable detected: %s", env)` |
| 2 | `vllm/engine/arg_utils.py:1488-1490` | the call site is `create_engine_config`: `current_platform.pre_register_and_update()` then `envs.validate_environ(self.fail_on_environ_validation)` — it runs in whichever process builds the engine config, before model loading; read the log prefix to name it (`APIServer` in every recorded paste) |
| 3 | `vllm/engine/arg_utils.py:617` | `fail_on_environ_validation: bool = False` — the flag is off unless the user passes it, so the line can never be the failure by itself |
| 4 | `vllm_ascend/envs.py:30,74,117-123` | vllm-ascend keeps its **own** registry `env_variables` (`"VLLM_ASCEND_ENABLE_FLASHCOMM1"` at `:74`), reached through `__getattr__`/`__dir__`. At this ref nothing merges it into `vllm.envs.environment_variables` (only `collect_env.py:26,311` reads that dict), so **every** `VLLM_ASCEND_*` name is unknown to upstream by construction |
| 5 | `vllm_ascend/platform.py:134-154` | the merge point already exists and already runs one line before the check: `AscendPlatform.pre_register_and_update` registers the `ascend` quantization choice for the parser. Registering `env_variables` there is the downstream fix — the gap is vllm-ascend's, not upstream's |
| 6 | `vllm_ascend/batch_invariant.py:79-81` | vllm-ascend also *writes* `VLLM_ASCEND_ENABLE_NZ=0` and `VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE=0` itself, which warn for the same reason — so a line can appear with the reporter having exported nothing (batch-invariant runs) |
| 7 | `collect_env.py:304-316` (vllm-ascend) | `get_env_vars()` prints a variable only if it is in upstream's `environment_variables` **or** starts with `TORCH`/`PYTORCH`/`ASCEND_`/`ATB_`. `VLLM_ASCEND_*` therefore never appears in the pasted `collect_env.py` block — that is why the reporter cannot see it, not evidence they did not set it |

## Procedure

1. Name the emitter and the process from the log prefix (`(APIServer pid=…)`,
   `(Worker pid=…)`); state that the variable really was exported in it.
2. Rule out the flag as a cause: `fail_on_environ_validation` default (row 3).
   If the report also asserts "FlashComm is on", check the flag's documented
   scope before believing it: the pinned doc says the optimisation is MoE-only
   with `tp_size > 1` (`docs/source/tutorials/features/long_sequence_context_parallel_single_node.md:129`
   at v0.18.0rc1), so on a single-card dense run the flag is a no-op and the
   warning is not evidence that FlashComm engaged.
3. Give the cheap reporter action: a fresh shell, or `unset <name>` before
   `vllm serve` — commenting an `export` in a script does not unset a variable an
   earlier command already exported in the same shell/container. Other injection
   paths worth one line: `docker run -e …`, a scheduler/launcher env block, an
   image `.bashrc`.
4. State the ownership split in one sentence: benign for the user; the fix is
   `AscendPlatform.pre_register_and_update` merging `vllm_ascend.envs.env_variables`
   into `vllm.envs.environment_variables` (row 5), or upstream accepting downstream
   names.

## Precedents

- **#7876** v0.17.0rc1, Kimi-K2.5-w4a8, TP8: two lines at `[envs.py:1710]`
  (`VLLM_ASCEND_BALANCE_SCHEDULING`, `VLLM_ASCEND_ENABLE_MLAPO`); the engine then
  ran on and failed later in an unrelated multimodal prompt-replacement
  `AssertionError`. The warning was never the fault.
- **#9986** v0.18.0rc1 image, single 910 card, Qwen3-0.6B dense TP=1, FLASHCOMM1
  export commented out: the only evidence was a screenshot; its OCR line reads
  `(APIServer pid=231) … [envs.py:1717] Unknown vLLM environment variable
  detected: …FLASHCOMM1` a few seconds after the `non-default args` dump.

## Do not claim

- which shell, script or image exported the variable — that is the reporter's
  machine and the snapshot cannot show it;
- that upstream raises here, or that the run will fail because of it;
- that "vllm-ascend sets this variable itself" for `VLLM_ASCEND_ENABLE_FLASHCOMM1`:
  at both refs the tree only *reads* it (`vllm_ascend/envs.py:74`;
  `vllm_ascend/utils.py:780` at v0.18.0rc1, `:806-809` at v0.19.1rc1); the
  self-written names are the batch-invariant pair in row 6.

## Recheck

`rg -n "def validate_environ" -A7 <vllm>/vllm/envs.py` and
`rg -n "environment_variables" <vllm-ascend>/vllm_ascend/` for the reported pair;
if a release merges the registries (or adds `VLLM_ASCEND_*` to upstream), the
warning disappears and this page is stale.
