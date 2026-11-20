# `--additional-config` rejected by the strict AscendConfig schema

Signature family — the text is a **pydantic `ValidationError` naming one config
key**, never an op, tensor, collective or model file:

```text
fuse_muls_add
        Unexpected keyword argument [type=unexpected_keyword_argument, input_value=True, input_type=bool]
For further information visit https://example.invalid/2.13/v/unexpected_keyword_argument
[ERROR] ... ERR99999 UNKNOWN application exception
```

Tells: no `WorkerProc` line, no `Starting to load model`, and the key is one the
reporter typed into `--additional-config '{...}'` (or the launcher's
`additional_config:` block), usually copied from a tutorial serve command or from
an older issue's launch line. `unexpected_keyword_argument` with
`input_value=True, input_type=bool` is the dataclass form of "this key is not a
field"; the pydantic version in the URL moves with the image, so match the error
*type*, not the number.

## Mechanism (verified at vllm-ascend v0.27.1rc1 `3b31886`, vllm v0.27.1 `6e448d0`)

- `--additional-config` becomes `VllmConfig.additional_config`; vllm-ascend
  validates it in `init_ascend_config()` while the platform resolves config:
  `AscendPlatform.check_and_update_config` (`vllm_ascend/platform.py:442`) →
  `init_ascend_config(vllm_config)` (`:475`). That is **before** workers, weights,
  KV cache and any device work — nothing about the reporter's hardware matters.
- `AscendConfig` is a pydantic dataclass (vllm-ascend's `@config`,
  `vllm_ascend/config_utils.py:43` `merged_config = ConfigDict(extra="forbid")`)
  with `model_config = ConfigDict(extra="forbid")` of its own
  (`vllm_ascend/ascend_config.py:383`). Any key that is not a top-level
  `AscendConfig` field is fatal.
- The factory strips only its own injected/derived inputs before validation
  (`_NON_USER_INPUT_KEYS`, `ascend_config.py:1437-1467`: `refresh`,
  `enable_flashcomm1`, `scheduler_config`, `sparse_kv_offload_config`,
  `dump_config*`, `enable_sp_by_pass`/`pd_*`/`num_head_replica`, the legacy
  scheduler keys), then computes `unknown_keys` (`:1472`) and constructs
  `AscendConfig(**kwargs)` (`:1484`).
- **Escape hatch, not a fix:** if `vllm_omni` is importable, unknown keys are
  dropped with a warning instead of raising (`:1473-1481`, the fix for #15418). On
  a plain vllm-ascend host they are fatal, so do not tell the reporter to install
  vllm-omni.
- Interface drift: the strict schema arrived with the `@config` migration
  (PR #13838, "Migrate AscendConfig to @config pydantic dataclass"); before it the
  same JSON was accepted (flat `fuse_muls_add` in #7613, v0.17/v0.18 era), and a
  stale key was only a silent no-op — the behaviour
  `../runtime-memory/hbm-growth-and-oom.md` Step 4 records for v0.23.0rc1. #15418
  reports the identical `unexpected_keyword_argument` text for a downstream-owned
  key at vLLM 0.27.1 — use it as the precedent for the error class, not as the
  fix.

## Procedure

1. Grep the key in the pinned tree: `rg -n '<key>' vllm_ascend/ascend_config.py`.
   Three outcomes:
   - a field of a nested `@config` class (`AscendCompilationConfig`,
     `SchedulerConfig`, `SparseKVOffloadConfig`, `XLiteGraphConfig`, …) → it must
     be **nested**, under the key that names that class on `AscendConfig`
     (e.g. `{"ascend_compilation_config": {"fuse_muls_add": true}}`);
   - a top-level `AscendConfig` field → the flat form is right and the reporter is
     on an older/newer image, so check the version before theorising;
   - absent everywhere → it was removed; check `_NON_USER_INPUT_KEYS` (a stripped
     name only warns) and the docs table before calling it a typo.
2. Read the field's **default**. If the default already equals the value the
   reporter wants, the cheapest action is to delete the key — the reported
   `input_value=True` then changes nothing (`fuse_muls_add: bool = True`,
   `ascend_config.py:72`).
3. Check the docs the reporter copied: `docs/source/user_guide/configuration/additional_config.md`
   lists the nesting (`:52` table row `ascend_compilation_config`, `:101,109` the
   nested fields), while `docs/source/tutorials/models/<model>.md` serve commands
   may still print the old flat form. A tutorial page whose header says the
   scripts are based on an older release truncates its own authority — the answer
   is "doc bug at this ref" plus the corrected JSON.
4. Answer: verdict (config-nesting/removed-key, not a defect or the card), the
   one-line corrected `--additional-config`, the `path:line` proof from step 1,
   the default that makes deletion safe, then the warnings-only leftovers. Stop —
   this lane needs no plog, no `npu-smi`, no HCCL theory.
5. One history grep (`scripts/history-titles.sh '<key>'`) tells you whether the key
   was *ever* top-level; the count of earlier issues using the flat form is the
   doc-drift argument, not a second investigation.

## Worked case #18130 — GLM-5.3-w8a8c8, 4×Atlas 800I A2, v0.27.1rc1

| Fact | Evidence (pin v0.27.1rc1 `3b31886`) |
|---|---|
| `fuse_muls_add` is a **nested** field with default `True` | `ascend_config.py:72` (`AscendCompilationConfig`), docstring example `:280` under `"ascend_compilation_config"` |
| Top-level keys are forbidden | `ascend_config.py:383`, `config_utils.py:43` |
| Not in the strip list, so it reaches pydantic verbatim | `ascend_config.py:1437-1472,1484` |
| Raised at platform config check, before any worker | `platform.py:442,475` |
| Consumer of the switch (explains "is it even on?") | `compilation/graph_fusion_pass_manager.py:66-69` — `MulsAddFusionPass` only when `fuse_muls_add and not is_310p()` |
| Docs table places it under `ascend_compilation_config` | `docs/source/user_guide/configuration/additional_config.md:52,101,109` |
| **The tutorial command prints it flat** | `docs/source/tutorials/models/GLM5.3.md:310` and `:369`; `:15` says "All the scripts below is based on **v0.23.0**, so some params are not supported in main code" |
| Nightly yaml uses the nested form | `tests/e2e/weekly/multi_node/internal_dp/config/GLM5.2-W8A8C8-A3_*.yaml:47` |
| The reporter's follow-up confirmed the source URL | later comment on #18130 quotes the v0.27.1rc1 tutorial page with the same flat JSON |

Same line, different severity — say which is fatal and which only warns:
`enable_balance_scheduling` top-level still resolves through
`SchedulerConfig.from_additional_config` with a deprecation warning
(`ascend_config.py:376-380,1426`; prefer `{"scheduler_config":
{"enable_balance_scheduling": true}}`), and `enable_flashcomm1` is stripped with a
warning (`:1402-1406,1442`).

Traps:

- Do **not** move the key into `--compilation-config '{"pass_config": {...}}'`:
  upstream `PassConfig` uses the same `extra="forbid"`
  (`vllm/config/utils.py:69`) and has no such field (`vllm/config/compilation.py:107`).
- This is not the drift row: a Python `TypeError: ... got an unexpected keyword
  argument` inside a `vllm_ascend/{ops,patch,models}` frame is patch drift; a
  pydantic `unexpected_keyword_argument` over `additional_config` is a schema
  question about a CLI value.
- The paste may show only the last of several unknown keys ("your paste cut the
  `N validation errors for …` header"): ask for the whole `--additional-config`
  line before promising that one edit is enough.
- Whether the flat form ever worked on the reporter's exact image is not provable
  from a one-commit snapshot; cite the older issues as precedent and say so.
