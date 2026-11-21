# `Prefix cache hit rate: 0.0%` on a hybrid linear-attention model

Verified at vllm-ascend v0.14.0rc1 (`52d4acfa51fb`) + vllm v0.14.1 (`d7de043d55d1`)
for Qwen3-Next-80B-A3B-Instruct; recheck the greps below at any other pin.

## The answer first

For a hybrid model that does not declare the prefix-caching interface, upstream
vLLM **turns prefix caching off during config resolution**, before the engine
starts. The printed `0.0%` is therefore structural at this version, not a
misconfiguration, not a vllm-ascend defect, and not fixable with
`--enable-prefix-caching`: the disabling branch does not distinguish an explicit
flag from the default. **Scope: vllm v0.14.1 and earlier.** At vllm v0.23.0 that
branch is gone and an explicit flag *is* honored — read *Version flip* at the end
before quoting this verdict at a newer pin.

## The chain (each step source-verified)

| Step | Site | What it does |
|---|---|---|
| default | `vllm/config/cache.py:77` | `enable_prefix_caching: bool = True` — caching is on by default, so a 0 % line needs an explanation |
| hook | `vllm/config/vllm.py:1283-1291` | `if self.model_config.is_hybrid: HybridAttentionMambaModelConfig.verify_and_update_config(self)` runs at config time, before any worker |
| gate | `vllm/model_executor/models/config.py:332-349` (`MambaModelConfig`) | `if enable_prefix_caching:` → `if model_config.supports_mamba_prefix_caching:` enable (and align `mamba_block_size` to `block_size`); **`else:` log `Hybrid or mamba-based model detected without support for prefix caching: disabling.` and set `cache_config.enable_prefix_caching = False`** |
| marker | `vllm/model_executor/models/interfaces.py:760-784`, `vllm/model_executor/models/registry.py:568,594` | the marker is the class attribute `supports_mamba_prefix_caching: ClassVar[Literal[True]] = True` (`SupportsMambaPrefixCaching` protocol); the helper is `getattr(model, "supports_mamba_prefix_caching", False)` and the registry stores the result on the model info |
| the model itself | `vllm/model_executor/models/qwen3_next.py:1210-1212` | `Qwen3NextForCausalLM.__init__`: `assert not cache_config.enable_prefix_caching, ("Qwen3Next currently does not support prefix caching")`; the class declares no `supports_mamba_prefix_caching`, so the config layer disables caching first and the assert then passes — the two lines are consistent, not contradictory |
| downstream | `vllm_ascend/patch/platform/patch_mamba_config.py` (imported unconditionally at `vllm_ascend/patch/platform/__init__.py:21`) | re-implements `HybridAttentionMambaModelConfig.verify_and_update_config` and its first act is `MambaModelConfig.verify_and_update_config(vllm_config)` (`:25`; the override is installed at `:95`), so the Ascend path **inherits** the disabling; it then sizes the attention block from `MambaSpec(block_size=model_config.max_model_len)` page bytes and may pad `mamba_page_size_padded` |
| the log line | `vllm/v1/metrics/loggers.py:234-245` | `GPU KV cache usage` / `Prefix cache hit rate` are appended to every metrics block unconditionally, so a `0.0%` line is not evidence that caching was enabled or even attempted |
| block size | `vllm_ascend/utils.py:1121-1143` (`refresh_block_size`) | vllm-ascend forces `block_size = 128` under prefix caching/chunked prefill **except** for `model_type == "qwen3_next"` (the `TODO(MengqingCao)` guard), so a Qwen3-Next deployment keeps the hybrid-computed block size |

Consequence worth stating to the reporter: with three GDN layers per full-attention
layer, the recurrent state that would have to be reused per block is exactly what
this version cannot cache, so prefix caching cannot be "tuned on" for this model.

## One command on the reporter's host

```bash
grep -i "without support for prefix caching" <serve.log>      # INFO level, printed at startup
grep -n "Prefix cache hit rate" <serve.log> | head -3        # always printed, ignore as evidence
```

The first line appearing is the confirmation. If it does *not* appear, the version
under test no longer takes this branch — then (and only then) re-run the gates:

```bash
grep -n "supports_mamba_prefix_caching" vllm/model_executor/models/qwen3_next.py
grep -rn "def verify_and_update_config" vllm/model_executor/models/config.py
```

## Version flip — at vllm v0.23.0 the flag wins

Verified at vllm **v0.23.0** (`0fc695f`) + vllm-ascend v0.23.0 (`5cb98caaadef`),
the pins of recorded #16979: the disable branch above is **gone**.
`MambaModelConfig.verify_and_update_config`
(`vllm/model_executor/models/config.py:339-375`) now only derives the cache mode —
with prefix caching on, `mamba_cache_mode` defaults to `"all"` when the model
declares `supports_mamba_prefix_caching` and to `"align"` otherwise, and `"all"`
falls back to `"align"` with the warning `Hybrid or mamba-based model detected
without support for prefix caching with Mamba cache 'all' mode: falling back to
'align' mode.` — nothing is disabled. The hybrid marker survives only as the
default for the flag: `is_prefix_caching_supported` (`vllm/config/model.py:1814`)
returns False for `attn_type == "hybrid"` (debug-level sentence "Hybrid models do
not support prefix caching since the feature is still experimental"), and
`vllm/engine/arg_utils.py:2374` reads it as `default_prefix_caching` only, so an
explicit `--enable-prefix-caching` is honored and no line above INFO reports it.

Consequence for a v0.23.0 report: `Prefix cache hit rate: 0.0%` on a hybrid model
is **not** explained by this page any more, and an explicitly enabled prefix cache
on a GDN model is a live, less-validated combination — the block/state geometry it
selects is `align` + chunked prefill (asserted at `config.py:366-369`), which is the
setting `../spec-decode-mtp/acceptance-collapse.md` and
`../accuracy-benchmarks/hybrid-block-size-corruption.md` rank first. Check which
regime the paste is in before reusing the v0.14 reasoning.

## A documented model under PD: the model's own tutorial carries the verdict

Recorded #12093: vllm-ascend v0.22.1rc1 (`5f6faa0cb883`) + vllm v0.22.1
(`0decac0d96c4`), DeepSeek-V4-Flash-w8a8-mtp, A2 4-machine PD (`1*2P` + `1*2D`),
30 720-token prompts at `repeat_rate 0.9`, `--enable-prefix-caching` on **both**
sides, `Prefix cache hit rate: 0.0%`, "开不开MTP都是这样". At that pin the hybrid
gate above does **not** apply (it is vllm ≤ 0.14.x); the model tutorial answers it
in one read, and the recipe diff is the first check:

| what | site at `5f6faa0cb883` |
|---|---|
| the capability | `docs/source/tutorials/models/DeepSeek-V4-Flash.md:949` (§9.1): "Currently, we support 4K prefix cache hit in an experimental manner. You only need to change the value of `--block-size` from 128 to 32 in the service." |
| which side may cache | §5.2.2 A2 PD recipe: P node `--enable-prefix-caching` (`:709`), D node **`--no-enable-prefix-caching`** (`:786`), both `--block-size 128`; the A3 recipe disables it on both (`:451`, `:522`) and §5.1 says "to enable it, remove this option" (`:246`) |
| why granularity matters | per-cache-kind block sizes `vllm_ascend/models/layer/attention/layer.py:33-46` (`_DSV4_BLOCK_SIZES`: `128 -> [mla, swa, c4 state, c128 state] = [128, 128, 8, 32]`), and the compressed groups divide token counts by `compress_ratio` (`vllm_ascend/core/single_type_kv_cache_manager.py:48-49`, `:178-188`), so a hit is quantised by the coarsest group |

The first message owes, in this order: (a) **the metric is per instance** — the D
(`kv_consumer`) side never prefills, so its line is structurally 0 and the recipe
disables caching there; read the **P** node's block; (b) 0 % on P at
`--block-size 128` is the §9.1 note — DSV4 prefix caching at this pin is
experimental and documented for ~4 K prefixes, switched on with
`--block-size 32`, so the recipe copy was run outside its validated setting;
(c) a 27 K-token shared prefix does not rescue (b) — that is an envelope
statement, not a workload one; (d) MTP on/off changing nothing is consistent with
both, so it is not an ablation. The discriminator is one re-run: `--block-size 32`
on the P node only, then read P's `Prefix cache hit rate`.

Do not start from the coordinator/manager or `patch_mamba_config` walk: recorded
#12093 held both tutorial lines and the `CompressAttentionManager` source around
turn 20 and shipped no message text at all. Ask for `D.txt`/`P.txt` equivalents
(the two launch scripts) instead of the mechanism, and quote the §9.1 sentence as
the project's own position at the pin.

## Version scope and recheck conditions

- At v0.14.1 there is no `mamba_cache_mode` field at all (`vllm/config/cache.py`
  carries `mamba_block_size` / `mamba_page_size_padded` / `mamba_cache_dtype`
  only). Later releases introduce `mamba_cache_mode` and a mamba-aware prefix
  cache, and vllm-ascend's later `patch_mamba_config.py` forces
  `mamba_cache_mode="align"` under PD and sets `mamba_block_size = block_size`
  only when prefix caching is on — see
  `../spec-decode-mtp/acceptance-collapse.md` and
  `../pd-disaggregation/hybrid-state-transfer.md`. Those line numbers are
  v0.20.2rc1/v0.22.1rc1; do not carry the "disabled" verdict across versions
  without the greps above.
- The same shape can appear for a *different* owner: an `ascend_scheduler`
  deployment where prefix cache + chunked prefill was reported not to work
  (precedent #2943). Ask which scheduler is installed (`platform.py` sets
  `scheduler_cls` only for the dynamic-batch and recompute variants at v0.14.0rc1)
  before attributing the 0 % to the hybrid gate.
- A genuinely shared-prefix workload on a *dense* model will show a non-zero rate;
  if the reporter's workload reuses one long system prompt and the rate is still
  0 %, that is the switch gate, not the workload.
