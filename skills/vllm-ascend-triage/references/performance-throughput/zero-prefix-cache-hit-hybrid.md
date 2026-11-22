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

**Not this page when the rate is nonzero.** A rate that is small, or that steps as the
prompt length changes, is the lookup arithmetic, not a gate: go to
`prefix-cache-hit-rate-at-block-boundary.md` (`hit = B·floor((L−1)/B)`, one whole block
lost at exact-multiple lengths). This page owns only the flat `0.0 %`.

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
| block size | `vllm_ascend/utils.py:1121-1143` (`refresh_block_size`) | vllm-ascend forces `block_size = 128` under prefix caching/chunked prefill **except** for `model_type == "qwen3_next"` (the `TODO(MengqingCao)` guard), so a Qwen3-Next deployment keeps the hybrid-computed block size. **That is v0.14-era and has a second exception at newer pins:** `refresh_block_size` moved to `:1240` and now returns early for `model_type == "deepseek_v4"`, keeping `--block-size` in `{32, 64, 128}` (default 32) whatever prefix caching does (`utils.py:1257-1266` at v0.26.0rc1 `f2f74a16`) — so a DSV4 serve's `--block-size 32` is *not* silently rewritten to 128, and a 32-block recipe is not the explanation for a hit-rate change there (that is `spec-on-prefill-ttft-regression.md`) |

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
`../accuracy-benchmarks/hybrid-block-size-corruption.md` rank first. On a
**310P/Atlas 300I** box **with MTP** the corruption shape of that same pair is owned by
`../accuracy-benchmarks/mtp-prefix-cache-310p.md` instead (both single-feature-off arms
are recorded there at this tag), so take that page rather than the two generic ones —
recorded #16979's own paste is that accuracy report, and this page cites #16979 only for
the v0.23.0 flag/default check. Check which
regime the paste is in before reusing the v0.14 reasoning.

## A documented model whose zero-hit was fixed in a later release: DSV4-Flash

**A flat `0.0 %` on a 64 K prompt is not what a granularity note predicts.** The DSV4 tutorial
says *"we support 4K prefix cache hit in an experimental manner"* — that bounds how much of a
long prefix is *reusable*, and a 4 K quantum over 64 K still reuses most of it, so the documented
limit gives a **reduced, nonzero** rate. An exact `0.0 %` means the lookup returned nothing, i.e.
a code path, not an envelope. Two recorded cases at the same pin show why the version rung comes
first: #12093 (below) and **#12041** — same `5f6faa0cb883`, **colocated** TP8 (`--data-parallel-size
1 --tensor-parallel-size 8 --enable-expert-parallel`, `FULL_DECODE_ONLY` + MTP n=1 +
`--async-scheduling`, `enable_dsa_cp: true`, `multistream_overlap_shared_expert: true`), 64 K
prompts at `repeat_rate 0.9`, `--block-size 128`, `Prefix cache hit rate 0` and TTFT doubling
6799.6 → 14955.1 ms against its own v0.20.2rc1 baseline. Its reporter's resolution (2026-07-30) was
**「v0.23.0测试已修复」**, and the next release's notes name the repair:
`docs/source/user_guide/release_notes.md:23` at v0.23.0rc1 — *"Improved the DeepSeek V4
prefix-cache hit rate. [#11107]"*.

So the repair stream is the discriminator at `≤ v0.22.1rc1`, and quoting it is cheap:

| release | DSV4 prefix-cache work in its own notes |
|---|---|
| v0.20.2rc1 | introduced DSV4 end-to-end (the reporter's 0.9 baseline predates the compressed-KV rewrite) |
| v0.21.0rc1 | `Added DeepSeek-V4 compressor block size [32,64,128] support to improve automatic prefix cache hit rate` (#10354), plus #9903 and #10297 "compress attention groups prefix caching hit" fixes |
| v0.23.0rc1 | `Improved the DeepSeek V4 prefix-cache hit rate` (**#11107**) — reporter-confirmed fix for #12041 |

When the pin's own notes show the same subsystem repaired in each of the last two releases, a
current-release regression in it is pre-fix by construction: **say "the upgrade is the first test,
#11107 is the PR to watch" before asking for a knob re-run.** At `--block-size 128` the colocated
case also matched the tutorial's single-machine recipe verbatim except one line —
`--no-enable-prefix-caching` (`DeepSeek-V4-Flash.md:175`) replaced by `--enable-prefix-caching` —
so the recipe-diff answer still holds for "is this a validated recipe" (no), just not for "is this
the documented limit" (also no).

Two non-causes at that pin, worth one line each so they are not re-derived: the only in-tree
force-disable on the DSV4 attention layer is backend-gated (`vllm_ascend/models/layer/attention/
layer.py:115-119`: `TRITON_MLA` / `FLASHINFER` ⇒ `enable_prefix_caching = False`) and never fires
under `AscendDSABackend`, so caching was not silently switched off; and
`VLLM_ASCEND_APPLY_DSV4_PATCH` has **no reference anywhere in the v0.22.1rc1 tree**, so that env
var in the reporter's export block is inert — neither a lever nor an explanation. The doubled TTFT
is the same symptom (a full re-prefill per request), not a second regression.

### The PD shape: the model's own tutorial also decides which side may cache

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
disables caching there; read the **P** node's block; (b) **the version rung** — at
`≤ v0.22.1rc1` this subsystem is being repaired release by release (table above,
`#11107` the last), and a flat 0 on a long shared prefix is a lookup returning
nothing rather than the ~4 K envelope, so name the PR and one image to test;
(c) then `--block-size 32` on the P node only — the §9.1 switch, and the way to get
hits on *this* pin today (a 27 K-token shared prefix does not rescue it; that is an
envelope statement, not a workload one); (d) MTP on/off changing nothing is
consistent with both, so it is not an ablation. If the reporter can only run one
row, run (b); if they must stay on the pin, run (c) and read P's
`Prefix cache hit rate`.

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
