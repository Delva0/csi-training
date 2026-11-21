# Weight ND→NZ layout: the capability answer, and the `is_310p()` polarity trap

## Scope

"910B3 是否支持权重 nd 到 nz 的转换，提高 matmul 的性能？0.11.0 的代码好像只支持 310P" —
a **capability** question with no log, usually because the reporter read one guard and
generalised it. Also "`weight_nz_mode` 怎么设", "有没有开关把 bf16 权重转成 NZ",
"量化权重是不是已经转 NZ 了". Answer in `index.md` part 1: the verdict first, then the
per-path inventory, then the one probe. The crash variants are
`../startup-failures/nz-weight-format-refusal.md` (shape: `Not support mat2 n = 1 or k = 1
when format is FRACTAL_NZ`) and `../pooling-embedding/fp32-head-nz-format-crash.md`
(dtype). Verified at **vllm-ascend v0.11.0rc0 `00ba071`** (#3859) and **v0.20.2rc1
`367b8e6`** (the crash page).

## Rule 1 — `is_310p()` reads both ways; classify the polarity before answering

- `if not is_310p(): <NZ cast>` — the cast runs on 910B/A2/A3 and is **skipped** on 310P.
  A reporter who reads this as "只支持 310P" has inverted it, and that inversion is the
  whole question in this shape.
- `if is_310p(): <NZ cast>` — genuinely 310P-only.

Both spellings sit in the same tree, sometimes in one module family, so start with
`rg -n 'is_310p\(\)' vllm_ascend/` and read the polarity per site. Answering with a
hardware claim instead ("910 不支持 NZ") is wrong even when the guarded cast is
310P-only: the hardware supports the layout, and the project's own default ships NZ
weights on 910 for every quantized checkpoint.

## Rule 2 — the inventory that answers #3859 (v0.11.0rc0 `00ba071`)

| path:line | guard | what converts |
|---|---|---|
| `quantization/w8a8.py:139-140` | none | W8A8-static **Linear** weight → `FRACTAL_NZ` |
| `quantization/w8a8_dynamic.py:104-105` | none (`:103` "for higher inference speed") | W8A8-dynamic Linear → NZ |
| `quantization/w8a8.py:323-327` | `not is_310p()` | W8A8-static **MoE** `w13_weight` / `w2_weight` → NZ on 910 |
| `quantization/w8a8_dynamic.py:268-269` | none | W8A8-dynamic MoE → NZ |
| `quantization/w4a8_dynamic.py:396-399` | none | W4A8-dynamic MoE → NZ |
| `ops/common_fused_moe.py:128-132`, `ops/fused_moe.py:81-85` | `not is_310p()` | **unquantized** MoE expert weights → NZ on 910 |
| `worker/model_runner_v1.py:2633-2643` | `is_310p()` | the **only** load-time cast for **unquantized dense** Linear (`MergedColumnParallelLinear` / `QKVParallelLinear` / `RowParallelLinear`), via `_convert_torch_format` (`:2672-2674` → `npu_format_cast(tensor, ACL_FORMAT)`) |
| `worker/model_runner_v1.py:138, 140-144` | `is_310p()` | `torch.npu.config.allow_internal_format = True` for everyone, but `ACL_FORMAT` is `FRACTAL_NZ` **only** on 310P and `FRACTAL_ND` otherwise — so the same `_convert_torch_format` is a no-op for dense weights on 910 (`:2780-2787` also uses it for the KV/rope caches on 310P) |
| `ops/linear.py` | — | has no `process_weights_after_loading`: `AscendLinearBase` keeps upstream `UnquantizedLinearMethod`, which is why bf16 dense weights have no NZ hook at this ref |

So the verdict for that report: **on 910B3 the quantized paths (Linear and MoE) and
unquantized MoE already run NZ by default — there is nothing to enable**; "只支持 310P"
is true of exactly one path, the unquantized **dense** Linear cast, which is the
`if is_310p():` line the reporter read. Say which path their model uses
(`--quantization ascend` ⇒ quantized ⇒ already NZ) before discussing a patch.

## Rule 3 — the switch is a version fact; grep it before naming it

At v0.11.0rc0 there is **no** `weight_nz_mode` and no `VLLM_ASCEND_ENABLE_NZ`:
`vllm_ascend/envs.py` has no `nz` entry and `ascend_config.py:121` carries only
`enable_kv_nz` (the torchair MLA **KV** layout, `:153-155` errors without graph mode) —
the layout is decided inline in the cast sites above. **Between those two refs the env
var arrives before the config key**: at v0.19.1rc1 (`da421af`) the gate functions already
exist — `utils.py:157-175` `_should_trans_nz` (`310P` always; non-310P follows
`VLLM_ASCEND_ENABLE_NZ`; bf16/fp16 only at `== 2`; fp32 never) and `maybe_trans_nz`
(consumers at `ops/linear.py:79`, `ops/fused_moe/fused_moe.py:106-107`,
`_310p/quantization/methods/*`), with `envs.py:87-91`
`VLLM_ASCEND_ENABLE_NZ` default `1` — while **no `weight_nz_mode` key exists**: `rg -n
'weight_nz' vllm_ascend/ascend_config.py` is empty. At v0.20.2rc1 the three-valued
switch exists: `weight_nz_mode` `0` off / `1` quantized-only (default) / `2` bf16+fp16
too (`ascend_config.py:216-223`, env `VLLM_ASCEND_ENABLE_NZ`), and mode 2 is what makes a
narrow bf16 weight crash (`../startup-failures/nz-weight-format-refusal.md`).
**A documented key can also be a key nothing reads.** `enable_weight_nz_layout` — the
spelling the project's own page uses (`docs/source/user_guide/feature_guide/large_scale_ep.md:168,234,443`)
and several nightly yamls pass — is read by **no** code at v0.19.1rc1
(`rg -n 'enable_weight_nz_layout' vllm_ascend/` empty; it survives only in docs, tests
and the zh_CN `.po`), because `additional_config` is consumed key by key with
`.get(...)` (`ascend_config.py:35-195`) and an unknown key is silently ignored rather
than rejected. So a launch carrying it at that pin behaves exactly like a launch without
it: not a cause, and not the feature the reporter thinks they enabled. Therefore:
`rg -n -i nz vllm_ascend/envs.py vllm_ascend/ascend_config.py` at the **reported** ref
first; a flag that is absent there is a version statement, not a lever, and "set
`weight_nz_mode: 2`" on a 0.11 tree is advice that does nothing. The same grep answers
the opposite question — a flag that is *present in the docs and absent from the code* is
inert at the reported ref, so quote both sides before ranking it as a cause or crediting
it as an experiment.
`scripts/nz-format-gate-check.sh` reports "no gate function at this ref" when
`is_enable_nz` / `_should_trans_nz` / `maybe_trans_nz` are all absent, instead of
implying a missing shape guard.

## Rule 4 — what to hand back

1. **Use the quantized checkpoint.** For any W8A8 / W8A8-dynamic / W4A8 checkpoint the
   dense Linear and MoE weights are already NZ on 910 (Rule 2), so the gain they are
   asking for is already banked and the answer is "no change needed", not a patch.
2. **If they still want bf16 dense NZ at that ref, the experiment is five lines** — and
   it decides whether a patch is worth writing: `torch.npu.config.allow_internal_format =
   True`; `torch_npu.npu_format_cast(w, 29)` on a 2-D weight whose dimensions are
   multiples of 16; compare `F.linear(x, w)` against `F.linear(x, w_nz)` by max-abs-diff;
   then time both. Hand back a `uv run --no-project` snippet rather than a fork.
3. **If the probe passes, the patch is the gate**, not the layer: force the dense classes
   through NZ at `worker/model_runner_v1.py:2633` (or add the same cast in an
   `AscendLinearBase` hook) **with a shape guard**, because a 1-length dimension cannot be
   stored in `FRACTAL_NZ` and turns a blind cast into a load-time `EZ1001`
   (`../startup-failures/nz-weight-format-refusal.md`). The acceptance check is numeric
   equality on one real request plus no `TransData` / a weight-NZ matmul in the profile —
   not "it started".
4. **Label the throughput claim unverified.** No pinned test measures NZ vs ND for dense
   bf16, and the reporter's premise ("提高 matmul 的性能") is the reason to run the probe
   in step 2 first.

## Recheck conditions

Re-grep the anchors at the reporter's ref: this layout code was rewritten at least twice
(inline casts at v0.11 → gate functions `maybe_trans_nz` / `_should_trans_nz` at v0.20 →
`_310p/quantization/methods/` copies at later refs), so `is_310p()` sites move and a
`not is_310p()` MoE cast can become unconditional between refs — the same table at
another ref is a new read, not a quote. `scripts/nz-format-gate-check.sh <paste|-> <root>`
prints what exists at a given root; `nz-weight-format-refusal.md` is the crash-side page
and carries the mode algebra. Claim class: source-verified at the two named commits;
nothing in #3859 was reporter-confirmed (its follow-up was an image with no text).
