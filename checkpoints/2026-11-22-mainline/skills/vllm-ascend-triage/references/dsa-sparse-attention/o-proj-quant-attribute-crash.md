# DSA-CP o_proj weight switching needs a W8A8-static o_proj

Scope: a Lane B engine-start failure on a **DSA/SFA** model (DeepSeek-V3.2 family,
GLM-5 / 5.1 — any checkpoint whose `hf_text_config` carries `index_topk`) started
with FLASHCOMM1, the legacy `VLLM_ASCEND_ENABLE_FLASHCOMM`, or shared-expert-DP:

```text
AttributeError: 'AscendRowParallelLinear' object has no attribute 'aclnn_input_scale'
  .../vllm_ascend/attention/sfa_v1.py:707 in _init_o_proj_tp_full_params
  .../vllm_ascend/attention/sfa_v1.py:544 in process_weights_after_loading
```

(or the `aclnn_input_scale_reciprocal` / `aclnn_input_offset` sibling; the class
name is `vllm_ascend/ops/linear.py:253`). It fires during weight post-processing,
before memory profiling, and it is a **quant-method/feature combination**, not a
broken install, an upstream-drift patch or a card.

Verified against vllm-ascend **v0.19.1rc1** (`da421afad7`), the ref in #9984; the
same frame is in the v0.18.0-a3 nightly of #7937, so treat the chain as stable
across those refs and re-grep the line numbers.

## The chain

| # | Site (v0.19.1rc1) | Fact |
|---|---|---|
| 1 | `vllm_ascend/utils.py:799-820` | `enable_sp()` is `VLLM_ASCEND_ENABLE_FLASHCOMM1` or legacy `VLLM_ASCEND_ENABLE_FLASHCOMM`, and `enable_shared_expert_dp` forces it `True` with an INFO line ("shared_expert_dp requires enable_sp = True") |
| 2 | `vllm_ascend/utils.py:1271-1280` | `enable_dsa_cp()` = `hasattr(hf_text_config, "index_topk") and enable_sp()`; the comment says it is the temporary gate for ds32 DSA-CP, so **the model family, not the flag, is the first half of the condition** |
| 3 | `vllm_ascend/utils.py:1282-1294` | `enable_dsa_cp_with_layer_shard()` is True only on the **P** instance (`kv_transfer_config.kv_role == "kv_producer"`); a decode instance or a plain non-PD launch has no `kv_transfer_config`, so it is `False` |
| 4 | `vllm_ascend/attention/sfa_v1.py:538-544` | `if self.enable_dsa_cp:` → layer-shard branch, **else** `self._init_o_proj_tp_full_params()` — unconditional inside that else |
| 5 | `vllm_ascend/attention/sfa_v1.py:700-720` | that function clones `self.o_proj.aclnn_input_scale` / `_reciprocal` / `_offset` into the TP-mode copies (`:707-709`) and `repeat`s them into the full-mode copies (`:718-720`), no `hasattr` guard |
| 6 | `vllm_ascend/quantization/methods/w8a8_static.py:142-152` | the **only** producer of those three attributes: `process_weights_after_loading` builds them from `layer.input_scale` / `input_offset`. It also serves compressed-tensors W8A8 (same method, `ascend_quant_method == COMPRESSED_TENSORS_METHOD`). `_310p/quantization/methods/w8a8s.py:84-86` and `w8a8sc.py:108-110` are the 310P equivalents |
| 7 | `vllm_ascend/quantization/methods/w4a8.py:143-180` | W4A8 has **no** `aclnn` attribute at all (grep `aclnn` in that file: no match): it dequantises through `weight_scale_second` / `antiquant_scale` (`process_scale_second`, `:95-140`). An unquantised o_proj never has the attributes either |

So the requirement is "an **input-scale** (static W8A8) o_proj", not "not w4a8":
a BF16 checkpoint with FLASHCOMM1 on a DSA model reaches the same line.

The same tensors are consumed at decode: `enable_dsa_cp_with_o_proj_tp()`
(`utils.py:1296-1310`) is True when `kv_transfer_config is None` — i.e. **every
plain non-PD launch** — and `sfa_v1.py:1269-1278` then calls
`_handle_o_proj_weight_switch_and_forward`, switching between exactly the TP and
Full copies built at `:707-720`. So a single-node serve is already on the default
path into this feature (no PD disaggregation needed to hit it), and a code fix has
to cover both consumers, not only the `hasattr` at `:707`.

## Answer rule

Verdict first: this combination is **not supported at this ref** — the DSA-CP
TP/Full weight switch assumes the W8A8-static attributes, and the code has no
guard. Cheapest action: drop the FLASHCOMM1 export (reporters typically confirm it
then starts), or run the **W8A8** checkpoint the tutorial ships. There is no other
flag that makes W4A8 work on this path, so do not offer an ablation ladder for MTP
or capture settings.

Before promising that unsetting is enough, one grep of the launch's
`--additional-config`: **`enable_shared_expert_dp` turns SP on by itself**
(`ascend_config.py:81-89` asserts `enable_sp(…, enable_shared_expert_dp=True)`, row
1), so the same line comes back with no export at all — that is the only other
switch that reopens this path. A similar-looking key is *not* it:
`multistream_overlap_shared_expert` (`ascend_config.py:106`) only feeds the
AIV-mode capture-size arithmetic (`utils.py:525-536`) and the mix-placement check
(`ascend_config.py:200-203`), never `enable_sp`, and #9984's launch carries exactly
that key with `true` — so there the unset is sufficient. Name which of the two the
paste has instead of warning generically about "other flags".

Then quote the validation status, which is what makes it "unsupported" rather than
"a config mistake":

- the pinned GLM5 tutorial's only FLASHCOMM1 recipes serve
  `/root/.cache/glm5-w8a8` (`docs/source/tutorials/models/GLM5.md:777,782` and
  `:859,864` at v0.19.1rc1) and both are PD **prefill-node** blocks
  (`--data-parallel-rank`, `VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT`,
  `ASCEND_A3_ENABLE`), so the pin never pairs the flag with a W4A8 weight; its
  notice list says FlashComm targets the prefill node (`:1330`, "with FlashComm
  enabled, layer_sharding list cannot include o_proj"). A reporter who says
  「使用文档默认配置」 and links `…/en/latest/…` read a build of main that can
  postdate the pin — quote the pin's own block and say the two may differ
  (`../tutorial-verification/docs-version-space-and-image-tag.md`);
- every `tests/e2e/**` config that sets `VLLM_ASCEND_ENABLE_FLASHCOMM1` at that ref
  is W8A8 or BF16 — no W4A8 coverage;
- the requirement is structural (`sfa_v1.py:707-709`), so the fix is a guard in
  `_init_o_proj_tp_full_params` (or a quant-method check in `enable_dsa_cp`),
  not a user flag.

Prove it from the paste in one step: the traceback's own frames (`:544` then
`:707`) already prove the DSA-CP branch ran, i.e. `index_topk` present and
`enable_sp()` true — no need to read the model config. Then one `rg` for which
linear method built `o_proj`.

## Adjacent, do not conflate

- The FLASHCOMM1 row-parallel fused path itself branches only on
  `UnquantizedLinearMethod` and `AscendW8A8LinearMethod`
  (`vllm_ascend/ops/linear_op.py:528-571`, inside `matmul_and_reduce` at `:500`)
  and falls back to `quant_method.apply` + `reduce_scatter` otherwise (`:572-574`).
  The crash above is in weight post-processing, before any of this runs; do not
  present the fallback as the failure.
- A generic `AttributeError` on a patched decoder/connector class is downstream
  drift (`../upstream-downstream-drift/index.md`). The missing attribute here is a
  *purpose-built* quant tensor, so that route is a dead end.
- FLASHCOMM1 scope for non-DSA models is `../feature-enable-recipes/index.md`;
  PCP/DCP for the same model family is
  `../feature-enable-recipes/context-parallel-pcp-dcp.md`.

## Precedents

- **#7937** v0.18.0-a3 nightly, GLM5-W4A8, PD disaggregation **decode** instance
  (so `enable_dsa_cp_with_layer_shard()` is False, row 3): same
  `_init_o_proj_tp_full_params` frame; the snapshot carries no resolution, so the
  defect is at least one release older than #9984.
- **#9984** v0.19.1rc1, GLM-5.1-w4a8 with the tutorial's FLASHCOMM1 + MTP flags on
  one node, `{"multistream_overlap_shared_expert": true}` in `--additional-config`:
  the reported `AttributeError`, confirmed to disappear with FLASHCOMM1 off
  (reporter-confirmed). The reporter's own control also retires MTP and the
  `patch_balance_schedule.py` EngineCore wrapper — the flag is the only variable,
  which is why no ablation ladder is owed.

## Recheck

`rg -n "aclnn_input_scale" <vllm-ascend>/vllm_ascend/` at the reported ref: if a
non-W8A8 method starts creating the three attributes, or
`_init_o_proj_tp_full_params` gains a guard, this combination becomes supported
and the answer changes.
