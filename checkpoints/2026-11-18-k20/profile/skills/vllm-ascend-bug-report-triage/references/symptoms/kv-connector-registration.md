# Connector fails at KV-cache registration, or "this connector does not work on this model"

Family: the serve process dies **before any forward**, while the worker registers the KV-cache tensors
with a connector: `KeyError: '<module>.self_attn.attn'` (or another layer name) out of
`kv_connector/v1/offloading_connector.py`, an `assert` on a layer/backend map, or a
`NotImplementedError` about cross-layer KV layout. Nothing about hardware, quantisation or the model
weights is involved; the connector's assumption about which layer class owns a KV tensor is. The
discriminator is the model, not the machine: an MLA model (Kimi-K2.x, GLM-5, DeepSeek) fails while a
GQA model on the same config starts.

Two shapes want different answers:

| shape | who owns it |
| --- | --- |
| worker dies while registering KV caches (`KeyError` on a layer name) | upstream vLLM: the connector looks its layers up by the wrong base class (§2) |
| the connector loads but transfers nothing / is declared unusable, and a maintainer says to use another connector | deployment choice on Ascend: `AscendStoreConnector` (§3) |
| **two** connectors listed in one `MultiConnector` (a layerwise PD child + a KV-pool child) and an `assert` on block-id lengths fires in the scheduler step | the tag's `MultiConnector` implementation: upstream gives the real blocks to one child only (§4) |
| the same `KeyError` with **no connector at all**, under `initialize_attn_backend` | vllm-ascend × PP: the layer list handed to `get_layers_from_vllm_config` names layers this rank did not build (see the section after §3) |

Not this page: the **same** `KeyError` string when the caller is not a connector — the attention-backend
init in `model_runner_v1.py` is a rank/stage scope problem, not a class filter
(`layer-name-keyerror.md`); a connector that starts but never stores/loads (`kv-pool-store-hang.md`), a
PD freeze with no traceback (`pd-collective-hang.md`), PD transfer garble (`hybrid-gdn-pd.md`,
`c8-and-kv-layout.md`), or two connectors composed under `MultiConnector` whose block accounting
collides (`multi-connector-composition.md`).

## Where the code is

- Upstream: `vllm/v1/worker/kv_connector_model_runner_mixin.py` (lifecycle) and, per connector,
  `vllm/distributed/kv_transfer/kv_connector/v1/<connector>.py` — the worker's
  `register_kv_caches(kv_caches)` maps `kv_caches` key names back to layer objects, usually via
  `get_layers_from_vllm_config(vllm_config, <LayerClass>, layer_names)`. At vLLM `v0.17.0` that call is
  `offloading_connector.py:594-604`:
  `layers = get_layers_from_vllm_config(self.spec.vllm_config, Attention, layer_names)` followed by
  `attn_backends = {n: layers[n].get_attn_backend() for n in layer_names}`. `kv_connector/utils.py`
  takes the class as a caller-supplied `layer_type`, so it is not affected by this shape.
- vllm-ascend: `vllm_ascend/distributed/kv_transfer/__init__.py` `register_connector()` registers the
  Ascend connector set (`AscendStoreConnector`, `MooncakeConnectorV1`, `MooncakeConnectorStoreV1`,
  `MooncakeLayerwiseConnector`, `UCMConnector`, `LMCacheAscendConnector`, and — from `v0.16.0rc1`
  only (#7032 `675387f1f`) — a replaced `MultiConnector`; at `v0.13.0` that name is upstream's class,
  which starves a layerwise connector (`multi-connector-composition.md`). The list is per ref, so
  read the registration file at the reported tag. `vllm_ascend/kv_offload/npu.py` `NPUOffloadingSpec` supplies the NPU↔CPU handler
  for the upstream offloading framework. **It does not replace or patch the upstream
  `OffloadingConnector`**, so that class runs as upstream wrote it.

## The check that settles it

```bash
# which layer classes exist, and are they related at the reported ref?
git -C /workspace/repos/vllm-project/vllm grep -n "^class Attention\|^class MLAAttention\|^class .*AttentionLayerBase" <vllm-ref> -- vllm/model_executor/layers
# what does the connector filter on?
git -C /workspace/repos/vllm-project/vllm show <vllm-ref>:vllm/distributed/kv_transfer/kv_connector/v1/<connector>.py | sed -n '/def register_kv_caches/,+12p'
```

A layer class named in the filter that is **not** an ancestor of the class the model uses gives an
empty map and then a `KeyError` on the first layer name. `git grep -n "get_layers_from_vllm_config"
<vllm-ref> -- vllm/distributed/kv_transfer` lists the sibling call sites to compare.

## Reading order for a new report

1. The deepest `vllm_ascend/` frame: if it is `kv_offload/npu.py` or `cpu_npu.py`, the Ascend side is
   only wiring; if the raise is a `KeyError` in `offloading_connector.py`, it is the registration shape.
2. Read the connector's `register_kv_caches` at the *reported* vLLM tag (the command above).
3. Is the model MLA? The layer names in the traceback (`...self_attn.attn`, `mla_attention`) answer it;
   a GQA model with the same crash means another cause.
4. Ancestry for the fix before naming a version: `scripts/ref_ancestry.sh vllm v0.17.2rc0 ca1954d58c`.

## 4. A `MultiConnector` that pairs a layerwise PD child with a KV-pool child → `multi-connector-composition.md`

This page's §2/§3 are about a *single* connector failing to register or to be chosen. Two connectors composed
under one `MultiConnector` (`MooncakeLayerwiseConnector` + `AscendStoreConnector` and siblings) is a different
contract — upstream hands the request's real blocks to only the child that won the external-token match — and
its symptom is an accounting assert inside the transport child, not a `KeyError` at registration. The
mechanism, the `AscendMultiConnector` replacement (first tag `v0.16.0rc1`, #7032 `675387f1f`), the per-ref
registration check and the worked #7085 answer shape are in `multi-connector-composition.md`; read the
registration file at the reported tag before naming a class, because the binding is per ref.

## Worked case: vllm-ascend#7971 — `OffloadingConnector` + MLA (GLM-5, Kimi-K2.5)

Reported: vLLM `0.17.0`, vllm-ascend `0.17.0rc1`, `--kv-transfer-config
'{"kv_connector": "OffloadingConnector", ...}'`; GLM-5 and Kimi-K2.5 fail, GLM-4.7 works.

**Verified upstream-vLLM defect, fixed in-vLLM, absent from the reporter's release.**

- At vLLM `v0.17.0`, `offloading_connector.py:596-598`
  `get_layers_from_vllm_config(self.spec.vllm_config, Attention, layer_names)` filters by **`Attention`**,
  then `:600` indexes `layers[layer_name]`. `Attention` (`vllm/model_executor/layers/attention/attention.py:229`)
  and `MLAAttention` (`.../mla_attention.py:426`) are both plain `nn.Module` + `AttentionLayerBase`;
  MLA layers are therefore filtered out, the map is empty, and the first MLA layer name raises
  `KeyError`. Any MLA model (Kimi-K2.x, GLM-5, DeepSeek family) reproduces; a GQA model such as
  GLM-4.7 does not — matching the report.
- Fix `ca1954d58c` (#37090, "[Bugfix] Disable cross-layer KV cache for MLA attention backends",
  2026-03-17) swaps the filter for `AttentionLayerBase`. The same commit makes
  `MLACommonBackend.get_kv_cache_stride_order` return the identity permutation, i.e. **cross-layer KV
  cache is not supported for MLA** — so repairing only the `isinstance` still leaves an unsupported
  layout. Ancestry: `v0.17.0` no, `v0.17.1` no, `v0.17.2rc0` yes, `v0.18.0` yes.
- Maintainer advice in the thread (`MengqingCao`): "The original offloading connector doesn't work on
  vllm-ascend, could you please have a try with `AscendStoreConnector` instead?" [reported]
  `AscendStoreConnector` exists at `v0.17.0rc1`
  (`vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py:66`, registered
  in `.../kv_transfer/__init__.py:36`), and `docs/source/user_guide/feature_guide/kv_pool.md:272`
  documents the MLA case: the decode node stores KV for the prefill node with
  `consumer_is_to_put: true`.

Workaround ladder: swap to `AscendStoreConnector` (semantics change: KV pool / prefix cache rather
than plain CPU offload — read the doc's pool-side parameters); or run a vLLM with `ca1954d58c`
(≥ `v0.17.2rc0`) with a vllm-ascend that pairs with it; or local patch of the filter type for a
test-only build. Reporter confirmation: none.

Falsifying check: print `len(layers)` after the `get_layers_from_vllm_config` call — `0` confirms this
mechanism; a non-zero map means the failure is a different registration bug.

## Distinguish: the same layer-name `KeyError` with no connector and no `register_kv_caches`

A layer-name `KeyError` at KV-cache init is not automatically the connector's class filter. With **no**
`--kv-transfer-config`, vllm-ascend#6678 raises one call earlier in the same stage, under
`initialize_kv_cache` → `initialize_attn_backend` → `get_attn_backends_for_group`, which passes the KV
group's `layer_names` to `get_layers_from_vllm_config` **unfiltered** — so the name that is missing is
absent from that rank's `static_forward_context` (a PP stage/scope mismatch) rather than filtered out by
a class. The `AttentionLayerBase` filter the attention-backend path passes is already the class that
contains MLA layers, which is exactly why the two cases need different answers.

Route by the frame that calls the lookup: a connector's `register_kv_caches` (this page, §1–§3) versus
`worker/model_runner_v1.py: initialize_attn_backend → get_attn_backends_for_group`. The full second
family — verified call chain, the four-stage first-layer evidence, the discriminating print, workaround
and falsifier — is `layer-name-keyerror.md` (#6678, open; no fix).

## Recheck conditions

The vLLM layer hierarchy and the connector's filter move (`vllm/model_executor/layers/attention/`
was flattened later; `offloading_connector.py` was split, moving `register_kv_caches` into a
`ConnectorWorker`). Re-read the filter at the *reported* vLLM tag, and re-check the ancestry of the
cited fix before naming a release.
