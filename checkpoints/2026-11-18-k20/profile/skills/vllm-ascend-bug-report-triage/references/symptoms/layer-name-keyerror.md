# A layer-name `KeyError` during KV-cache init: which caller, which scope

Family: the serve process dies **after the weights load and before any forward** with
`KeyError: 'model.layers.<n>.self_attn.attn'` raised out of upstream
`vllm/config/vllm.py: get_layers_from_vllm_config`. Nothing is missing on disk and no hardware is
involved. Two members, and **the frame that calls the lookup decides which**:

| caller of `get_layers_from_vllm_config` | what is actually wrong |
| --- | --- |
| `vllm_ascend/worker/model_runner_v1.py: initialize_attn_backend → get_attn_backends_for_group` | the name is not in **this rank's** `static_forward_context` at all — a stage/scope mismatch (§2) |
| `vllm/distributed/kv_transfer/kv_connector/v1/<connector>.py: register_kv_caches` | the connector filtered on a layer **class** the model does not subclass, so its map is empty — `kv-connector-registration.md` |

Not this page: a `KeyError: '<module>.weight'` out of `is_layer_skipped_ascend` (checkpoint
description, `quantization-load.md` §2) or `KeyError: '...weight_scale'` out of a model's own
`load_weights` (`quantization-load.md` §3). Those index a checkpoint dict; this family indexes
`static_forward_context`.

## §1 The lookup itself is unconditional — read it before theorising

vLLM `v0.14.1` (identical at `v0.14.0`), `vllm/config/vllm.py:1464-1487`:

```python
if layer_names is None:
    layer_names = list(vllm_config.compilation_config.static_forward_context.keys())
forward_context = vllm_config.compilation_config.static_forward_context
return {layer_name: forward_context[layer_name]
        for layer_name in layer_names
        if isinstance(forward_context[layer_name], layer_type)}
```

A caller that passes an explicit list gets a bare `KeyError` on the **first** name that is absent —
before any `isinstance` filtering can skip it. Only `layer_names=None` takes the safe path.

```bash
git -C /workspace/repos/vllm-project/vllm show <vllm-ref>:vllm/config/vllm.py | sed -n '/def get_layers_from_vllm_config/,+24p'
```

Fingerprint bonus: a traceback whose frames are `vllm.py:<N>` `return {` and `<N+3>`
`if isinstance(forward_context[layer_name], layer_type)` pins this body (`1483`/`1486` at
`v0.14.0`/`v0.14.1`), which is how you confirm the reporter's installed vLLM without their wheel.

The body moved afterwards `[verified]`: from vLLM **v0.22.0** it is
`isinstance(layer := forward_context.get(layer_name), layer_type)` — missing names are *skipped*
instead of raising — introduced by `47d4407d7c` (2026-05-22, "[Model Runner V2] Support sharing kv
cache layers", #35045); absent at `v0.21.0`, present at `v0.22.0` onward. Read the body at the
reported tag rather than assuming either shape.

## §2 vllm-ascend's attention-backend init under PP (vllm-ascend#6678)

Reported: GLM-4.7 (92 layers), the official `v0.14.0rc1` image with vLLM `0.14.1`, CANN 8.5.0, ray
backend — `PP=4 TP=8` on 4 nodes, `PP=3 TP=8` on 3 nodes both die; `PP=2 TP=8` and `PP=4 TP=4` on
2 nodes start. Keys seen: `model.layers.{0,23,46,69}.self_attn.attn`, "正好对应92层 + PP4 的切分".
`[reported]`

Call chain at `v0.14.0rc1` `[verified]`:

- `worker/model_runner_v1.py:2227` `initialize_kv_cache` → `:2634` `initialize_attn_backend` →
  `:2645` `get_attn_backends_for_group` → `:2649-2651`
  `get_layers_from_vllm_config(self.vllm_config, AttentionLayerBase, kv_cache_group_spec.layer_names)`
  → `:2659-2663` `layers[layer_name].get_attn_backend()`.
- A second recorded read of the same tag places the enclosing entry points differently
  (`:2701` `initialize_attn_backend → get_attn_backends_for_group`, lookup call at `:2649`); the lookup
  call `:2649-2651` is the anchor both reads share, so re-read the chain at the tag before quoting an
  entry line. `[verified]` for the call, `[reported]` for the differing entry lines.
- The group's `layer_names` are passed through **unfiltered**: nothing intersects them with the
  names this rank owns. The call shape is unchanged at `v0.15.0rc1:2755`, `v0.16.0rc1:2944` and HEAD
  (`vllm_ascend/worker/v1/model_runner.py:6534`), so "newer build fixes it" is not available.
- **Upgrading the paired vLLM does not fix this class**: the loop right after the lookup still does
  `layer = layers[layer_name]` (HEAD `:6559`), so a name missing from the static context raises one
  line later even where the upstream `.get` form skips it. Do not offer "upgrade vLLM past
  `v0.22.0`" as a workaround for a missing-name KeyError.

The deduction that saves the next investigation: the four keys are the **four stages' first layers**.
A single global list broadcast to every rank cannot produce that set — stage 0 would raise at
`model.layers.23…` and stages 1–3 at `model.layers.0…`, i.e. two distinct keys. Four distinct
stage-front keys mean each rank was handed a list belonging to a **different** stage (stage+1's or
stage−1's list both give exactly `{0,23,46,69}`). So suspect the rank↔stage assignment of
`kv_cache_spec` / `layer_names` in the gather-broadcast path, not the dict comprehension. `[inference]`

Who supplies the list `[verified]` at vLLM `v0.14.1`: `vllm/v1/executor/abstract.py:128-129`
`get_kv_cache_specs() -> self.collective_rpc("get_kv_cache_spec")`, and `ray_distributed_executor.py` at
that tag does not override it — so **every** worker returns its own stage's spec and the engine merges them
into the single `KVCacheConfig` that is broadcast back to all ranks. That is the end of what the traceback
can settle: *which* rank ends up holding which stage's list is a runtime assignment, so **do not re-derive
the four-key arithmetic**. The paragraph above is that deduction, labelled `[inference]`; the only thing
that moves it is the print below. Re-litigating `{0,23,46,69}` with the workaround, the status and this
page already on screen is how a run ships nothing (#6678, three recorded attempts — see
`../workflow/answer-and-claims.md`, "Stop triggers").

Cheapest discriminating print, one line per rank, before the failing call:

```python
ctx = self.vllm_config.compilation_config.static_forward_context
print(sorted(ctx)[:2], sorted(ctx)[-2:], kv_cache_group_spec.layer_names[:2])
```

- local names present but the group's first name is another stage's → assignment/scope defect (§2);
- the group's first name equals the local first name, yet the lookup still raises → the
  registration/prefix of those layers, not the list.

Workaround available from the report's own matrix: stay inside the shapes that start — drop PP below
3 (or run it on 2 nodes; `--pipeline-parallel-size 1`/TP-only is the safe form), accepting the
memory/throughput cost of doing so. `[reported]`, not a fix.

Status: **open** — the thread has no maintainer diagnosis (a second reporter confirms "4 nodes,
tp8 pp4, glm5 也报这个错"; the issue was stale-bot closed). Not established: whether the
differentiator is PP degree or node count (PP3 and PP4 both fail, both 2-node shapes pass), and
whether any feature env in the report (`VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE=1`, `VLLM_ASCEND_ENABLE_NZ=2`,
`--enable-expert-parallel`) participates. Say so rather than asserting a cause.

## Recheck conditions

Re-read the lookup at the reported tag every time (indexing ↔ `.get` changes the failure frame), and
re-read vllm-ascend's loop after any attention-backend refactor — it is the half that still indexes by
name at HEAD. The sibling print for the connector member is in `kv-connector-registration.md` ("The
check that settles it").
