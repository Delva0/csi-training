# Hybrid linear-attention (GDN/Mamba) models under PD or a KV pool: which cache actually moves

## Scope

A hybrid model — full-attention layers interleaved with gated-delta-net / linear-attention layers
(Qwen3-Next, Qwen3.5, Qwen3.8) — deployed with PD disaggregation or a KV pool, where the decode side
answers mixed-language garbage, drifts on an accuracy benchmark, or stalls. Covers the "two caches"
problem: the paged attention KV and the recurrent/conv state of the linear layers, which are separate
`KVCacheSpec`s and are not automatically transferred together. Not the dense-model PD hangs
(`pd-collective-hang.md`), not sparse C8 (`c8-and-kv-layout.md`), not the warm-up SP crash
(`flashcomm1-and-moe-reduce.md` §A).

Also not this page: a **single node with no connector and no pool** answering the same mixed-language
garbage. Nothing is moving a cache there, so the "two caches" mechanism cannot apply — the levers are the
spec-decode path (`spec-decode-and-parallelism.md`) and a hand-written `--hf-overrides` rope block that
disagrees with the checkpoint (`model_config`/yarn factors), each falsified by re-running at
`temperature=0` with that one feature removed (#10405).

## The naming trap that produced a wrong conclusion

vLLM models those linear layers as **`MambaSpec`** caches; the connector code says `MambaSpec`,
`mamba_cache_mode`, `use_attn_mamba_hybrid`, `_update_hybrid_attention_mamba_layout`. There is no `gdn`,
`linear_attn` or `state_cache` symbol in the transfer code. One recorded run grepped
`gdn\|linear_attention\|state_cache` inside `vllm_ascend/distributed/kv_transfer`, got an empty result
and concluded "the connector has no notion of hybrid state at `v0.18.0`" — wrong: at that tag
`mooncake_layerwise_connector.py` contains 23 `mamba` references and handles `MambaSpec` explicitly.
**Grep `mamba` first**, in the connector the report actually configures, and remember that an empty grep
is only a finding when the ref resolves (`../workflow/version-pinning.md`).

## What the tags actually contain (verified by `git grep -c -i mamba <ref> -- <path>`)

| ref | `kv_p2p/mooncake_layerwise_connector.py` | `kv_p2p/mooncake_connector.py` | `kv_pool/ascend_store/*` |
| --- | --- | --- | --- |
| `v0.18.0` | 23 hits — `MambaSpec` layer specs (`:289`, `:1135`, `:1416`, `:1463`), single-block transfer, `mamba_cache_mode == "align"` | **none** | **none** (`kv_pool/cpu_offload` has 9) |
| `v0.23.0` | 29 hits | 50 hits | present (`config_data.py`, `pool_scheduler.py`, `pool_worker.py`) |

- At `v0.18.0` the layerwise connector also carries per-layer conv/ssm base addresses and picks the
  transferred mamba block from `cache_config.mamba_cache_mode`: `"align"` →
  `len(local_block_ids) - num_speculative_tokens - 1`, otherwise `0` (comment: "only support one block
  transfer for mamba"). A multi-block layer mapping is silently unsupported there.
- The plain connector (`MooncakeConnectorV1`) at that ref unpacks per layer as
  `for _, (k_cache_layer, v_cache_layer) in self.kv_caches.items()` with no `MambaSpec` branch, so a
  linear-attention layer's recurrent state has no transfer path through it. At `v0.18.0` the tree has
  only those two P2P connectors (no `MooncakeConnectorV2`, whose Mamba support arrives later), so on
  that release the choice is literally "layerwise (state moves) or V1 (it does not)".

So the deployment detail that decides everything is **which connector** the report configures (and
whether both roles use it): read the report's `--kv-transfer-config` (`kv_connector`, `kv_role`) before
theorising, and note that a dense model on the same config is the clean control.

## Hybrid-cache support that lands after `v0.18.0` (a range, not a bisect; none an ancestor of the tag)

- `5fa57c55b` #14340 — "Support non-contiguous KV cache in hybrid models like Qwen3.5 and GQA models".
- `44d767e1c` #17867 — "Normalize combined attention caches in MooncakeConnectorV1" (the plain
  connector, i.e. the hole above).
- `ba58e08a8` #17924 — "Keep KV cache contiguous for paged attention";
  `897da6d50` #17868 — block-strided KV caches in the transpose operator.
- `23b76ca20` #19974 — "Handle TP-replicated Mamba state in Mooncake V2";
  `749d743eb` #19518 — Mamba transfers under PCP head sharding;
  `2f285ddcc` #19880 — preserve GLM shared slots with layerwise KV transfer.

That sequence says the plain connector gained hybrid handling somewhere after `v0.18.0`, so "upgrade
past it" is defensible while naming one commit as *the* fix is not.

## #8920 (`0.18.0`, Qwen3.5-MoE, PD per the docs' config)

The body is the issue template with `collect_env` left unedited ("Your output of above commands here")
plus one sentence: the curl answer is wrong, Chinese and English mixed. No start command, no log.
Thread: `yiz-liu` "Same with #8916, waiting for more info"; `LoganJane` "This issue has been resolved
in the latest version. Please use the latest release or rc version". Closed, no diagnosis. Twins: #8916
(same model, PD, stuck after ~1 h — a different symptom) and #8932 (AIME score 3.5 points lower on
`0.18.0` than on `0.18.0rc1`, stale-closed).

Honest answer shape: workaround = move to a later release (maintainer-attributed, unverified for this
report); mechanism = *hypothesis*, the decode side continues from an attention-only transfer while the
linear layers' state is missing, which is consistent with fluent-but-wrong text rather than a crash;
status = closed without diagnosis; limits = no logs, no config file, and the connector variant the
reporter used is unknown.

**Cheapest falsifying checks to request** (do not skip them for a report that *does* carry logs):
1. Switch the same deployment to the layerwise connector (which does carry `MambaSpec` at that tag) or
   to a dense model — if the dense model is clean, the hybrid cache is implicated.
2. Ask whether the garbage appears only for requests actually prefilled on the other role (pure local
   prefill also wrong ⇒ not a transfer defect).
3. Check `mamba_cache_mode` and the connector's single-block rule for mamba layers.

## Recheck conditions

The connector set, the `MambaSpec` plumbing and the hybrid KV layout are being rewritten release to
release; re-run the three `git grep -c -i mamba` counts at the reported tag, and re-read the report's
`--kv-transfer-config` before reusing any claim above. Nothing here was reproduced on hardware.
