# PD + hybrid linear-attention state: connector choice and `mamba_cache_mode`

Worked case: #12339 — Qwen3.5-397B-W8A8-MXFP8-FULL_QUANT, Ascend950, vLLM v0.23.0
/ vllm-ascend releases/v0.23.0, CANN 9.1.0.B103, PD disaggregation with
`MooncakeConnectorV1` (P `kv_producer`, D `kv_consumer`), TP=8 + EP, no MTP:
**the same request alternates between a good answer and a bad one**. Closed
`已解决` by the reporter (public follow-up 2026-08-03) with no published mechanism —
so the deliverable is the ranks + the ablation that separates them, never a single
asserted cause, and no row below was ever confirmed on the reporter's side.

Why this model is different: Qwen3.5-397B is a **hybrid** stack (GDN linear
attention layers + full attention, plus MTP layers). Under PD the decode node
needs the attention KV *and* the GDN conv/ssm state; a connector that transfers
only `FullAttentionSpec` groups gives the D node a state the P node never
produced.

## 1. Check the connector against the documented one (2 greps)

| Model family | Tutorial's PD connector at v0.22.1rc1 |
|---|---|
| Qwen3.5-397B-A17B | `MooncakeLayerwiseConnector` (`docs/source/tutorials/models/Qwen3.5-397B-A17B.md:294,336-361`), with `--speculative-config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3, ...}'` on both sides, `recompute_scheduler_enable`, `enable_cpu_binding`, `enable_fused_mc2` (`:357, 437-442, 550-553`) |
| DeepSeek-V4-Pro / V4-Flash (also hybrid) | `MooncakeHybridConnector` (`docs/source/tutorials/models/DeepSeek-V4-Pro.md:646,1178`; `DeepSeek-V4-Flash.md:465,853`) |
| GLM-5.2 (**DSA/MLA + MTP, not hybrid**) | `MooncakeConnectorV1` with `use_ascend_direct` (`docs/source/tutorials/models/GLM5.2.md:694,763,831,899,1371,1442`; A2 variant `MooncakeConnector` `:1040,1121`), MTP on both sides, and **no** `--no-disable-hybrid-kv-cache-manager` anywhere in that tutorial — a reporter pairing `MooncakeHybridConnector` with that flag here has copied the DeepSeek-V4 recipe (`recipe-contract-diff.md`, #13381) |

A `MultiConnector` of `MooncakeHybridConnector` **plus** `AscendStoreConnector`
(the DeepSeek-V4 pool recipe) adds a second config surface that is easy to
misattribute: the connector's `kv_buffer_size` / `kv_block_size` live in
`kv_connector_extra_config` and are registered by `register_buffer`
(`mooncake_hybrid_connector.py:1686`), while the KV pool's
`global_segment_size` lives in the separate Mooncake store json behind
`MOONCAKE_CONFIG_PATH`. Neither value bounds the other, and an oversized segment
breaks both connectors' channels at once — see `kv-pool-store-segments.md`
(#13439: 40 GB segment, `503900` on the p2p connector, garbled decode output,
2 GB fixed it).

Registry at v0.22.1rc1 (`vllm_ascend/distributed/kv_transfer/__init__.py:29-55`):
`MooncakeConnectorV1` → `kv_p2p/mooncake_connector.MooncakeConnector`;
`MooncakeHybridConnector` → `kv_p2p/mooncake_hybrid_connector.MooncakeConnector`
(a separate 1.9k-line implementation); `MooncakeLayerwiseConnector`;
`AscendStoreConnector` / `MooncakeConnectorStoreV1`. So "Mooncake" is four
different transfers: the reporter's `MooncakeConnectorV1` is the generic one and
is **not** the connector either tutorial uses for a hybrid model. Constraints
inside the hybrid connector are hard asserts, worth quoting when you compare:
`pcp_size * dcp_size == 1` (`mooncake_hybrid_connector.py:1083,1395`) and equal
P/D TP with Mamba (`:1419`, "Mooncake connector does not support different TP
size with Mamba").

**The generic connector's mamba support is a version question — settle the era before
quoting the mismatch at all.** At vllm-ascend **v0.22.1rc1** (`5f6faa0`)
`kv_p2p/mooncake_connector.py` carries a full Mamba branch, so "the generic connector
cannot carry hybrid state" is **false** there: `is_mamba_group = group_spec["kv_cache_spec_type"]
== "MambaSpec"` (`:691`), a dedicated branch that moves one block per MambaSpec group
(`:724-730`), `_append_mamba_transfer_meta` (`:920-960`) splitting
`remote_conv_addr, remote_ssm_addr = dst_layer_base_addr[:2]` with
`remote_ssm_len = local_ssm_len // tp_ratio`, and `reformat_kv_cache_hybrid_linear_torch`
(`:892`); the worker's geometry is per-address too
(`block_len_per_addr[layer_idx][cache_idx] = element_size * prod(block_shape)`,
`:1993-2014`, plus a dedicated `_get_mamba_conv_padding` for the conv page at
`:1907-1912`), so the flat `num_blocks * block_len`
registration the paragraph below describes is gone. One call places the pin:

`rg -c 'MambaSpec|_append_mamba_transfer_meta' vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py`
→ `13` at v0.22.1rc1, **empty** at v0.18.0. Non-empty ⇒ the connector is the *recipe*
deviation (row 2 of §4, rank 3), never the root cause, and the ranks to keep are the
single-block index arithmetic (§3 item 2) and `mamba_cache_mode` (§2). Empty ⇒ the
paragraph below is the lane.

**On that older pin the mismatch is a page-geometry assumption, not only a missing
branch** — verified at vllm-ascend **v0.18.0** (`e18643f8`) + vllm **v0.18.0**
(`bcf2be96`), the pin of recorded #8920 (Qwen3.5 MoE under PD "per the docs", curl answer
mixes Chinese and English, no env dump / serve command / log; answered with the ranked
ladder). `rg -c Mamba kv_p2p/mooncake_connector.py` is **empty** at that pin, but that grep
is not the evidence — its transfer is deliberately model-agnostic (one flat
`local_block_ids`/`remote_block_ids` list, `:376-394`, `:471-505`). The evidence is
`register_kv_caches` (`:1183-1232`): `num_blocks` and `block_len` are read off the
**first** cache tensor (`first_kv_cache.shape[0]`, `block_len[i % length]`) and every
layer's region is registered as `num_blocks * block_len`, while the reformat paths unpack
each layer as a `(k_cache, v_cache)` 2-tuple (`:595`, `:641`). A hybrid model's mamba group
carries the conv-padded page `patch_mamba_config.py:80-86` builds at that pin and an
`(ssm, conv)` state pair, which that arithmetic cannot describe. Quote those sites, and
ask for the connector's own line — it prints its assumption at `:1196` (`num_blocks: %s`),
`:1202/:1210` (`block_shape: %s`) and `:1213` (`Registering KV_Caches. use_mla: %s,
use_sparse: %s, shape %s`), so one pasted D-node line shows which geometry was registered.

**"I followed the vllm-ascend PD doc" is not a connector statement.** At v0.18.0 the
generic feature page `docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md`
uses `MooncakeLayerwiseConnector` in its 2P1D external-DP blocks
(`:295,354,413,471`) and `MooncakeConnectorV1` in the later ones (`:538,597,656,714`), all
on DeepSeek-R1 — so ask which section they copied instead of calling the connector user
error, then diff against the model page: at that pin
`docs/source/tutorials/models/Qwen3.5-397B-A17B.md` pins `--no-enable-prefix-caching`
(`:312` P / `:391` D), `--no-disable-hybrid-kv-cache-manager` (`:328` / `:408`),
`--speculative-config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3,
"enforce_eager": true}'` (`:329` / `:409`) and `--enforce-eager`; `:145` states that
enabling prefix caching on a mamba-like family needs `--enable-prefix-caching
--mamba-cache-mode align` and may raise `block_size` to 2048, and `:518` carries the
D-node known-issue note (#7944 — cite the tutorial line, its body is not in these
snapshots).

## 2. The `mamba_cache_mode` gate (source-verified, v0.22.1rc1 / vllm v0.22.1)

**Older pin, before quoting any of it:** at v0.19.1rc1 (`da421af`) that file has
**no** `_using_kv_store()` and no forced-align block — it only delegates to
`MambaModelConfig.verify_and_update_config` and sets `mamba_block_size` at
`:104-107` (`block_size` when prefix caching *and* `align`, else `max_model_len`);
at v0.18.0 the same two effects are at `patch_mamba_config.py:89-92`, and upstream
`MambaModelConfig.verify_and_update_config` is `vllm/model_executor/models/config.py:354-411`
with the identical `none`/`all`/`align` logic.
So on a report older than v0.22.1rc1, `align` is whatever the reporter's
`--mamba-cache-mode` / upstream default made it, and the two consumers to read are
`worker/model_runner_v1.py:1590` and, under PD, the connector itself —
`distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py:230,290` branch on
`mamba_cache_mode == "align"`, i.e. the transport, not only the runner, changes
behaviour with the mode.

At v0.22.1rc1, `vllm_ascend/patch/platform/patch_mamba_config.py` patches
`HybridAttentionMambaModelConfig.verify_and_update_config`:

- `_using_kv_store()` (`:11-29`) returns True only for `AscendStoreConnector` (or
  a `MultiConnector` containing it). Its own docstring: *"In the scenario where
  only PD separation is used, mamba_cache_mode is not automatically set to
  align."*
- `:136-142` therefore forces `mamba_cache_mode = "align"` (and asserts it, "only
  support 'align' when kv_transfer enabled now!") **only** when that KV-store
  connector is in use and the spec method is not `extract_hidden_states`.
- `:143-146` then sets `mamba_block_size = block_size` only when prefix caching is
  on and the mode is `align`; otherwise `mamba_block_size = max_model_len`.
- Consistency check: `kv_pool/ascend_store/pool_scheduler.py:68-71` raises
  `NotImplementedError("AscendStore hybrid linear-attention support currently
  requires mamba_cache_mode='align'.")` — the KV-store path refuses anything else,
  while the p2p Mooncake path never enters that branch.
- The mode is not cosmetic: the runner's per-step hybrid state handling is gated
  on it at `vllm_ascend/worker/model_runner_v1.py:2087` (`preprocess_mamba` /
  deferred state corrections) and `:2242` (`mamba_utils.do_mamba_copy_block`).
- Scope of that docstring: it describes what the *patch* does. Independently of any
  connector, upstream `MambaModelConfig.verify_and_update_config`
  (`vllm/model_executor/models/config.py:351-372` plus its `"all"`→`align` fallback at
  `:373-382`, at vllm v0.20.2) already sets
  `align` whenever `enable_prefix_caching` is on and the model lacks
  `supports_mamba_prefix_caching` — check that flag before quoting the next
  paragraph at another ref.

Consequence to state carefully: on plain Mooncake PD **with prefix caching off**, a
hybrid model keeps `mamba_cache_mode="none"` with a whole-sequence-sized mamba page,
and the align-mode copy paths never run. That is what the tutorials' own PD blocks
ship (`--no-enable-prefix-caching` on both sides, e.g. `Qwen3.5-397B-A17B.md:312,391`
at v0.18.0), while the KV-store path, the connector's `align` branch and the unit
tests (`mamba_cache_mode="align"`/`"all"`) only exercise the other side — i.e. the
**documented deployment is the unexercised one**, not the exotic one. It is a
**candidate**, not the proven cause of #12339: neither run could show the D node
receiving a stale state.

## 3. MTP under PD: the method-string trap and the accounting sites

For a "PD 分离 + MTP 精度" report, the cheapest false lead is the *method name*.
vllm-ascend compares `speculative_config.method` with the literal `"mtp"`
(`vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py:479`) while the
tutorials pass a model-specific name (`{"method": "qwen3_5_mtp", ...}`,
`docs/source/tutorials/models/Qwen3.5-397B-A17B.md:127`; `deepseek_mtp`,
`qwen3_next_mtp`, `glm4_moe_mtp` elsewhere), which looks like a missed branch. It
is not: upstream normalises every `MTPModelTypes` variant *before* vllm-ascend
sees it — `vllm/config/speculative.py:520-524` (vllm v0.21.0) warns "method `<x>`
is deprecated and replaced with mtp" and sets `self.method = "mtp"` — so the gate
does fire for the documented command. One grep settles it:
`grep -n "MTPModelTypes" -A8 vllm/config/speculative.py`. Recorded #10961 ranked
the opposite conclusion first ("the gate misses `qwen3_5_mtp`"), labelled it
source-verified and left it open; that ranking was wrong at v0.21.0rc1, and the
report had no log to support any ranking in the first place.

What actually carries MTP through the connector, cheapest print first:

1. **`num_draft_layers`** — `mooncake_connector.py:476-489`: 1 when the method is
   `mtp`, otherwise the draft model's `hf_config.num_hidden_layers`, consumed by
   the PP layer window (`end_layer_index += self.num_draft_layers`, `:654` inside
   `pp_layer_indices`). Print it on both roles; a mismatch moves the wrong layer
   set and shows up as wrong tokens, never as an error.
2. **MambaSpec block indexing** — the arithmetic
   `transfer_block_idx = len(remote_group_block_ids) - self.num_speculative_tokens - 1`
   takes the P node's **last** mamba block into the D node's **first** allocated one
   (`mooncake_connector.py:728-730` at v0.22.1rc1, `:698-703` at v0.21.0rc1), so exactly
   one block moves whatever the D-side page size is; `num_speculative_tokens` (`:449-453`)
   must match on both roles and moves that index — an ablation row (3 → 1), not a theory.
   **The block-count equality check is version-scoped**: v0.21.0rc1 raises
   `RuntimeError("For MambaSpec num block should equal on P node and D node.")` there,
   while v0.22.1rc1 replaces it with a comment ("When Prefix Caching is enabled on both P
   and D nodes, num_block should not be forced to match", `:724-726`) and no assert —
   grepping the `RuntimeError` at a newer pin finds nothing, and that absence is not a fix
   for the alternating output.
3. **Layer-name predicates** — `"mtp" in layer_name` at `:972,1714,1754` routes MTP
   KV-cache layers to indices above `num_layers`. These key off the *cache dict
   names*, not the method string, so the check is `print(kv_caches.keys())` on both
   roles: names without `mtp` drop all three sites into the non-MTP branch at once.

Items 1 and 3 are the v0.21.0rc1 tree (`80610e4`); item 2 is quoted at **both**
v0.21.0rc1 and v0.22.1rc1 (`5f6faa0`). #10961 reported v0.22.1rc1, which that snapshot
does not carry — say so and have the reporter re-grep.

## 4. The ablation to hand back (this is the answer when the symptom alternates)

Ask for a single fixed prompt/seed, then one row at a time, both P and D:

1. `--speculative-config` MTP **on** (the tutorial's shape) — the reporter's
   "works with MTP, breaks without" is itself a strong signal that the failing
   path is the single-token (query-len 1) decode step.
2. Connector: `MooncakeConnectorV1` → `MooncakeLayerwiseConnector` (tutorial for
   this model) and → `MooncakeHybridConnector` (hybrid-aware); note which of the
   three has the same TP on both sides and `dcp=pcp=1`. This row asks whether the
   *validated* recipe works, not whether the generic connector can carry mamba state —
   at ≥v0.22.1rc1 it can (§1), so a clean result here is a coverage finding.
3. `--enforce-eager` on D (kill the `FULL_DECODE_ONLY` graph path) while keeping
   everything else; if the alternation disappears, the graph/padding path is in
   scope, not the transport.
4. `--no-enable-prefix-caching` on both (already set here) vs the #7944 D-node
   prefix-cache issue (`Qwen3.5-397B-A17B.md:553`) — say that this row is already
   excluded in this report.
5. EP off (`--enable-expert-parallel` removed) and `recompute_scheduler_enable`
   off on P/D: two independent re-prefill/dispatch paths that can resample the
   same request differently.
6. Weight identity: same checkpoint directory on both nodes (md5 of one shard +
   `config.json`), because a P/D weight mismatch looks exactly like alternating
   corruption.
7. Quantisation: `W8A8_MXFP8` on A5/Ascend950 has its own version gates
   (`vllm_ascend/device/mxfp_compat.py:1-58` `ensure_mxfp8_*`, and MoE dispatch
   `quant_mode=4` for MXFP8 at `ops/fused_moe/token_dispatcher.py:279-284`);
   the tutorial's checkpoint is `-w8a8-mtp`, not MXFP8 — flag the difference
   instead of assuming equivalence.

Ranking language that survived review: (a) the single-block mamba move into a
whole-sequence-sized page (`mamba_cache_mode` staying `"none"`, §2 + §3 item 2) —
source-verified asymmetry, unproven here; (b) non-documented decode shape (no MTP)
with a graph-mode/padding interaction — test by row 1/3; (c) connector mismatch vs
the documented recipe — cheapest to test, and on a pin where §1's grep is non-empty
it is only a recipe deviation, never the mechanism; (d) weights/quantisation
mismatch — row 6/7. Name what each row's outcome would confirm, and say that the
reporter's "resolved" carries no mechanism.

## 5. Budget note (why this page exists)

The recorded run on this case spent the entire session on source archaeology
across four mechanisms (rank mapping, mamba mode, A5 MXFP8 dispatch, recompute
scheduler) and delivered **no answer at all** — 20 turns, the final message
reasoning-only, `stop_reason: output_tokens`. The replay that had this page read it
on its 5th assistant turn and shipped the full ranked answer at turn 11 (4.8 K
characters of message text, 13 tool calls) — the connector grep above was the one
source read that changed a rank, demoting §1's old "connector cannot describe the
hybrid page" claim to recipe row 2. The material above was already sufficient for a
complete ranked answer plus the ablation table. For an alternating-accuracy PD report, the
ranked causes + ablation matrix *is* the deliverable; further source walks are
refinement and must not delay it. The gates that enforce this are rule 1 of
SKILL.md and `../evidence-workspace/delivery-discipline.md`.
