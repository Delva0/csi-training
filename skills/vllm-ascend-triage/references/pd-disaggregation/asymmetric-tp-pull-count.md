# P/D with different TP sizes: the two pull-count derivations disagree

## When this page applies

Mooncake P/D where the two sides use **different tensor-parallel sizes** (P `TP=8`,
D `TP=2`, P `PP=1`) on an **MLA** (DeepSeek/Kimi-class) or **DSA-sparse**
(`index_topk`) model. The decode worker finds the remote blocks, then dies before any
byte moves:

```text
MooncakeConnectorWorker.start_load_kv
  … _get_kv_split_metadata / _get_group_pulls_metadata
  … _get_hybrid_remote_rank_group_pulls
AssertionError: chosen_rank_list([3]) does not match num_group_pulls(4) and prefill pp size(1)
```

(`vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py:2515-2518` at
v0.22.1rc1). Lane B, mapping phase: no KV was lost, so neither a KV-miss, transport
nor accuracy theory applies yet. Symmetric P/D TP does **not** reproduce it.

## Step 1 — the arithmetic is three functions, and one of them forgot the rule

The number of P ranks a D rank must pull from for one KV group is derived three
times; two derivations apply the MLA/sparse single-pull rule and one does not:

| Derivation | Site (v0.22.1rc1) | MLA/sparse |
|---|---|---|
| `_get_tp_num_need_pulls` | `:2626-2639`; construction `:1795-1805` | `is_deepseek_mla` → `1` |
| `_get_remote_ranks_for_req` (rank selector) | `:2686-2716`, `num_kv_head = 1` at `:2701-2704` | yes |
| `_get_attention_group_num_need_pulls` (group helper) | `:2534-2538` | **no** — plain `max(1,H/D)//max(1,H/P)` |

With kv heads `H=8`, P TP 8, D TP 2: the rank selector returns 1 rank, the group
helper returns `4//1 = 4`, and the assertion at `:2515` compares them. The two numbers
are one quantity on the receive side too: `group_pull.num_group_pulls` becomes
`tp_num_need_pulls` (`:689`) and slices the transfer (`:712`, `:776`
`inner_block_len = block_len // tp_num_need_pulls`). **Do not delete the assert** — with
`pulls=4` and one rank collected it reads a quarter of the block and produces silently
incomplete KV. The fix has to make the two derivations agree; which one is authoritative
is the open question (the single-pull rule is the one the consumer already assumes).

## Step 2 — `H` itself is ambiguous for a hybrid group

`_get_attention_group_num_key_value_heads` (`:2540-2547`) reads an **int**
`num_kv_heads`/`num_key_value_heads` from the top of `group_spec["kv_cache_spec"]`, else
falls back to the HF `hf_text_config.num_key_value_heads` (`:1756`). But a group whose
spec is `UniformTypeKVCacheSpecs` serializes to `{layer_name: spec}` 
(`_serialize_kv_group_spec`, `:1829-1866`; upstream class contract
`vllm/v1/kv_cache_interface.py:667`), so the int lookup misses and the HF number
supplies `H`. Two independent defects therefore sit on the same line of reasoning: no
MLA/sparse short-circuit, and no per-layer unwrap. One DEBUG of `group_spec` at `:2513`
records which one produced the 4 — ask for it rather than asserting the source.

## Step 3 — is the hybrid/group path even reached?

`_is_hma_required` (`:1767-1769`) = hybrid KV manager enabled **and** any group whose
spec is not `FullAttentionSpec`. `MLAAttentionSpec` **is** a `FullAttentionSpec`
subclass (upstream `v1/kv_cache_interface.py:337`), so a pure-MLA engine alone does not
trip it; a `UniformTypeKVCacheSpecs` group or a sliding-window-MLA group does. That
predicts the ablation matrix: turning **Eagle3/MTP off is not expected to clear the
crash** (MTP layers only add entries in `_build_kv_group2layeridx`, `:1871-1893`),
while `disable_hybrid_kv_cache_manager` / `--no-enable-prefix-caching` changes whether
the group path runs at all. Say that the mechanism is unproven until the log names the
group.

## Step 4 — why "it worked on the older release" is consistent

Symmetric P/D TP returns from the early exit at `:2692-2699` (one rank per port) and the
group helper divides `H/H = 1`, so the pair agrees and the group path is invisible. A
report of "same model, same topology, previous vllm-ascend release was fine" therefore
locates the change in the group-level code, not in the rank selector — but check that
the older tree has the group functions at all before calling it a regression (an older
release rarely ships in the snapshot; say which trees are absent instead of guessing).

## What to send back / what decides it

1. Both serve commands (`prefill`/`decode` `tp_size`, `dp_size`, `pp_size` in the
   `--kv-transfer-config` extra config, plus the engine flags).
2. The model's `num_key_value_heads`, and one DEBUG `group_spec` line per KV group at
   `:2513` (`kv_cache_spec_type`, `num_kv_heads`).
3. The crash rerun with Eagle3/MTP off and then with the hybrid KV manager disabled —
   rows that keep the crash alive move the cause to the group helper's head count.
4. For a proposed patch: a CPU unit case asserting the group helper and
   `_get_remote_ranks_for_req` return the same count for MLA and for GQA at the same
   P/D TP pair, next to the existing `test_get_tp_num_need_pulls`
   (`tests/ut/kv_offload/test_mooncake_connector.py`). Passing the assertion is not
   evidence of correct KV: still require one accuracy run against the
   non-disaggregated baseline.

## Scope

Verified at vllm-ascend v0.22.1rc1 `5f6faa0cb883` and vllm v0.22.1 `0decac0d96c4`
(paths above). Line numbers move between releases and the group-pull code is new in
this series; older release trees (e.g. v0.20.2rc1) are not in this snapshot, so a
"worked before" claim is reporter-confirmed only. `EngineDeadError` on the client is
expected for a worker-side exception — "fail only that request" is a separate,
connector-side change and not a mapping tweak.
