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
v0.22.1rc1). Recorded #11877 (v0.22.1rc1 + vllm v0.22.1, 8×910B2,
Kimi-K2.6-W4A8 `ascend` quant, P TP8 / D TP2 with D `dp_size` 8, Eagle3 K=3,
hybrid KV manager on, `VLLM_COMPILE` + `FULL_DECODE_ONLY`): the paste is a full
问题分析报告 carrying its own hunks, and its proposed `MLA/sparse → 1` short-circuit is
right in direction but incomplete — Step 2 says why. Lane B, mapping phase: no KV was
lost, so neither a KV-miss, transport nor accuracy theory applies yet. Symmetric P/D TP does **not** reproduce it — a
symmetric-TP producer that dies in the *layerwise* connector's
`_get_kv_split_metadata` (no `chosen_rank_list` text in the paste at all) belongs to
`layerwise-split-metadata-crash.md`, not to this page.

## Step 1 — the arithmetic is three functions, and one of them forgot the rule

The number of P ranks a D rank must pull from for one KV group is derived three
times; two derivations apply the MLA/sparse single-pull rule and one does not:

| Derivation | Site (v0.22.1rc1) | MLA/sparse |
|---|---|---|
| `_get_tp_num_need_pulls` | `:2626-2639`; construction `:1795-1805` | `is_deepseek_mla` → `1` |
| `_get_remote_ranks_for_req` (rank selector) | `:2686-2716`, `num_kv_head = 1` at `:2701-2704` | yes |
| `_get_attention_group_num_need_pulls` (group helper) | `:2534-2538` | **no** — plain `max(1,H/D)//max(1,H/P)` |

With P TP 8 / D TP 2 the group helper returns **the TP ratio, not a head count**:
`max(1, H//2) // max(1, H//8)` is `8//2 = 4` for every `H` the prefill TP divides
(8→4, 16→4, 64→4, 128→4; only a non-multiple differs, and then it is *larger* — `H=12`
gives `6`). Two things follow before any head-count theory. The pasted
`num_group_pulls(4)` is **not** evidence about `H`, so do not adopt the reporter's own
`H=8` derivation and do not read heads off the number; and the assert trips for any model
whose KV heads are divisible by the prefill TP, MLA and GQA alike, so the defect is in the
*rule*, not in the count. The rank selector returns 1 rank for MLA, the helper returns 4,
and the assertion at `:2515` compares them. The two numbers
are one quantity on the receive side too: `group_pull.num_group_pulls` becomes
`tp_num_need_pulls` (`:689`) and slices the transfer (`:712`, `:776`
`inner_block_len = block_len // tp_num_need_pulls`). **Do not delete the assert** — with
`pulls=4` and one rank collected it reads a quarter of the block and produces silently
incomplete KV. The fix has to make the two derivations agree; which one is authoritative
is settled per wire shape in Step 2 (the single-pull rule is the one the consumer already
assumes).

## Step 2 — `H` is the HF fallback here, and the pasted 4 says the wire carries no int

`_get_attention_group_num_key_value_heads` (`:2540-2547`) reads an **int**
`num_kv_heads`/`num_key_value_heads` from the top of `group_spec["kv_cache_spec"]`, else
falls back to `self.num_key_value_heads` (`:1756` = HF
`hf_text_config.num_key_value_heads`).

What that int *would* be is not open: at this vllm pin MLA layers build their spec with
`num_kv_heads=1` (`vllm/model_executor/layers/attention/mla_attention.py:960-971`,
`vllm/model_executor/models/deepseek_v2.py:590-596`) and `MLAAttentionSpec.merge` carries
the field through (`vllm/v1/kv_cache_interface.py:371-396`), so
`max(1,1//2) == max(1,1//8) == 1`, the helper returns `1` and the assert **passes**. The
crash therefore excludes "the wire carried an int `num_kv_heads` of 1": the group spec that
arrived either carries no int head count at all — the HF fallback then supplies the 4 — or
carries one greater than 1. `_serialize_kv_group_spec` (`:1830-1862`) has exactly two
shapes on the wire, and neither delivers an int:
`UniformTypeKVCacheSpecs` serializes to `{layer_name: spec}` (`:1836-1838`; upstream class
contract `vllm/v1/kv_cache_interface.py:667-675`; the pin's own fixture shape,
`tests/ut/kv_offload/test_mooncake_connector.py:531-532`), so the int lookup misses on
layer names; a plain spec goes through `to_msgpackable`, whose
`except TypeError: return repr(value)` arm (`:1843-1844`) fires for anything msgspec cannot
encode, and a `KVCacheSpec` carries `dtype: torch.dtype`. One DEBUG of `group_spec` at
`:2513` shows which shape is live: a per-layer dict of `<…Spec object at 0x…>` strings, or
a bare repr, proves the fallback. Read that line rather than asserting the encoding.

Consequence for the fix — and the trap in a proposed patch. A *per-layer unwrap* of
`kv_cache_spec` cannot recover a head count from `repr` strings, so it changes nothing on
the shape that actually arrives; the two live options are the MLA/sparse short-circuit
(cheap, and the two other derivations already assume it) or making
`_serialize_kv_group_spec` carry an explicit int (`num_kv_heads`, or the group's pull
count) beside `kv_cache_spec_type` and consuming it. Also say what the short-circuit alone
leaves behind: a **non-MLA** hybrid group still takes its heads from the HF fallback, i.e.
the model-global number rather than the group's — right by coincidence, not by
construction. Write that, not "GQA stays correct".

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
   (`tests/ut/kv_offload/test_mooncake_connector.py`). **Grade every case the reporter
   proposes against the shape that actually reaches the helper**: a fixture of the form
   `{"kv_cache_spec": {"num_kv_heads": 8}}` is a flat int dict the serializer never
   produces (Step 2), so its expected `4` is satisfied by the fixture and pins none of the
   production path — #11877's two cases are exactly that shape, so neither discriminates.
   Passing the assertion is not evidence of correct KV either: still require one accuracy
   run against the non-disaggregated baseline.

## Scope

Verified at vllm-ascend v0.22.1rc1 `5f6faa0cb883` and vllm v0.22.1 `0decac0d96c4`
(paths above): the three pull-count derivations, the serializer's two shapes and every
`:line` here are read at those commits, and the MLA `num_kv_heads=1` constructions, their
`merge` and the `UniformTypeKVCacheSpecs` keying are read in the vllm tree at
`0decac0d96c4`. The `repr` arm is read from the serializer's `except TypeError`, not
observed (msgspec is not runnable in the review workspace; on the reporter's host it is
one probe — `python -c "import msgspec, torch; msgspec.to_builtins(<spec>)"`), so the
`:2513` DEBUG line is the observation that settles it. Line numbers move between releases and the group-pull code is new in
this series; older release trees (e.g. v0.20.2rc1) are not in this snapshot, so a
"worked before" claim is reporter-confirmed only. `EngineDeadError` on the client is
expected for a worker-side exception — "fail only that request" is a separate,
connector-side change and not a mapping tweak.
