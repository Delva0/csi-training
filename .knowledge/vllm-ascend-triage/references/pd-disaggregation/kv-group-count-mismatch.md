# Decode-side `IndexError: list index out of range` in `_get_kernel_block_ids`

## When this page applies

The p2p Mooncake connector (not the layerwise one) is in `--kv-transfer-config`
and the decode worker — on the **first** request, or after the deployment has been
serving — raises a traceback ending

```
File ".../vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py", line 2838, in _get_kv_split_metadata
    local_kernel_block_ids, remote_kernel_block_ids = self._get_kernel_block_ids(
File ".../mooncake_connector.py", line 2740, in _get_kernel_block_ids
    kernel_remote = self._expand_block_ids(list(meta.remote_block_ids[kv_cache_group_id]), remote_scale)
IndexError: list index out of range
```

This is a **host-side Python index on a metadata list**, not a transport, device or
rank-table failure: no plog, no `npu-smi`, no HCCL read belongs in the answer. The
line numbers are v0.26.0rc1 (`f2f74a16`); they drift, the function names do not.
Recorded #16177 (Kimi-K3 + DSpark, A3 8-node PD, "仅关闭 prefill 节点 dspark 投机"),
where the reporter's own follow-up confirmed the cause: *"kv group 不一致导致的，
decode 与 prefill 需要保持同步开启投机"*.

## Step 1 — read the index: local group id against a remote list

| Anchor (v0.26.0rc1 `f2f74a16`) | Fact |
|---|---|
| `mooncake_connector.py:2836-2844` | `remote_block_ids = [[] for _ in meta.remote_block_ids]`, then the loop walks **local** `self.kv_group2layeridx` and writes `remote_block_ids[kv_cache_group_id]` |
| `:2713-2716` + `:2740` | `kv_cache_group_id = group_spec.get("kv_cache_group_id", group_idx)`; `:2740` (attention) and the MambaSpec sibling `:2727` both index `meta.remote_block_ids[kv_cache_group_id]` with **no length check** |
| `:2278-2340` (`_build_kv_group2layeridx`) | the map is built from the **local** `self.kv_cache_config.kv_cache_groups`; layer names containing `"mtp"`/`"eagle"` get ids from `total_layers` upward, and groups are split per KV spec ("Split KV cache manager group %d into %d Mooncake transfer groups by KV spec") |
| `:1989-1993` (producer) → `:583-590` → `:1541` (consumer) | `remote_block_ids` is built on the **prefiller** from its own block table and shipped in `kv_transfer_params`; the decode side stores it verbatim in `ReqMeta` |

So the exception is exactly *local group count > remote group count*. The reverse
asymmetry (remote has more groups) does **not** raise — the extra remote entries are
ignored — which makes the direction itself part of the diagnosis and makes "it
crashes, so the connector is broken" the wrong reading.

## Step 2 — the one check: diff the two `--speculative-config` blocks

A drafter configured on one side only changes that side's group set, because the
draft layers carry their own KV specs: `_build_kv_group2layeridx` special-cases
them (`:2295`), upstream `GPUModelRunner.get_kv_cache_spec()` walks every
attention layer in the config, and `vllm/v1/spec_decode/llm_base_proposer.py`
registers the loaded drafter's `_draft_attn_layer_names` as layers with KV cache.
Decode with DSpark on and prefill with it off therefore gives the decode side the
drafter's group(s) on top of the target's — counts differ by exactly those groups (2
against 1 in the recorded case) — and the loop's extra local group indexes an empty
remote tuple.

The pinned tutorial states the requirement in words, which is why this is a config
answer before a code answer — `docs/source/tutorials/models/Kimi-K3.md` §5.2
(`:560`) "Configure the same `Inferact/Kimi-K3-DSpark` draft-model path and
`num_speculative_tokens` on both Prefill and Decode nodes" and the FAQ (`:886`)
"set `DRAFT_MODEL_PATH` on both Prefill and Decode nodes, and pass the same
`--speculative-config` to both. Prefill must retain `--enforce-eager`".

The mismatch is **silent at INFO**: the handshake compares the two maps and logs at
DEBUG only (`:1427-1434`, "Remote kv_group2layeridx differs from local … remote=%s,
local=%s", no raise), so nothing in a default-level paste hints at it.
`VLLM_LOGGING_LEVEL=DEBUG` on either side prints both maps — that is the 100 %
confirmation and costs one restart.

Same asymmetry, different face: matching *counts* with differing *geometry* (other
`num_speculative_tokens`, other draft model) does not raise here; it corrupts the
decode output instead (`recipe-contract-diff.md` invariant 2,
`hybrid-state-transfer.md` §3). Do not expect one symptom to cover both.

A third face of the same one-sided-spec class does not touch the connector at all:
on a launch with **structured output** as well, the decode node's EngineCore dies on
`assert accepted` with `-1` scheduled draft ids (`FAILED to advance FSM … tokens -1`),
and #6421 was closed by the same action — "inconsistent configuration between
prefilling and decoding nodes", key not disclosed — while the pin also ships a
source-verified seeder for those ids (`RecomputeScheduler.add_request`). If the paste
is a grammar assert rather than a `_get_kernel_block_ids` `IndexError`, read
`structured-output-draft-placeholder-assert.md`: the parity diff is still row 1, but
one ablation (`recompute_scheduler_enable` off on both sides) has to come with it.

## Step 3 — ranked causes

1. **`--speculative-config` present on one side only** (reporter-confirmed #16177).
   Confirm by diffing the two serve commands, not by reading the connector.
2. **Different draft configs on the two sides** (same group count is not the same
   geometry) — silent corruption rather than this `IndexError`; check
   `num_speculative_tokens`, draft path, `draft_tensor_parallel_size` while the two
   commands are already open.
3. **Group-layout flags that differ between the sides** — hypothesis, one check:
   `--no-disable-hybrid-kv-cache-manager` (HMA) splits groups on a hybrid family, so
   enabling it on one side alone can move the count; the DEBUG line above settles it.
4. **Peer image/version skew** — hypothesis: a peer built from another commit can
   carry a different group layout. Check `vllm --version` / image tag on both sides
   (`../startup-failures/environment-correspondence.md`) before any code theory.
5. **A missing guard in the connector** — real gap, not the reporter's root cause:
   `_get_kernel_block_ids` would be a two-line assert
   (`kv_cache_group_id < len(meta.remote_block_ids)` plus both counts) and the
   handshake DEBUG should be a WARNING. Worth a separate issue; do not hand it back
   as the fix.

## Step 4 — ladder to hand back

1. Restore on the prefiller the **same** `--speculative-config` the decode node runs
   (draft path, `num_speculative_tokens`, `draft_tensor_parallel_size`,
   `draft_sample_method`), keeping the tutorial's `--enforce-eager` on prefill — the
   documented 4P+4D layout. Expected: the `IndexError` disappears.
2. Or drop `--speculative-config` from **both** sides: a pure-target PD deployment.
   Never "spec on decode only" — that is the shape in the paste.
3. `VLLM_LOGGING_LEVEL=DEBUG` and compare `kv_group2layeridx=…` on P and D (Step 2);
   paste both lines with the answer so the ranking becomes a fact.
4. If the configs are already identical and it still raises, collect
   `kv_cache_config.kv_cache_groups`, `block_size` and `remote_block_size` per side —
   the neighbouring non-`IndexError` trigger on this path is the
   `assert remote_block_size % kernel_size == 0` at `:2734` (the group-params
   sibling is `:2760`) — then treat it as a
   connector defect with #16177 as the counter-example.

## What to ask for, and the neighbouring page

Both serve commands verbatim (the `--speculative-config` and `--kv-transfer-config`
blocks), per-side `vllm --version`/image tag, and the DEBUG group maps. Nothing about
plog, HCCL or rank tables applies until the index is explained.

The **layerwise** connector (`mooncake_layerwise_connector.py`) cuts in
`_get_kv_split_metadata` with no exception text and a different throw-site
inventory — that paste is `layerwise-split-metadata-crash.md`; a symmetric-TP pull
count assert is `asymmetric-tp-pull-count.md`.
