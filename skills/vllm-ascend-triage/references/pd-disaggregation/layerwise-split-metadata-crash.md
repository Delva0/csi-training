# Layerwise PD producer: the P worker dies inside `_get_kv_split_metadata`

## When this page applies

The **prefiller** (`kv_role == "kv_producer"`, `MooncakeLayerwiseConnector`) kills its
own worker during a run, the decode side stays alive, and the traceback is **cut at a
call line** with no exception class and no inner frames:

```text
kv_connector_model_runner_mixin.py … kv_connector.start_load_kv(get_forward_context())
mooncake_layerwise_connector.py:769  in start_load_kv
    self.connector_worker.start_load_kv(self._connector_metadata)
mooncake_layerwise_connector.py:1627 in start_load_kv
    single_group_transfer_mappings = self._get_kv_split_metada        <- paste ends here
```

`start_load_kv` is called on **both** roles by the upstream mixin, so a producer
crashing in `start_load_kv` is ordinary plumbing, not a mislabelled node. Typical
report: hybrid attn+mamba model (Qwen3.5/3.6), MTP spec decode on both sides,
chunked prefill (`--max-num-batched-tokens` far below the prompt), 20 k prompts,
concurrency ramp — physical TP/DP **the same on both sides**, which is what keeps
`asymmetric-tp-pull-count.md` out of scope (that page needs P TP ≠ D TP and a
different, non-layerwise connector).

## Rule 1 — a traceback cut at a call line is answered by a throw-site inventory, not a mechanism

The exception text is not in the paste and **cannot be recovered from the source**: the
call reaches thirteen throw sites over two files and four exception types
(`AssertionError`, `ValueError`, `KeyError`, `IndexError`). So the deliverable is the
inventory below, the one log line that prints the inputs, and one ablation row per axis
— owed as soon as the inventory is read (2–3 calls), not after a mechanism is guessed.
Recorded #15394 (v0.23.0-a3, A3 双机,
Qwen3.5-122B-A10B-w8a8-mtp) walked the whole inventory and then spent the rest of the
budget on the `chunk_finish`/`to_trans_idx` arithmetic, shipping no answer at all.
A later session of the same issue read this page, confirmed `:1627` at the pin with one
`sed`, and delivered the inventory plus the ladder in nine turns; its one wasted call was
guessing `…/kv_transfer/kv_p2p/utils/utils.py` for the throw-site list, so Rule 3 now
spells the path and the verification command.

## Rule 2 — identify the exact line first; one character of the paste decides the group type

At v0.23.0 `5cb98caaadef`:

| Line | What it is |
|---|---|
| `mooncake_layerwise_connector.py:1623-1625` | `_get_kv_split_metadata_for_mamba(...)` — the **MambaSpec** group call |
| `:1627` | `_get_kv_split_metadata(...)` — the full-attention group call (a paste ending `…_get_kv_split_metada` that fits on one line is this one) |
| `:1644` | `assert len(transfer_mappings) <= 1, f"Not support add mutil transfer task for req_id:{req_id}"` — **after** the loop, so a traceback pointing at 1627 is not it |
| `:1441` | `def _get_kv_split_metadata(...)` — the frames the paste cut off live inside `:1441-1550`, then in `get_transfer_mappings` (`utils.py:266-313`, called at `:1535`), so a continuation naming `utils.py` is this branch, not the mamba one |

## Rule 3 — the throw sites (v0.23.0 `5cb98caaadef`)

The load-bearing file is `vllm_ascend/distributed/kv_transfer/utils/utils.py` — a
**sibling of `kv_p2p/`**, not `kv_p2p/utils/utils.py` (imported at
`mooncake_layerwise_connector.py:59-66`); the wrong path returns nothing and reads as
"no such site". Only the head-arithmetic sites carry `raise`/`assert`: the block-count
family (`:283`, `:289`, `:290`, `:291`, `:294`, `:304`) is bare subscripting, so
`rg -n 'assert|raise' utils.py` **cannot confirm it** and reads as if those sites did not
exist. Confirm the first group with that grep, then read `utils.py:266-313` for the second.

| Site | Raised as | Has to be true |
|---|---|---|
| `…layerwise_connector.py:1496` | bare `AssertionError` | `len(selected_p_cp_groups) != len(selected_d_cp_groups)` — needs the two sides' cp-group derivations to differ |
| `:1553` (mamba group) | `AssertionError: Mamba group prefill TP_size must equal or larger than decode TP_size.` | P TP < D TP |
| `kv_transfer/utils/utils.py:96` | bare `AssertionError` | `(p pcp*dcp) % (remote pcp*dcp) != 0` |
| `utils.py:98-100` | bare `AssertionError` | `not use_mla` **and** `ceil(H/remote_dcp) % ceil(H/p_tp) != 0` |
| `utils.py:108` / `:118` | `ValueError: Number of heads (H) cannot be evenly divided by TP (N).` | TP and kv-head count not in a multiple/divisor relation |
| `utils.py:129` | `ValueError: Total number of devices (N) cannot be divided by the number of groups (n).` | head-group count does not divide TP |
| `utils.py:283` (implicit) | `IndexError` | `p_rank_block_mapping[pcp_rank][p_head_group_rank][dcp_rank]` overshoots its preallocated `[pcp_size][tp_size//dcp_size][dcp_size]` (`utils.py:165-168`) — the request's pcp/dcp ranks and this worker's `(tp_rank - dcp_rank) // dcp_size` (`:282`) disagree on the geometry. An inner list that was never filled is **empty**, not missing: that path yields an empty send set, not an exception |
| `utils.py:289` (implicit) | `KeyError` | `pd_head_mapping[p_head_group_rank]` — keys are only the P head-group ranks in `selected_p_cp_group` (`utils.py:169-174` fills `if p_head_group_rank in selected_p_cp_group`; same list at `:227-229`), and that list is a `req_idx % cp_ratio` slice (`connector:1487-1496`) — **needs CP on**, since `cp_ratio == 0` selects every group (`:1488-1490`). With CP on, a request whose slice does not contain this worker's head-group rank raises here **before** any block id is compared, and `:1496`'s assert compares lengths only, so it passes |
| `utils.py:290` | `IndexError` | `p_block_ids[p_block_idx]` — this rank's allocated block list for the group is shorter than the block indices it must send |
| `utils.py:291-294` | `KeyError` / `IndexError` | `d_block_rank_mapping[logic_block_idx][d_head_group_rank]`, or `d_block_ids[d_block_idx]` — P and D disagree on the **number of blocks per group** |
| `utils.py:304` | `KeyError` | `d_trans_count_mapping[(host, port)]` — the port/host key the send set chose was never built |

Two consequences worth stating in the answer instead of re-deriving them:

- **Bare asserts are the only candidates that produce a message-less abort**, and the
  paste's cut line hides which one. Of the bare ones, `utils.py:96` and `:98-100`
  are parallel-shape checks that pass for a symmetric 16/16 deployment with
  P TP ≥ kv heads, so on those reports the **block-count family** (`utils.py:283-304`)
  outranks the head arithmetic. Check the two inputs rather than assuming:
  the model's `num_key_value_heads`, and `remote_tp_size` — which arrives in
  `kv_transfer_params` from the peer (`req_meta.remote_tp_size`, read at
  `mooncake_layerwise_connector.py:1444`; the producer fills it from its own
  `parallel_config.tensor_parallel_size` at `:968`, the consumer receives it at `:684`),
  **not** from `--kv-transfer-config`. The `prefill`/`decode` `tp_size`/`dp_size` blocks
  in that config are read only by `mooncake_connector.py:2066-2091` and
  `mooncake_hybrid_connector.py:1584-1604`, and `use_layerwise` only by the pool
  connector (`ascend_store_connector.py:83-92`), so on a layerwise report both are
  inert: asking the reporter to correct `decode.tp_size`, or reading their
  `use_layerwise: false` as a contradiction with the connector's name (#15394 carries
  both), is a dead ask. Do confirm each side's `--tensor-parallel-size`, because D's
  derivation uses the number P actually sent.
- `…_get_kv_split_metadata` **mutates** `req_meta.chunk_finish = True` at `:1479-1480`
  when `num_local_blocks == already_send_blocks`, so `chunk_finish` is not simply
  `computed >= prompt_len`. Do not build a mechanism on that inequality.

## Rule 4 — the decisive artefact is already a DEBUG line in the code

`VLLM_LOGGING_LEVEL=DEBUG` on the P node prints exactly the inputs of the failing
function and the mappings it builds:

| Log | Fields |
|---|---|
| `build_connector_meta` (`…layerwise_connector.py:1066-1079`) | `prompt_len`, `local_computed_tokens`, `local_transed_tokens`, `remote_cache_tokens`, `chunk_finish`, `local_block_ids`, `remote_block_ids` |
| `:1485`, `:1508-1514` | `p_cp_group`, `d_cp_group`, and the selected P/D head-group |
| `utils.py:253-262`, `:305` | logic-block → rank mappings, per-head-group P→D mapping, final transfer tasks |

Ask for those lines for the crashed `req_id` plus the ten traceback lines **below** the
cut (exception class + message + the frames inside `_get_kv_split_metadata` /
`utils.get_transfer_mappings`). That request is the answer's action rung 1; nothing
else in the snapshot substitutes for it.

## Rule 5 — the hybrid/MTP block-count contract, and the ablations

On an attn+mamba hybrid the producer is not a passive sender: `need_truncate =
self._has_attn_mamba_hybrid_cache(kv_cache_config)` (`:819`) makes P **truncate the
prompt by one token** for the MTP hybrid-prefill path (`_hybrid_prefill_token_count`
`:854`, `_truncate_request_for_hybrid_prefill` `:859`, propagated as
`params["_p_side_truncated"]` and applied at `:923`), while D trims its `remote_block_ids`
by the same rule
(`_trim_hybrid_remote_block_ids` `:881-899`, applied at `:939` when
`prompt_len % block_size == 1`). Both sides must therefore agree on blocks-per-group,
and any P/D asymmetry moves that agreement — including an `enforce_eager` inside
`--speculative-config` on one side only, or MTP on one side only. One axis per launch:

1. `--max-num-batched-tokens` ≥ prompt on P (single chunk) with the same concurrency —
   if the crash disappears, the `chunk_finish`/`to_trans_idx`/block-id path owns it.
2. The **identical** `--speculative-config` string on both sides, then MTP off on both —
   if the crash disappears with MTP off, the truncation/trim contract owns it.
3. `--v 1`/`VLLM_LOGGING_LEVEL=DEBUG` as above (rule 4) — this one is mandatory, not an
   alternative.

## Scope and recheck

Source-verified at vllm-ascend v0.23.0 `5cb98caaadef` (vllm v0.23.0 `0fc695fc6d1d`), the
ref the reported image runs (`…:1627` lands on the `_get_kv_split_metadata` call in that
tree). The **cause of #15394 is left open** — the paste carries no exception text and
the case view has no reporter follow-up; do not let a ranked list turn into a claimed
mechanism without the DEBUG lines. Recheck: if line 1627 no longer lands on a
`_get_kv_split_metadata` call, the ref has moved and the inventory must be re-read
(`rg -n '_get_kv_split_metadata|_get_kv_split_metadata_for_mamba' vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py`).
