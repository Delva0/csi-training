# C8 sparse-attention flags and the LI C8 KV-cache write path

## Scope

A sparse-attention (SFA / LI C8) deployment whose **prefill side stops answering** or
answers garbage on the first request, with `enable_sparse_sfa_c8` /
`enable_sparse_li_c8` / `c8_enable_reshape_optim` inside `--additional-config`:
GLM-5.1 / 5.2 / 5.3 (`w8a8c8`, `w4a8c8`) and the DeepSeek-V3.2 DSA family. Verified at
**v0.26.0rc1 `f2f74a1`** (SOURCES `15927`; upstream vllm v0.26.0 `568afb3`). The
reporter's build is 0.26.0rc1 and the same shape was filed at 0.23.0rc1 (#12533), so
both refs are in scope; re-grep the anchors when the pin differs.

Not this page: a single-request device fault naming a scatter/gather kernel
(`indexer-cache-scatter-fault.md`), a concurrency crash with no log
(`concurrency-crash-with-no-log.md`), an engine-start `AttributeError`
(`o-proj-quant-attribute-crash.md`).

## Rule 1: three flags, one dependency — read the algebra before the ablation

`vllm_ascend/ascend_config.py:286-290`:

- `enable_sparse_sfa_c8` = `additional_config.get(...) and use_sparse`
- `enable_sparse_li_c8`  = `additional_config.get(...) and use_sparse`
- `c8_enable_reshape_optim` = **`self.enable_sparse_li_c8` and** `additional_config.get(...)`

State these consequences in the answer, because they are what the reporter's own
ablation ("`c8_enable_reshape_optim` = false and it serves normally") actually proves:

- `c8_enable_reshape_optim: true` **without** `enable_sparse_li_c8` is a silent no-op —
  no error, no warning.
- All three collapse to `false` when `model_uses_sfa_sparse(model)` (`utils.py:111`) is
  false, so "I enabled it and nothing changed" can be a model question, not a flag one.
- Setting `c8_enable_reshape_optim: false` does **not** disable LI C8. It switches only
  *how the indexer cache is written*, which is exactly why it is the single-variable
  ablation to ask for.

## Rule 2: the flag selects between two implementations of the same write

`sfa_v1.py:1547-1549` (`_use_li_c8_reshape_optim`), used by the forward branch at
`sfa_v1.py:1983-2016`:

| flag true | flag false (fallback) |
|---|---|
| `torch.ops._C_ascend.store_kv_block(k_li[_scale], kv_cache[idx], group_len, group_key_idx, group_key_cache_idx, block_size)` (`:1988`, `:2006`) | `torch_npu.npu_scatter_nd_update_(kv_cache[idx].view(-1, d), slot_mapping.view(-1, 1), k.view(-1, d))` (`:1997`, `:2015`) |

`_build` additionally launches `torch.ops._C_ascend.store_kv_block_metadata(slot_mapping,
group_len, group_key_idx, group_key_cache_idx, block_size)` (`sfa_v1.py:460-467`), i.e.
**once per layer per step**, on every worker that builds metadata — including the
prefill node.

Both are in-tree custom ops (`csrc/attention/store_kv_block/`,
`csrc/attention/store_kv_block_metadata/`). The metadata one is an **AICPU kernel that
reads `slot_mapping` from device memory**; its outputs are pre-allocated at
`slot_mapping` length and the consumer recovers the group count by scanning for the
**first zero `group_len`** — see the header comment in
`csrc/attention/store_kv_block_metadata/store_kv_block_metadata_torch_adpt.h`. So a hang
on this path is a device/AICPU-side wait or an index-space mismatch between the
slot-derived groups and the rows actually written; it is never a connector, rank-table
or network fault, and asking for network evidence first costs the session.

## Symptoms, ranked, one check each

1. **P worker stops answering with all three flags true.** Curl to the P port never
   returns. Reporter-confirmed #15927 (GLM-5.3-w8a8c8, 2×Atlas 800 A3, TP16 × PP2,
   `MultiConnector` = `MooncakeConnectorV1` + `AscendStoreConnector`/memcache,
   `--enforce-eager`, DCP on): `c8_enable_reshape_optim: false` restores normal serving.
   Precedent #12533 (same shape at 0.23.0rc1, C8 on both P and D, "关掉C8就可以正常推理")
   shows how the hang surfaces at the client boundary: the API server prints only
   `EngineDeadError`, while the worker log carries
   `TimeoutError: RPC call to sample_tokens timed out.` at `core.py:1197`. That is the
   RPC watchdog over a stuck worker — ask for the **worker/EngineCore** log.
2. **First curl garbled, later ones clean, `enable_sparse_sfa_c8` on.** Precedent #12905
   (v0.23.0rc1, A3 4-machine PD, GLM-5.2, DCP=8). First-request-only corruption is a
   warm-up/init effect, not transport: `../accuracy-benchmarks/garbled-long-context-output.md`
   (its ablation table is keyed on exactly this flag).
3. **`MTE invalid GM address` / `LightningIndexerQuant` with a `block_table` longer than
   the k_cache** under DCP — #14320, an index-space overflow:
   `indexer-cache-scatter-fault.md`.
4. **No hang, only accuracy or throughput regression** with `sfa_c8` + `li_c8` — #14329
   (GLM-5.1 PD, 64K-in/1K-out), and the project's own note that GLM-5.2-W4A8C8 PD with
   `enable_sparse_sfa_c8` moves GPQA by ~2-3 points run to run (`release_notes.md:270`,
   #14378). Not a hang hunt.

## Rule 3: quote the pinned recipe for the reporter's *weights*

`docs/source/tutorials/models/GLM5.2.md` at the pin ships two different C8
combinations, and they are not interchangeable:

| recipe | lines | flags |
|---|---|---|
| 5.1.2 multi-node, "A3 series", **GLM-5.2-w4a8c8**, 2 nodes | `:211`, `:261` | `enable_sparse_sfa_c8: false`, `enable_sparse_li_c8: true`, `c8_enable_reshape_optim: false` |
| 5.1.3.1 "Deployment on 4 Atlas 800 A3", **GLM-5.2-w8a8c8**, TP16 × PP2, `max-model-len 202752` | `:505`, `:579` | all three `true` |

Support matrix (`:1817-1818`): `enable_sparse_sfa_c8` is *experimental in v0.23.0*,
combinable with DCP on **w8a8c8** weights, while on **w4a8c8** "enabling both together
has known issues and is not recommended"; `enable_sparse_li_c8` is documented as
independent of it. Release notes at the pin: DCP must be enabled on P and D or on
neither, asymmetric DCP causes accuracy issues (`:260`, #14322); GLM-5.2 DCP ×
`enable_sparse_sfa_c8` has "known issues, including performance degradation, and is not
recommended" (`:262`, #14011); GLM-5.1 sfa_c8+li_c8 64K/1K regression (`:268`, #14329).

## Delivery order for this class

1. Verdict line: runtime hang inside the SFA/LI C8 KV-write path, not transport; the
   flag the reporter's own ablation moved is the one to turn off.
2. The flag diff against the pinned recipe for their weights (w8a8c8 four-node vs
   w4a8c8 two-node) plus "DCP identical on P and D".
3. Two ablations, cheapest first: `c8_enable_reshape_optim: false` (write path only),
   then `enable_sparse_sfa_c8: false` with `enable_sparse_li_c8: true` (the w4a8c8
   recipe's own combination).
4. Open question, and say it is open: the snapshot carries **no fix**, and nothing in
   the pin decides whether the hang is the AICPU metadata kernel itself or a mismatch
   between the `slot_mapping`-derived groups and the rows written under
   `enable_dsa_cp`. A P-worker plog of the last launched op settles which.
