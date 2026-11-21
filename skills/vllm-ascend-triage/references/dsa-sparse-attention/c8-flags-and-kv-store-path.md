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
   (v0.23.0rc1, A3 4-machine PD, GLM-5.2, DCP=8, `MultiConnector` = `MooncakeConnectorV1`
   + `AscendStoreConnector`/memcache). **The reporter's own control is not one yet:** the
   report does not say whether the second curl used a different prompt, neither the
   tutorial block nor their launch sets `--no-enable-prefix-caching`, and a KV pool was
   attached — so if it was the same prompt, the clean second answer is a prefix/pool hit
   that never re-runs the prefill write path. Row 0 of the
   ladder is therefore **re-send a fresh prompt (or add a nonce)**, before any warm-up
   story; after that this row is `../accuracy-benchmarks/garbled-long-context-output.md`
   (its grade table and ablation ladder are keyed on exactly this flag).
3. **`MTE invalid GM address` / `0x800000` in a KV-reading kernel** under
   `--decode-context-parallel-size` > 1, `retCode=0x26 [aicore exception]` + `507015`
   with `fault kernel_name=LightningIndexerQuant_…` (#14320, reporter-confirmed: the
   DCP replicated `block_table` ids exceed the k_cache capacity) **or**
   `retCode=0x31 [vector core exception]` + `507035` with the
   `============ tiling入口 ============[FILE:kv_quant_sparse_flash_attention_tiling.cpp]`
   marker (#15906), on the first request or after hundreds — `dcp-c8-kv-address-fault.md`.
   Its arrival by way of a benchmark client's `ConnectionResetError` is the same page's
   Rule 1.
4. **No hang, only accuracy or throughput regression** with `sfa_c8` + `li_c8` — #14329
   (GLM-5.1 PD, 64K-in/1K-out), and the project's own note that GLM-5.2-W4A8C8 PD with
   `enable_sparse_sfa_c8` moves GPQA by ~2-3 points run to run (`release_notes.md:270`,
   #14378). Not a hang hunt.

## Rule 3: quote the pinned recipe for the reporter's *weights*

`docs/source/tutorials/models/GLM5.2.md` ships four different C8 combinations across
its layouts, and they are not interchangeable:

| recipe | lines | flags |
|---|---|---|
| 5.1.2 multi-node, "A3 series", **GLM-5.2-w4a8c8**, 2 nodes | `:211`, `:261` | `enable_sparse_sfa_c8: false`, `enable_sparse_li_c8: true`, `c8_enable_reshape_optim: false` |
| 5.1.3.1 "Deployment on 4 Atlas 800 A3", **GLM-5.2-w8a8c8**, TP16 × PP2, `max-model-len 202752` | `:505`, `:579` | all three `true` |
| at v0.23.0rc1 `f4a08bd`, the **8-node Atlas 800 A2 (910B)** PD block, "Deployment on 8 Atlas 800 A2" | `:982-1212`; flags `:1059-1060` (P), `:1142-1143` (D); prose `:984` | **both** `enable_sparse_sfa_c8: false`, `enable_sparse_li_c8: false`, and **no DCP**; connector `MooncakeConnector` + `kv_connector_module_path` (`:1040,1121`), decode `DP8 TP4`, P/D MTP `1`/`3`, mml `115168`/`135168` |
| at the same pin, the **1M PD** block (A3-class context) | `:1365` (P), `:1436` (D) | both C8 flags `true`; `enable_flashcomm1`/`enable_dsa_cp` true on P and false on D; `recompute_scheduler_enable: true` on **both** (re-verified at `f4a08bd` — the P block is `--enforce-eager` with no `--compilation-config`, D carries `cudagraph_mode: FULL_DECODE_ONLY`); connector `MooncakeConnectorV1` **alone** (`:1371`, `:1442`) — `AscendStoreConnector`/memcache appears nowhere in the file, so adding it is the deviation to rank, not a recipe row |

**The C8 flags are a per-SoC, per-topology recipe row, not a project default.** The
same model's tutorial carries `false/false` for the 910B 8-node layout and
`true/true` for the A3 4-node and 1M layouts, so "both sides match" is not the
check: match the reporter's whole `--additional-config` key lattice against each
block (the recorded #14085 deployment — 910B, `max-model-len 200000` — carried the
1M block's lattice verbatim, C8 `true` and DCP 8 included). Say which block the
command was copied from, then rank the deviating rows. A hand-written
`cudagraph_capture_sizes` entry that is not a multiple of `num_speculative_tokens + 1`
is not one of them (round-up, `../pd-disaggregation/recipe-contract-diff.md`).

Release notes at the same pin also scope the **1M-context** rows: `:259`
"GLM-5.2 1M-context deployments were validated only on Atlas 800 A3; the A2 series was
not validated for 1M context" — so a 1M `--max-model-len` command on 910B/A2 copied from
the A3 1M block is off-recipe before any mechanism (recorded #15906, whose 1M + DCP8 + C8
command faulted on the first request: `dcp-c8-kv-address-fault.md`).

Support matrix at the **v0.26.0rc1** pin (`:1817-1818`): `enable_sparse_sfa_c8` is
*experimental in v0.23.0*, combinable with DCP on **w8a8c8** weights, while on
**w4a8c8** "enabling both together has known issues and is not recommended";
`enable_sparse_li_c8` is documented as independent of it. That row is **absent from
the v0.23.0rc1 matrix** (checked for #14085: `rg -n 'enable_sparse_sfa_c8'
docs/source/user_guide/support_matrix/supported_models.md` returns nothing at
`f4a08bd`), so on the rc1 pin quote the tutorial blocks and the precedents rather than
this matrix — and say which pin each statement came from. Release notes at the pin: DCP must be enabled on P and D or on
neither, asymmetric DCP causes accuracy issues (`:260`, #14322); GLM-5.2 DCP ×
`enable_sparse_sfa_c8` has "known issues, including performance degradation, and is not
recommended" (`:262`, #14011); GLM-5.1 sfa_c8+li_c8 64K/1K regression (`:268`, #14329).

## Not this page

A launch that never reaches an inference step and dies in
`_reshape_kv_cache_tensors` on `assert sum_page_size_bytes % … page_size_bytes == 0`
is the KV-page arithmetic, not the C8 write path —
`c8-page-size-reshape-assertion.md` (it also carries the `ls` that tells the two
spec contracts apart).

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
