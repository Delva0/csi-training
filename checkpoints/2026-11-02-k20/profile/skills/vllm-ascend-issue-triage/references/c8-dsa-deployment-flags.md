# GLM-5.x sparse C8, DSA-CP and KV-pool flags (5.2 flag cluster, 5.3-Flash pool)

Flag cluster seen in GLM-5.2 / DeepSeek-sparse PD deployments:
`enable_sparse_sfa_c8`, `enable_sparse_li_c8`, `c8_enable_reshape_optim`, `enable_dsa_cp`,
`enable_sparse_c8`, plus KV connectors (`MultiConnector` of `MooncakeConnectorV1` +
`AscendStoreConnector` with `backend: memcache` or `mooncake`) and
`enable_hybrid_kv_cache_manager`.

## Where the flags are defined (vllm-ascend v0.23.0rc1)

Note that the same deployment block carries `recompute_scheduler_enable`, which is a
different class of switch: it changes scheduler and proxy behaviour on PD decode nodes (see
[pd-proxy-recompute-retries.md](pd-proxy-recompute-retries.md)) and gates
`finegrained_tp_config` above; do not fold it into a C8 analysis.

- `vllm_ascend/ascend_config.py`: `c8_enable_reshape_optim` is derived, not raw —
  `self.c8_enable_reshape_optim = self.enable_sparse_li_c8 and additional_config.get("c8_enable_reshape_optim", False)`.
  Setting it without `enable_sparse_li_c8` silently does nothing.
- `docs/source/user_guide/configuration/additional_config.md` documents the trio:
  `enable_sparse_sfa_c8` = packed C8 KV cache for SFA in DSA models; `enable_sparse_li_c8` =
  C8 key/scale caches for LightningIndexer; `c8_enable_reshape_optim` = use the StoreKVBlock
  operator for LightningIndexer C8 cache writes, "In the PD separation scenario, only the P
  node is enabled". Both C8 flags note "SFA prefill context parallelism and Ascend 950 DCP are
  not supported" — check that against the report's `enable_dsa_cp`, whose `enable_sp()`/EP
  gating at v0.18.0rc1 is traced in
  [context-parallel-moe-paths.md](context-parallel-moe-paths.md).

## The C8 path is missing from `v0.17.0rc1` although that tag's release notes advertise it

`docs/source/user_guide/release_notes.md:20` at `v0.17.0rc1` reads
"W8A8C8 quantization is now supported for DeepSeek-V3.2 and GLM5 in PD-mix scenario. [#7029]" — but
the tree has no C8 op: `git ls-tree -r --name-only v0.17.0rc1 | grep -c lightning_indexer_quant`
is 0 (13 files at `v0.18.0rc1`), and `enable_sparse_c8` greps clean in both `vllm_ascend/` and
`configuration/additional_config.md`. The commit chain explains it: `7ed9e9de6` (#7029) added the
op, `0c299f79b` (#7288) reverted it ("the patch doesn't work with recompute scheduler enabled") and
**the revert is contained in `v0.17.0rc1`**, and `4d443b922` (#7294, "[bugfix] restore pr-7029 and
fix patch error") restored it with that interaction fixed plus the user-facing switch —
`enable_sparse_c8` in `additional_config` (`ascend_config.py:156` at `v0.18.0rc1`, docs
`additional_config.md:47`, default `False`, "Not supported on A5 devices now"). `git tag --contains
4d443b922` starts at `v0.18.0rc1`; it is not in `v0.17.0rc1`.

So a `v0.17.0rc1` report asking for C8/LI-C8 in PD-mix is answered by the version boundary, not by
a mechanism: neither the kernels nor the switch exist at the reporter's tag, and the release note in
that same tag is stale. The revert reason is a fact about the C8 patch's history, not a licence to
fold `recompute_scheduler_enable` into a C8 analysis (see the note at the top of this file).

## Reported failing config = the published tutorial config

Issue #12533's failing P-node config
`{"recompute_scheduler_enable": false, "enable_dsa_cp": true, "enable_sparse_sfa_c8": true,
"enable_sparse_li_c8": true, "c8_enable_reshape_optim": true}` is byte-identical to
`docs/source/tutorials/models/GLM5.2.md:683` at tag `v0.23.0rc1`. A tutorial match means the
reporter followed the docs, so the finding is a documented-path bug (or a stale doc), not
misconfiguration. Verify the same way for any report that quotes `--additional-config`.

Issue #15927 repeats this at `v0.26.0rc1`: that tag's P-node block (`GLM5.2.md:505`) carries the same
`enable_dsa_cp` / `enable_sparse_sfa_c8` / `enable_sparse_li_c8` / `c8_enable_reshape_optim`
quartet, and the reporter's P0/P1 blocks add only `recompute_scheduler_enable: false` and
`multistream_overlap_shared_expert`. Their own delta is the report: setting
`c8_enable_reshape_optim` to false makes it answer, and `enable_sparse_sfa_c8` alone gives a
garbled first curl.

## `c8_enable_reshape_optim` lifecycle (verified in the clone)

| ref | state | commit / note |
| --- | --- | --- |
| v0.23.0rc1, v0.23.0, v0.24.0rc1, v0.25.1rc1, v0.26.0rc1, v0.26.0rc2, v0.27.1rc1 | present in `vllm_ascend/` | used by `sfa_v1.py` StoreKVBlock paths |
| v0.28.0.rc1 | absent from `vllm_ascend/`, still referenced in tutorial configs | `c30092741` "[Refactor] Auto-enable LI C8 reshape optimization (#15752)" removes the switch and enables the optimization unconditionally (contained from `v0.28.0.rc1`) |
| v0.30.0rc1 | present again in `ascend_config.py` (default `true`, gated by `enable_sparse_li_c8` and prefill-node check) | contains `1e0742722` "[Misc][Config] Restore C8 reshape optimization switch (#16952)" |
| v0.30.0 | absent | `ac5324b2a` "[Refactor] Remove StoreKVBlock optimization and c8_enable_reshape_optim switch (#18128)" — `git tag --contains ac5324b2a` is empty; the tag carries only the backport `c604a5c86`; release note in `docs/source/user_guide/release_notes.md:118` |

The switch moved independently on release branches (`main` and `releases/v0.30.0` removed it;
`v0.27.1rc1` still had it without the restore commit), so state is only meaningful per tag.
"the switch no longer exists upstream" is true only for specific tags. A report naming two
versions (`0.23.0/0.26.0rc1`, issue #15927) must be checked at both tags; both contained it.

## What the reshape switch actually launches (v0.26.0rc1)

`get_ascend_config().c8_enable_reshape_optim` gates two custom ops in
`vllm_ascend/attention/sfa_v1.py`: `torch.ops._C_ascend.store_kv_block_metadata(...)` at `:461`
(during sparse-attention metadata build, an **AICPU** op —
`csrc/attention/store_kv_block_metadata/CMakeLists.txt:10 add_modules_sources_aicpu`) and
`torch.ops._C_ascend.store_kv_block(...)` at `:1988`/`:2006` (cache write, AI-core
`csrc/attention/store_kv_block/op_kernel/store_kv_block.cpp`). A P-node hang whose plog names
`store_kv_block_metadata` is the AICPU tiling-launch path of the documented conflict below.
Both call sites are skipped when the flag is off: the cache-write branch tests
`enable_sparse_li_c8 and c8_enable_reshape_optim` (`sfa_v1.py:1547-1549`, `:1983`) and falls back
to `torch_npu.npu_scatter_nd_update_`. The switch therefore explains a hang at op launch, not an
accuracy change by itself — the reporter's separate "first curl garbled with `sfa_c8`" observation
stays an accuracy question, and no cause is established for it here.

## Documented AICPU/pooling conflict and its boundary

`docs/source/user_guide/configuration/additional_config.md:88` at v0.30.0rc1 carries:

> With HDK 0.26.0 or earlier, `c8_enable_reshape_optim` may conflict with pooling models that
> use AICPU operators. Set `c8_enable_reshape_optim` to `false` ... See issue #15896.

The linked issue vllm-project/vllm-ascend#15896 (open, framework already mitigates) says the
hang on a custom AICPU Tiling operator launch happens for "HDK before 26.0, 26.0 is not
included, for example 25.x, 24.x". The docs sentence is broader than the issue's own boundary.
Before applying this as a cause, read the report's `npu-smi` version: an environment already on
HDK 26.0.RC1 is *outside* the issue's stated range, so this remains a candidate, not a root
cause. The maintainer-authored workaround (from #16952: "can be disabled as a workaround")
does not say the knob alone fixes a hang on a newer HDK.

## sfa_c8 accuracy at v0.23.0: documented known issues

`docs/source/user_guide/release_notes.md:144` at tag `v0.23.0` carries the entry

> GLM-5.2-W4A8C8 P/D-disaggregated serving with `enable_sparse_sfa_c8` can show GPQA accuracy
> fluctuations of approximately 2-3 percentage points across repeated runs. [#14378]

and `docs/source/tutorials/models/GLM5.2.md` at the same tag states, in the 1M-context
warning (`:1273-1276`) and the tuning table (`:1661`), that DCP and `sfa_c8` are both
experimental in v0.23.0, that enabling them together has known issues (including performance
degradation) and is not recommended, that DCP must be enabled or disabled on **both** P and D
nodes, and that `enable_sparse_li_c8` is independent and stays recommended. The 1M examples
therefore enable DCP and leave `enable_sparse_sfa_c8` disabled.

Issue #14378 (closed 2026-09-03) is the maintainer-authored record behind that entry: its
recommendation is W8A8C8 for production and accuracy-sensitive workloads, W4A8C8 only for
experimental/memory-constrained deployments, and `enable_sparse_sfa_c8` left disabled in PD
unless memory requires it and accuracy was evaluated.

Use it like this: a report naming `enable_sparse_sfa_c8` with GLM-5.x and wrong/garbled
output at a v0.23.0-era build has a documented answer already — quote the release-note entry,
the tutorial warning and the W8A8C8/sfa_c8-off recommendation, and label the runtime cause
unknown. If DCP appears on one side only, the two-side warning is the first thing to check.
A report whose symptom is worse than "2-3 pp fluctuation" (e.g. one curl producing garbage)
goes beyond the documented issue; so does a claim that W8A8C8 weights are affected too — that
is a new observation, not covered by #14378, and no root cause is established for it.

Do not rely on issue search to find this: a lexical GitHub search for `sfa_c8 accuracy` in
the repository returned `total_count: 0` while the release note carried #14378. Read the
tag's `release_notes.md` "Known Issues" list and the model tutorial first.

## GLM-5.3-Flash / GLM-5-Next: which tags have the model, and the pool numbers

Long-prompt hangs on GLM-5.3-Flash (`--max-num-batched-tokens`, ~8K prompts, engine stops answering
even for 7 tokens) are read against `vllm_ascend/models/glm5next/`, not against the DSA/C8 flag
cluster above. Start by checking that the reported build can contain that code.

- **Availability.** `vllm_ascend/models/glm5next/` exists from `v0.28.0.rc1` and is absent at
  `v0.23.0`, `v0.24.0rc1`, `v0.25.1rc1`, `v0.26.0rc1`, `v0.26.0rc2` and `v0.27.1rc1` (no
  `glm5_next.py` anywhere in those trees). Landing commits: `e11eaf0c7` Triton KeyPool compression and
  pooled indexing (#16243, 2026-09-12), `5bba97cf6` AscendC KDA + causal convolution (#16251,
  2026-09-14), `58e48b536` state cache -> request-owned circular tail cache (#16537, 2026-09-15),
  `9693cc1a4` GLM-Next MTP with split KV cache groups (#16793, 2026-09-21), all contained from
  `v0.28.0.rc1`. A 2026-09-11 report against a 2026-09-03 image therefore runs a dev/branch build:
  say the pin is unresolved instead of quoting "v0.23.0" as the model's source
  ([revision-pinning.md](revision-pinning.md)).
- **The numbers that size the pool.** `glm5next/config.py:65-70` defaults `index_kpool = 4` and
  documents it as the compression ratio ("every `index_kpool` indexer K entries pool into one stored
  entry ... the 300B checkpoint ships 4; topk selects at pool granularity"). `glm5next/kv_cache.py:28-42
  get_kpool_tail_ring_capacity` returns `compress_ratio + num_speculative_tokens`, i.e. the tail ring
  holds one open pool plus the not-yet-accepted speculative rows. `glm5next/cache_config.py:145-167`
  at `v0.30.0` (same three messages at `:148-167` in `v0.28.0.rc1`) already validates that main MLA and
  compressed-indexer caches share one logical `block_size` (divisible by `compress_ratio`) and that the
  tail ring is at least `compress_ratio` — so no tag from `v0.28.0.rc1` onward carries an indexer block
  size that is silently `block_size // 4` of the main one. Treat the "admission predicted at pool
  granularity" reading of a follow-up thread as a description of that dev build, not as
  current-tree behaviour.
- **Why nothing errors.** Upstream `vllm/v1/core/kv_cache_manager.py` at vLLM `v0.23.0` returns `None`
  from `allocate_slots` when the prediction exceeds free blocks — the full-sequence gate at `:354-368`
  (`if num_blocks_to_allocate > self.block_pool.get_num_free_blocks(): return None`) and the per-step
  path at `:385-398`. The scheduler then leaves the request waiting: no 400, no 5xx, and an oversized
  request that holds the pool head-of-line-blocks later small ones. That is the code shape behind
  "requests that cannot be admitted should fail fast", and it means a hung request's HTTP silence is
  expected, not a second bug.
- **Cheap next checks.** Lower `--max-model-len` / `--max-num-batched-tokens` so the admission
  prediction fits the allocated blocks and see whether the cliff moves with the pool (config
  dependence) rather than with hardware; sample `vllm:*_waiting` and KV-cache usage from `/metrics`
  while the request hangs ([kv-cache-metrics-and-hbm-usage.md](kv-cache-metrics-and-hbm-usage.md));
  capture the engine log for the request id, since the client only sees the gateway timeout.

## Issue states (as observed)

- #12533 (GLM-5.2 + memcache + C8, P node hang on first curl): closed `not_planned`, 5 comments.
  Maintainers asked to (a) give driver/firmware version, (b) test with memcache off while C8
  stays on to isolate memcache vs C8, (c) collect hung-worker stderr — the pasted log only had
  EngineCore-level `TimeoutError`. No answer, closed stale by bot. No confirmed cause.
- #15927 (0.23.0/0.26.0rc1 + GLM-5.2 + Memcache, 4-machine PD, P node unresponsive): open,
  one ping comment. No confirmed cause.
- #10724 (DSV4-Flash 2×A2 PD-Mix crash): closed `completed` but the reporter said a changed
  deployment never reproduced it and did not know which parameter mattered; a second user
  reported the same symptom. Treat the close as non-reproducibility, not a fix.

Runtime evidence (worker stdout/plog, `npu-smi`, `pip show`) is required to move any of these
from candidate to cause; source inspection alone can only name the guard, the path and the
documented workaround.
