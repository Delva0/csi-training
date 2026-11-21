# DCP × C8 on the first request: `MTE accesses an invalid GM address` in a KV-reading kernel

## Scope

A DSA / SFA deployment (GLM-5.1 / 5.2 / 5.3 `w8a8c8` / `w4a8c8`, the DeepSeek-V3.2
family) that

- serves with `--decode-context-parallel-size N` (N > 1), usually with
  `--cp-kv-cache-interleave-size`,
- has C8 on (`enable_sparse_sfa_c8` and/or `enable_sparse_li_c8`),
- replays a `FULL_DECODE_ONLY` ACL graph (`acl_graph.py` frame in the traceback),

and dies with a device **addressing** fault: `errcode:(0x800000, …)` /
`errorStr: MTE accesses an invalid GM address or the cross-device memory access times
out`, with a `fault kernel_name=` or an op marker naming a sparse-attention kernel that
reads the KV cache. Verified at v0.26.0rc1 `f2f74a1` (SOURCES `15906`; upstream vllm
v0.26.0 `568afb3`). #14320 is the same class at `releases/v0.26.0rc`, and both reports
carry the same control: **the identical script served on v0.23**.

Not this page: a mixed prefill+decode batch whose indexer scatter receives the
full-batch `slot_mapping` (`indexer-cache-scatter-fault.md`); a P node that stops
answering or a garbled first request with the same flags
(`c8-flags-and-kv-store-path.md`); a KV-cache-construction
`sum_page_size_bytes % … page_size_bytes` assert (`c8-page-size-reshape-assertion.md`).

## Rule 1 — the client is the last to know

A perf harness (`evalscope perf`, `benchmark_serving.py`, `vllm bench serve`) that
prints `aiohttp.client_exceptions.ClientOSError` / `ConnectionResetError: [Errno 104]
Connection reset by peer` and then `ConnectionRefusedError` is reporting a **server
death**, not a network, port, TLS or tokenizer problem. Read the serve log's first
`ERROR` block from the same second. Recorded #15906: evalscope reached
`--parallel 1` / `Total / Success / Failed 5 / 1 / 4` with the first failure 6 s after
its own connection test, and the master log has the fault at 14:21:22 on
`Worker_DP0_TP5_DCP5_EP5` (Device 5): `torch.OutOfMemoryError: synchronize: …
AclrtSynchronizeStreamWithTimeout(stream()), error code is 507035` →
`[Error]: The vector core execution is abnormal.` → `aivec error, core id is 47, error
code = 0x800000` → `retCode=0x31, [vector core exception]`.

The **peer node's later timeout is a follower**, not a second bug: the slave log's
`Worker_DP2_TP3_DCP3_EP19` reports `507034` + `[vector core timeout]` six minutes after
the master's fault. Rank it as a consequence and send it back as context.

## Rule 2 — identify the kernel by its op marker, not by the core class

Two ops in this class fault, and the dump's own wording follows the op's core type, so
the core class separates *kernels*, never causes:

| op | dump | shape |
|---|---|---|
| `LightningIndexerQuant` (AIC) | `retCode=0x26, [aicore exception]`, `507015`, `fault kernel_name=LightningIndexerQuant_<hash>` | #14320: v0.26.0rc, GLM-5.2-w4a8 PD + DCP=8 + MTP, `FULL_DECODE_ONLY`, after ~400-532 requests of 20-40k input |
| `kv_quant_sparse_flash_attention` (AIV) | `retCode=0x31, [vector core exception]`, `507035`, `The vector core execution is abnormal`, aivec dump with `core id` / chipId | #15906: v0.26.0rc1, GLM-5.3-w8a8c8, TP8 × DP4 (2 nodes × 16 × 910B2C), DCP=8 + interleave 128, `max-model-len 1048576`, first `--parallel 1` 4096-in/1024-out request |

## Rule 3 — `============ tiling入口 ============` is a leftover debug print

In #15906's dump that line reads
`============ tiling入口 ============[FUNC:GetAttrParaInfo][FILE:kv_quant_sparse_flash_attention_tiling.cpp][LINE:1672]`,
and at `f2f74a1` line 1672 of that file is exactly

```cpp
OP_LOGE("ccccc"," ============ tiling入口 ============");
```

— the last statement of `QSFAInfoParser::GetAttrParaInfo` before `return
ge::GRAPH_SUCCESS` (`:1673`), and the **only** occurrence of `tiling入口` in `csrc/`.
It prints on every tiling call of that op, so it says *which op the dump's log tail came
from*; it is not the failing instruction and carries no diagnosis. Use it to name the
kernel, then read the `errorStr`/`retCode` lines for what happened. (When a dumped
`LINE:` lands on an `OP_LOGE`, read the line in the pinned tree before crediting the
tiling entry with the fault — the same family as SKILL.md rule 11's wrapper text.)

Two corollaries, both from that run:

- `csrc/attention/kv_quant_sparse_flash_attention/op_host/` **does** carry
  `kv_quant_sparse_flash_attention_tiling.cpp` at this pin, so the printed line is
  readable locally. The recorded session read a `rg -l … | head -20` file list as "the
  tiling source is not exported": a `head` on a `-l` list is not an existence test.
- The op name in the dump is enough to pick the lane; do not `ls csrc/<op>/` to look for
  a kernel before the config question below.

## Rule 4 — the mechanism: the ids are derived, and nothing checks them

Anchors at `f2f74a1`, file `vllm_ascend/attention/context_parallel/sfa_cp.py`:

| line | fact |
|---|---|
| `:114-121` | `total_cp_size = self.dcp_size`; `max_local_block_table_cols = cdiv(max_model_len, block_size * total_cp_size) * blocks_per_phys_block`; the replicated-view buffers are sized from it |
| `:177-211` | `_build_block_table_replicated_view`: `local_logical_blocks * total_cp_size + rank_in_replicated_view` (`:200`), and for `blocks_per_phys_block > 1` `(local_phys_blocks * total_cp_size + rank) * blocks_per_phys_block + local_sub_blocks` (`:204-207`) |
| `:297-312` | the metadata swap: `common_attn_metadata.slot_mapping = slot_mapping_replicated_view` and `block_table_tensor = block_table_replicated_view` for `build_metadata()`, restored in the `finally` |

So the ids the attention op receives are **derived** (`local × dcp + rank`), not the
block pool's own ids, and that path performs no comparison against the KV cache's block
capacity. A mismatch therefore surfaces as an out-of-range GM address in whichever
KV-reading kernel consumes them.

What is confirmed, and what is not:

- **Reporter-confirmed for LIQ** (#14320: instrumented prints): the replicated block ids
  exceed the k_cache capacity and `LightningIndexerQuant` indexes with them; the same
  config does not crash on v0.23, and the reporter blames the v0.26 PCP-removal change
  that made `total_cp_size` `dcp_size` instead of `pcp_size * dcp_size`.
- **Hypothesis for the SFA-C8 kernel** (#15906): the same class, one op over. State it
  as the leading hypothesis with this check, not as the cause.
- **Honest differences to keep in the answer.** With `--prefill-context-parallel-size 1`,
  `pcp × dcp == dcp`, so #14320's blamed arithmetic is numerically identical for the
  #15906 command — what remains is the unvalidated capacity invariant, not that diff.
  And #14320 needed hundreds of requests, while #15906 faulted on the **first** request,
  so load-dependent block-table growth is not a necessary ingredient.

## Rule 5 — check the recipe row before the mechanism

#15906's serve command is the GLM-5.2 tutorial's **1M-context block** (both C8 flags,
`enable_flashcomm1` + `enable_dsa_cp`, `--decode-context-parallel-size 8
--cp-kv-cache-interleave-size 128`, `FULL_DECODE_ONLY` + MTP 3) transplanted onto
A2/910B2C ×32 (`--tensor-parallel-size 8 --data-parallel-size 4
--data-parallel-size-local 2`). Three pinned statements decide the first paragraph of the
answer:

- `docs/source/user_guide/release_notes.md:259` — "GLM-5.2 1M-context deployments were
  validated only on Atlas 800 A3; the A2 series was not validated for 1M context."
- `:262` — GLM-5.2 DCP × `enable_sparse_sfa_c8`: "known issues, including performance
  degradation, and is not recommended" (#14011).
- `docs/source/user_guide/configuration/additional_config.md:81-82` — C8 SFA: "SFA
  prefill context parallelism and Ascend 950 DCP are not supported".

The support matrix at this pin (`:1817-1818`) *does* sanction DCP with
`enable_sparse_sfa_c8` on w8a8c8 weights (and not on w4a8c8) — see Rule 3 of
`c8-flags-and-kv-store-path.md` — so do not turn the matrix into a prohibition. The
verdict line is the combination: "your command is the A3 1M block; on A2 with 1M + DCP8 +
C8 the pin validates none of it", with the mechanism below it.

## Rule 6 — ablation ladder, one variable per launch

1. **`--max-model-len 1048576 → 131072`**, everything else fixed. One flag: it removes
   the A3-only 1M claim from the equation and shrinks the derived block-table columns
   `dcp`-fold. If the fault survives, the capacity reading is refuted and the id-space
   question stays open — say which happened.
2. **`enable_sparse_sfa_c8: false`, keep `enable_sparse_li_c8: true`.** A sanctioned
   tutorial row (`docs/source/tutorials/models/GLM5.2.md:145`, `:211`), and the faulting
   op is the SFA packed-C8 kernel.
3. **`--decode-context-parallel-size 1`** and drop `enable_dsa_cp`. Removes the
   replicated-view path entirely while keeping C8: this is the rung that separates "DCP
   index space" from "C8 kernel".
4. `c8_enable_reshape_optim: false` reverts only the indexer-cache write; never present
   it as "the C8 ablation" (Rule 1 of `c8-flags-and-kv-store-path.md`).
5. The reporter's own control is **0.23.0rc1 ok / 0.26.0rc1 crash**, so for this
   regression shape the next rc or final is rung 1; the snapshot cannot show what a later
   release fixed — say so instead of deriving a fix that is not in it.

Memory is not the lever. `--gpu-memory-utilization 0.92` at 1M is worth one launch at 0.8
only after 1-3, and the dump's `Memory_Allocation_Failure(EL0004): Failed to allocate
memory requested by UNKNOWN module` carries **its own earlier timestamp**
(`2026-09-04-13:57:31`, 24 min before the 14:21:22 fault, same PID): that is an
error-history record inside the error report, to be placed with
`scripts/plog-timeline.sh --since <report date>` before it is promoted to a cause
(`../hccl-and-hardware/error-report-timestamps.md`, SKILL.md rule 13). The crash itself
is an addressing fault, not a size refusal.

## What to send back

Both nodes' serve logs whole (not grepped), the plog for the `chipId` in the dump, the
launch script, `npu-smi info`, and one line: does the same command at
`--max-model-len 131072`, and at `--decode-context-parallel-size 1`, still fault?

## Scope and recheck conditions

- Anchors verified at v0.26.0rc1 `f2f74a1`. Recheck with
  `rg -n '_build_block_table_replicated_view|total_cp_size = self.dcp_size'
  vllm_ascend/attention/context_parallel/sfa_cp.py` and `rg -n 'tiling入口' csrc/`; the
  builder exists since the v0.26 PCP-removal refactor, so an older reporter's tree may
  have `pcp_size * dcp_size` there instead.
- Claim classes: code anchors and doc lines are source-verified at the pin; the capacity
  mechanism is reporter-confirmed for LIQ (#14320) and a hypothesis for the SFA-C8 kernel
  (#15906, whose reporter never posted a resolution).
- The recorded #15906 session reached this signature on its 3rd tool turn and was then
  cut at the output cap inside its second ranked cause: the delivery lesson is rule 1 of
  SKILL.md, and it is not evidence about the fault.
