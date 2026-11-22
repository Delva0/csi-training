# `sparse indices … shape is [T1, 1, topk], expected shape is [T2, 1, topk]` at `npu_sparse_flash_attention`

## Scope

A serve that starts cleanly, accepts a request, and the **first prefill** dies inside
the SFA op with the tiling's *shape comparison* — the only member of the
attribute-rejection family that prints **both** numbers:

```text
RuntimeError: call aclnnSparseFlashAttention failed, detail: Ez9999: lnner Error
EZ9999 [PID: 1296] … (EZ9999): sparse indices layout is TND, shape is [4996, 1, 2048],
expected shape is [512, 1, 2048] [FUNC:CompareShape] [FILE:sparse_flash_attention_tiling.cpp] [LINE: 579]
SparseFlashAttention do tiling failed, ret is -1.
Check NnopbaseExecutorDoTiling(executor) failed …
```

Python frames, at vllm-ascend v0.21.0rc1 (`80610e4438db`):
`vllm_ascend/ops/mla.py:192 mla_forward` → `attention/sfa_v1.py forward` →
`device/device_op.py:451` `torch.ops._C_ascend.npu_sparse_flash_attention(query=…,
key=…, value=…, sparse_indices=topk_indices, sparse_block_size=1,
layout_query="TND", layout_kv="PA_BSND", sparse_mode=3, attention_mode=2)`.
The third dim (`2048` here) is the model config's `index_topk`, i.e. constant; only the
two `T` values are information. Recorded #11080 (8×A2, GLM-5.1 W8A8, PD, `enable_dsa_cp`).

Not this page: the same `do tiling failed` chain on a **caller-built mask** or on
`npu_kv_rmsnorm_rope_cache`'s `cos or sin shape is invalid`
(`../custom-ops-runtime-faults/tiling-attribute-rejection.md`, which also carries the
general "GetShapeAttrsInfo/CompareShape = argument refusal, nothing launched" rule), and
a `561002`-only paste with no `EZ9999` body.

## First: read the two numbers, do not derive them

`CompareShape` hands you the numbers, so the only thing left is *which side is which*:

- The **first** shape is the tensor the caller passed (`topk_indices`).
- The **second** is what the tiling derived from the other inputs — normally the
  query's own rows.

So the finding is a **row-count disagreement between the index tensor and the query the
op was given**, and the arithmetic that pays is a division of the two Ts by the things
`SKILL.md` rule 3 allows (tp, `--max-num-batched-tokens`) — nothing else.

Recorded #11080's pair: `4996` vs `512`. `512 = 4096/8` = `--max-num-batched-tokens / tp`
(tp = 8 from the P instance's `--tensor-parallel-size`), a **per-rank** count. `4996` is not an integer multiple of `512` (9.75×), so it is **not a
CP slice of the same total** — stop hunting a `pcp`/`dcp` factor. And `4996 > 4096`, so it
cannot be that step's scheduled token count either: it is a *padded/global* count or a
buffer extent. Two independent counts, one local and one metadata/budget-derived, is the
whole shape of this family. (The exact provenance of the larger number was **not**
established in the recorded run and needs the dump below.)

## Where the two counts come from at v0.21.0rc1

All line numbers re-grepped at `80610e4438db`; `num_input_tokens` includes padding,
`num_actual_tokens` does not (`sfa_v1.py:139`, `:149`).

| Site | What it does |
|---|---|
| `sfa_v1.py:100` `get_impl_cls()` → `if enable_cp()` | the **CP** impl is chosen by `enable_cp()` (`attention/utils.py:59`: `pcp>1 or dcp>1`), **not** by `enable_dsa_cp()`. With `enable_dsa_cp: true` and no `--prefill/decode-context-parallel-size`, the **base** `AscendSFAImpl.forward` runs — a paste whose `forward` frame is `sfa_v1.py` is therefore not evidence that DSA-CP is off |
| `sfa_v1.py:267-320` (builder, `if self.enable_dsa_cp`) | builds `DSACPContext`: `num_tokens_pad = _round_up(num_input_tokens, tp)`, `num_tokens_per_device = num_tokens_pad // tp`, slices `cos`/`sin`/`slot_mapping` to `num_tokens_per_device` and **rebases `actual_seq_lengths_query` per request**. So under DSA-CP the op is handed a *per-rank* query and per-rank cumulative lengths |
| `sfa_v1.py:1129-1135` (forward) | uses `dsa_cp_context.actual_seq_lengths_query/_key` and `slot_mapping_cp`; the q/kv tensors themselves are the ones derived from `hidden_states`, which in the native branch stay **per-rank** (the `if self.enable_sp and not self.enable_dsa_cp:` guard skips `maybe_all_gather_and_maybe_unpad`, `:1175-1177`) |
| `sfa_v1.py:1328` | `topk_num_tokens = num_input_tokens or hidden_states.shape[0]` — a **metadata** count |
| `sfa_v1.py:1073` `_get_indexcache_topk_indices(n)` | `return self.topk_indices_buffer[:n]` — returns rows sized by that metadata count |
| `sfa_v1.py:457-461` `use_index_cache` | read off `hf_config.use_index_cache` (so `--hf-overrides`), plus `skip_topk`; `:1329` takes the buffer path when `skip_topk`, and `:1342` writes it back when `use_index_cache` |
| `models/deepseek_v4.py:1022-1027` | the buffer is `torch.empty(max_num_batched_tokens, index_topk)`: a **budget** extent, so any slice of it is not proof of this step's tokens |
| `models/deepseek_v4.py:842-853` | who sets `skip_topk`: `compress_ratio == 4 and getattr(config, "use_index_cache", False) and ".mtp." not in prefix`, then `freq = getattr(config, "index_topk_freq", 1)`, `skip_topk = max(indexer_seq_idx - 1, 0) % freq != 0` — so with `index_topk_freq: 4`, **three of every four indexer layers take the cached-buffer path** and the failure is expected on the first request, not intermittent |
| `sfa_v1.py:1152-1157` | the other branch, MLAPO, gathers `hidden_states` first (`maybe_all_gather_and_maybe_unpad`) — so it *changes the row convention*; its gate is `num_input_tokens <= MLAPO_MAX_SUPPORTED_TOKENS` (1024), and a `T` above 1024 puts the batch on the native branch |

Ranked owners, each with its one check:

1. **The index-cache path under a per-rank query** (`:1328` counting rows by
   `num_input_tokens` while the op is fed SP/DSA-CP-local rows). Check: same serve with
   `--hf-overrides '{"use_index_cache": false}'` (or drop the key) — if the request now
   runs, the cached-buffer branch is the owner and it is a vllm-ascend defect, not the
   reporter's flags. Confidence: **source-verified call site, mechanism hypothesis**
   (it is the only place in the base forward where the index tensor's rows come from
   metadata instead of from the tensors).
2. **The DSA-CP slice arithmetic** (`:270-271` deriving `num_tokens_per_device` from
   `num_input_tokens` while q comes from the per-rank `hidden_states`). Check: the same
   request with `enable_dsa_cp: false` (leave `VLLM_ASCEND_ENABLE_FLASHCOMM1=1` on).
3. **A budget/padding count reaching the op** (graph bucket or `num_input_tokens`
   padding). Already excluded on a script that passes `--enforce-eager`; otherwise one
   launch without graph mode.

## Gates the reporter's own launch script already closes — do not spend rungs on them

Read the script before ranking anything; on #11080 it already contained:

- `VLLM_ASCEND_ENABLE_FLASHCOMM1=1` — so `enable_dsa_cp`'s own gate
  (`utils.py:1345-1367`, `raise ValueError("DSA CP requires SP to be enabled …")`) passed.
  Do not answer "enable FLASHCOMM1".
- `--enforce-eager` — so torch.compile/ACL-graph bucket padding is **not** the variable,
  and "re-run with `--enforce-eager`" is a no-op rung. (The pinned GLM5 DSA-CP recipe also
  passes it, `docs/source/tutorials/models/GLM5.md:819`.)
- `--max-num-batched-tokens 4096`, `--enable-chunked-prefill` — same, already the
  documented value.
- `--additional-config '{"layer_sharding": ["q_b_proj", "o_proj"], "enable_dsa_cp": true, …}'`
  with `kv_role: kv_producer` — so `enable_dsa_cp_with_layer_shard()` is True on the P side
  (`utils.py:1369-1381`) and the P-side layer-shard path, not the D side, is the suspect.

What the script *adds* beyond the pinned recipe is `--hf-overrides '{"use_index_cache":
true, "index_topk_freq": 4}'`. At `80610e4438db` **neither key appears anywhere under
`docs/`** (`rg -n 'use_index_cache|index_topk_freq' docs/` → 0 hits) while `enable_dsa_cp`
is documented (`docs/source/user_guide/configuration/additional_config.md:102`). An
`--hf-overrides` key that no doc carries is invisible to a recipe diff — grep it by name in
`vllm_ascend/` (`getattr(...hf_config, "<key>", …)`) and in `vllm/`; that two-grep check is
what turns such a key into a code path with a gate. That is how this case's owner was found,
and it is the cheapest thing to do first on the next report that carries `--hf-overrides`.

## The one artefact that settles it

One process, no rebuild — wrap the op call and print both sides plus the two counts. It
also removes the OCR risk when the numbers arrive as a screenshot:

```python
import torch, vllm_ascend.attention.sfa_v1 as S
_orig = S.AscendSFAImpl._execute_sparse_flash_attention_process
def _spy(self, ql_nope, q_pe, kv_cache, topk_indices, md, asq, ask):
    print("SFA", "q", tuple(ql_nope.shape), "idx", tuple(topk_indices.shape),
          "num_input_tokens", md.num_input_tokens, "num_actual_tokens", md.num_actual_tokens,
          "asq[-1]", int(asq[-1]), "skip_topk", self.skip_topk,
          "dsa_cp", md.dsa_cp_context is not None, flush=True)
    return _orig(self, ql_nope, q_pe, kv_cache, topk_indices, md, asq, ask)
S.AscendSFAImpl._execute_sparse_flash_attention_process = _spy
```

For a CP impl (`pcp`/`dcp` > 1) wrap `sfa_cp.py`'s `_execute_sparse_flash_attention`
instead; it is the method the CP forward calls. `skip_topk`/`dsa_cp` in that line separate
owners 1 and 2 in one request.

## Delivery note

#11080 was also a delivery failure: 24 turns / 33 tool calls, **43.1 K characters of
reasoning against 115 characters of message text**, cut with the ranked answer already
drafted and no verdict ever sent (its one interim line said the OCR "已经给出决定性签名").
The tiling source (`sparse_flash_attention_tiling.cpp`) is in **no** export — `ops-transformer`
is absent and the string appears in no tree — so no source read resolves which side is
which. One grep to confirm the absence, then send the ranking above, the spy snippet and
the script/`tf` numbers as the request list.
