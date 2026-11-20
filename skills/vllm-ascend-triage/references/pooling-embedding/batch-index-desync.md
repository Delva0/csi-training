# Pooling service dies under concurrency: a downstream `InputBatch` copy's slot fast path

Worked case: vllm-ascend #5525 — vllm-ascend **v0.11.0** (`2f1aed9`) + vllm
**v0.11.0** (`b8b302c`), `quay.io/ascend/vllm-ascend:v0.11.0`,
`vllm serve …/Qwen3-Embedding-8B --task embed --max-num-seqs 256
--max-num-batched-tokens 8192 --max-model-len 8192`, driven by `vllm-bench`
(`backend openai-embeddings`, sharegpt). ~1800 requests answer `200`, then the
engine dies and every later request gets `400 Bad Request`. The reporter's only
evidence is four screenshots of the bench output plus the engine traceback; the
replay can read them only as OCR, which recovers the frame lines
(`model_runner_v1.py:1928 execute_model` -> `:766 _update_states` -> `:1219
_may_reorder_batch` -> `vllm/v1/attention/backends/utils.py:798`, the two
`model_runner_v1.py` numbers matching the v0.11.0 layout exactly) but **not** the
key printed by the `KeyError`. Everything else below is source-read at the two
pins above.

## The claim, in the order it should be said

1. The crash is real and it is **downstream (vllm-ascend)**, not upstream vLLM:
   v0.11.0 ships its own copy of upstream's worker batch class
   (`vllm_ascend/worker/npu_input_batch.py:94 class InputBatch`, a copy of
   `vllm/v1/worker/gpu_input_batch.py`), and the copy adds a pooling fast path
   upstream does not have.
2. It is **pooling-only**: the fast path is taken only when
   `self.is_pooling_model`, so the same bench on a generation model does not
   take it. That is the ownership split to state in the first paragraph.
3. It is **concurrency-shaped**, not model-shaped: it needs a request finishing at
   a non-tail batch slot *and* a new request admitted in the same step, which
   `--max-num-seqs 256` produces constantly and a single request never does.
4. It is **released-fixed by construction**, not by a patch: by v0.23.0
   (`5cb98ca`) `vllm_ascend/worker/npu_input_batch.py:33` is
   `class NPUInputBatch(InputBatch)` with `__init__` as its only method — no
   `add_request` / `remove_request` / `condense` override — so the slot machinery
   is upstream's again.

## The divergence (v0.11.0, both pins)

| Downstream `npu_input_batch.py` | Upstream `gpu_input_batch.py` |
|---|---|
| `_register_add_request` starts `assert request.sampling_params` (`:303`) | records the added-request metadata only `if request.sampling_params:` (`:301`) — no assert |
| `add_request` `if not self.is_pooling_model: req_index = self._register_add_request(request) else: req_index = self.num_reqs` (`:316-324`) | `req_index = self._register_add_request(request)` for every model (`:314`) |
| `remove_request` calls `batch_update_builder.removed_append(req_index)` only when **not** pooling (`:462-465`) | always (`:462`) |
| `condense` pooling branch: `del self._req_ids[num_reqs:]; del self.req_output_token_ids[num_reqs:] ; return` (`:573-577`) — no compaction | no pooling branch at all (`:580` onward is the swap-down loop) |

The `assert` is the origin of the fast path: the copy's `_register_add_request`
cannot be called for a pooling request (`sampling_params is None`), so the copy
routes pooling around it instead of relaxing the assert the way upstream did. The
rest of the fast path is the downstream consequence of "the batch is contiguous
for pooling models" — an invariant nothing enforces.

## The sequence that breaks it

`req_ids` returns the raw `self._req_ids` list (`:291-294`, `None` holes
included) while `num_reqs` is `len(self.req_id_to_index)` (`:807-808`):

1. Batch `_req_ids = [A,B,C,D]`, map `{A:0,B:1,C:2,D:3}`.
2. `B` finishes → `remove_request("B")`: map `{A:0,C:2,D:3}`,
   `_req_ids = [A,None,C,D]` (no `removed_append`, nothing recorded).
3. A new request `E` is admitted in the same step → `add_request`: pooling, so
   `req_index = num_reqs = 3`; `3 == len(_req_ids)`? no → `_req_ids[3] = E`,
   **overwriting live slot `D`**. Map becomes `{A:0,C:2,D:3,E:3}`: two req ids
   on one slot.
4. `condense()`: `num_reqs = 4`, `del _req_ids[4:]` deletes nothing →
   `_req_ids = [A,None,C,E]`. The `None` hole is still inside `[0,num_reqs)`.
5. `_update_states` (`vllm_ascend/worker/model_runner_v1.py:603`, tail at
   `:764 condense()` → `:766 self._may_reorder_batch(...)`, body at `:1200`,
   gate `:1218`) calls `reorder_batch_to_split_decodes_and_prefills`, which does
   `for i, req_id in enumerate(input_batch.req_ids): num_tokens =
   scheduler_output.num_scheduled_tokens[req_id]` —
   `vllm/v1/attention/backends/utils.py:797-798` → `KeyError`.

Two survivors of the same defect, so the answer covers either log: the `KeyError`
key is `None` (unhashable-looking but printed as `None`) when the hole reaches
the reorder, and a **silently wrong embedding** when the duplicated slot is used
first — `token_ids_cpu` / `block_table` rows belong to the request that used to
own index 3. `refresh_metadata` then reads `num_reqs` slots, so nothing repairs it.

## What to check, and what to send back

- Ask for the plain-text engine log with the `KeyError` line, not the screenshot:
  `KeyError: None` confirms this page, `KeyError: '<cmpl-…>'` moves the case to
  "a scheduled-request id missing from `num_scheduled_tokens`" (scheduler-side,
  compare against upstream's `Scheduler.schedule`).
- Ablation that costs one launch: `--max-num-seqs 8`, then `1`. The crash rate
  collapses and then disappears — consistent with the slot arithmetic, and it is
  also the field workaround while the reporter stays on v0.11.0.
- Same bench against a **generation** model at the same concurrency should not
  crash (the fast path is `is_pooling_model`-gated). One run separates
  "pooling batch bookkeeping" from "benchmark-client / HTTP" theories.
- Send back: the engine traceback as text, the serve log's
  `max_num_seqs`/`max_num_batched_tokens` lines, and whether `--max-num-seqs 8`
  survives the same dataset.

## Recheck conditions

Both line-number sets are for vllm-ascend v0.11.0 + vllm v0.11.0. Before citing
them for another report, re-run rule 2 of this topic's index: if
`npu_input_batch.py` at the reporter's ref defines `condense`, the copy is
present and the page applies; if the class only has `__init__`, the copy is gone
and this mechanism is out of scope for that version.
