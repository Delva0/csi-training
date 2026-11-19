# The KV admission cliff: a long prompt never runs, and everything behind it stops

## Scope

No traceback, no OOM line, service "healthy": prompts above some token count never return, and once one
has been accepted nothing else is answered either — including a 7-token request — until the operator
restarts. AICore sits at 0 %, `curl` on `/v1/models` still answers, and KV-cache usage in the metrics
stays low. This is a **scheduler admission** shape, not a compute path, not a device or collective fault
(`pd-collective-hang.md` is the plog `STUCK` family) and not the startup refusal of
`startup-and-stall.md` §3b; `startup-and-stall.md` §2 points here for exactly this signature.

## Why it looks like a hang instead of an error

Upstream's block managers return `None` from `allocate_slots` when the request cannot fit, and the v1
scheduler then leaves that request in `waiting` and tries the next one — so an oversized request that can
*never* be admitted blocks the queue (head-of-line) while the engine keeps spinning. Nothing raises, and
the frontend's 504 is the gateway's, not the engine's. Distinguish it from a real stall by the request
counters: `vllm:*_waiting` climbing with no running request, and a restart clearing it.

## The discriminating number: per-request blocks of the *smallest-page* group

The admission bound upstream uses is a **block count**, while the startup concurrency estimate is a
**byte** ratio, so a cache group whose page is tiny disappears from the estimate while still consuming the
shared block index pool. Compute both, do not reason about them:

- the cliff token count ≈ `num_blocks × block_size` **of the group with the smallest block_size**;
- the cap at `vLLM ≥ v0.23.0` for a recycling spec is
  `cdiv(min(sliding_window - 1 + max_num_batched_tokens, max_model_len), block_size) + 1`
  (`vllm/v1/kv_cache_interface.py` `SlidingWindowSpec.max_admission_blocks_per_request`;
  `ChunkedLocalAttentionSpec` uses `attention_chunk_size + max_num_batched_tokens`), applied by the
  factory `vllm/v1/core/single_type_kv_cache_manager.py` (~`:1339-1345`) and consumed as
  `apply_admission_cap` (~`:133-144`).

The `max_num_batched_tokens` term is how the reported workaround works: halving it halves the cap, so a
request that used to overshoot the pool now fits. That is also the check — if lowering
`--max-num-batched-tokens` moves the cliff, the class is confirmed; if it does not, look elsewhere.
The same arithmetic predicts a second cross-check the reporter can run: because the cliff is
`pool_blocks × block_size` of that group, it must move with `--gpu-memory-utilization` (pool size) and
with `--block-size`, and must not move with the model alone. Ask for one run at a different utilisation
before quoting a mechanism.

## The downstream half (verified at `v0.23.0`)

`vllm_ascend/core/single_type_kv_cache_manager.py` **overrides** `get_manager_for_kv_cache_spec`, which is
why the cap has to be re-applied there:

- `AscendMLAAttentionSpec` with `compress_ratio > 1` → `CompressAttentionManager` with
  `max_admission_blocks_per_request = cdiv(max_model_len // compress_ratio, block_size) + 1`;
- `SlidingWindowSpec` / `ChunkedLocalAttentionSpec` → the upstream spec method is called explicitly, with
  the in-code comment that upstream's own block "lives in dead code (never reached)" after the override.

Where a per-request state cache gets its geometry: `vllm_ascend/models/deepseek_v4.py:108`
`AscendCompressorStateCache.__init__` stores `block_size` **from its caller** (`:646`/`:654` pass
`_dsv4_block_sizes()[cache_config.block_size][0][2 or 3]`), and its `get_kv_cache_spec` builds an
`AscendSlidingWindowMLASpec(block_size=self.block_size, sliding_window=self.sliding_window, ...)`. So read
the caller before blaming the manager: a spec whose `block_size` is the *compress ratio* (4 or 128) is
indexed per token, and `cdiv(prompt_tokens, block_size)` then eats the whole pool. At `v0.23.0` the
kpool/compressed-attention code lives under `vllm_ascend/attention/kvcomp_attn/` (only
`__init__.py`/`attention_utils.py` — the cache classes are in the model file).

## The kpool / indexer-state variant (`v0.28.0.rc1`+, #16365's twin)

The group that goes small can be a kpool or indexer-state cache rather than the KV group the model's
attention declares. In the GLM-5.3-Flash family those specs exist only from **`v0.28.0.rc1`**:
`vllm_ascend/models/glm5next/kv_cache.py` (`78f3a90f2`, #15127; the path does not exist at `v0.23.0` —
see `../workflow/version-pinning.md`). At that tag `Glm5NextTailCache.__init__` (`:136`) sets
`self.block_size = compress_ratio if ring_capacity is None else ring_capacity`, and `get_kv_cache_spec`
(`:146-153`) returns `AscendIndexerKPoolTailSpec(block_size=self.block_size, num_kv_heads=1,
sliding_window=self.compress_ratio, compress_ratio=…)` — **one tail page per request, with the page sized
as the compression ratio**, which is the ratio-as-block-size shape the reported analysis names.
`KpoolTailManager` at that tag is a `FullAttentionManager` subclass with `enable_caching=False` whose
docstring says it must "own one unshared ring block until request completion or preemption", so which
arithmetic a given build uses depends on the manager its spec resolves to: read it at the reporter's ref.

## Worked report (vllm-ascend #16365, follow-up to #15984)

GLM-5.3-Flash, TP8+EP, `v0.23.0`, "~8K-token prompts hang and permanently poison the engine after
redeploy"; the same payloads served fine on an earlier build, short prompts fine, a second model behind
the same gateway fine. The thread's own analysis (commenter, quoting an AI write-up — `reported`, not
verified) gives the arithmetic: pool 1937 blocks, a state group with `block_size = 4`, so the cliff sits
at `1937 × 4 ≈ 7748` tokens and the measured break is between 7648 and 7676 tokens; the same comment
reports `--max-num-batched-tokens 4096` restoring concurrency and 32K sequences. Its open question — is
`block_size = compress_ratio` a kernel requirement or a modelling slip? — is genuinely unresolved, so do
not present "change `block_size`" as the fix.

Two facts worth carrying: the reported peak occupancy was 218–330 blocks against a predicted
~1900, i.e. the *prediction* is what blocks admission, not real device memory; and a startup line of the
form `Maximum concurrency for N tokens per request: 1.10x` is consistent with this class rather than a
refutation, because that estimate divides by the group's `page_size_bytes`.

The **permanent poisoning** half is a scheduling question, not a memory one: a request that can never be
allocated leaves the waiting queue blocked, so restarting the client does not help. Keep that as
`hypothesis` unless the report shows the scheduler's own log. Levers to offer, labelled `proposal` until
the reporter runs one: `--max-num-batched-tokens 4096` (the reported working value — the cap is linear in
it), or serving without the kpool/speculation feature that creates the group; the cost is prefill
throughput. Confirm the cap formula at the reporter's ref before quoting it as fact.

## Check before reuse

```bash
# what the spec asks for, and what the manager is built with
git -C /workspace/repos/vllm-project/vllm show <vllm-ref>:vllm/v1/kv_cache_interface.py \
  | sed -n '/def max_admission_blocks_per_request/,+22p'
git -C /workspace/repos/vllm-project/vllm-ascend show <tag>:vllm_ascend/core/single_type_kv_cache_manager.py \
  | sed -n '/def get_manager_for_kv_cache_spec/,+45p'
# the state cache's block_size comes from the caller, not from the manager
git -C /workspace/repos/vllm-project/vllm-ascend grep -n 'StateCache(' <tag> -- vllm_ascend/models
# the kpool variant: does the tag have the tail-cache spec at all, and what block_size does it pass?
git -C /workspace/repos/vllm-project/vllm-ascend show <tag>:vllm_ascend/models/glm5next/kv_cache.py \
  | sed -n '/class Glm5NextTailCache/,+40p'
```

Use the **paired** vLLM tag (`../vllm-ascend/version-pairing.md`), not the vllm-ascend number. Falsifier: a small request
also stalls with no oversized request in the queue (then it is not head-of-line admission), the cliff does
not move when `--max-num-batched-tokens` changes (then read the spec the model actually registers), or the
threshold does not move with pool size (`--gpu-memory-utilization`, `--block-size`) but does move with model
length — then it is a model-side limit, not an admission cliff.
