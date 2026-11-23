# `num_decodes` vs `num_spec_decodes`: the GDN backend refuses a step that mixes spec and plain decode

## Scope

A hybrid linear-attention (gated-delta-net) family — Qwen3.5/Qwen3.6, Qwen3-Next — runs
with MTP/eagle speculative decoding for a while and then **every rank raises in the same
second**:

```text
File "/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py", line 2133, in _build_attn_group_metadata
    attn_metadata_i = builder.build(
File "/vllm-workspace/vllm/vllm/v1/attention/backends/gdn_attn.py", line 310, in build
    assert not (num_decodes > 0 and num_spec_decodes > 0), (
            f"num_decodes: {num_decodes}, num_spec_decodes: {num_spec_decodes}")
AssertionError: num_decodes: 1, num_spec_decodes: 1
```

Clients see `POST /v1/chat/completions … 500`, the pods restart, and **with MTP off the
same deployment runs for hours** — reported as "偶发崩溃 … 关闭MTP 则运行4小时能正常". The
counts in the message are ≥ 1 on both sides, and the step is a *decode* step: a prefill
half would raise the prefill branch, not this assert.

This is a host-side batch-composition refusal in the attention metadata builder. It is not
a device fault (no `EZ`/`EE`/`507xxx` record, no plog, nothing to blame on HCCL or on the
card), not an accuracy report, and not the `torch.argsort` patch: the downstream tree
patches only that symbol for GDN (`vllm_ascend/patch/worker/patch_module.py:29-35` at
v0.17.0rc1) and the assert fires before a kernel is launched.

Recorded #7532 (image v0.17.0rc1, Qwen3.5-27B, TP=2, 400 pods, `qwen3_5_mtp` with
`num_speculative_tokens: 3`, `--async-scheduling`, `FULL_DECODE_ONLY`). The recorded
session named the assert, the owner and the phase in a first verdict, then spent the rest
of its budget deciding *which* request had no drafts and shipped no ranked answer; the
reporter's own later comment ("可能需要这个 pr") points at an upstream vLLM fix PR. **Take
rule 2 first — it decides the answer in two calls.**

## Rules

1. **Read the two counters as the classification, then stop.** `num_spec_decodes` is how
   many requests in *this step* carried draft tokens; `num_decodes` is how many non-spec
   requests had a query length of exactly 1. Both ≥ 1 means one decode request was
   scheduled with drafts and another without. That is a property of the scheduler's output
   for that step, so no artefact from the reporter (npu-smi, plog, tiling log) can refine
   it further — the only remaining question is the version one (rule 2).

2. **Version window first: the refusal is dead at vllm ≥ v0.22.1, so "upgrade/backport" is
   the rung — the reporter's own follow-up lands on an upstream fix, and the pins below
   confirm such a fix exists.** Verified in the snapshot at two pins:

   | pin | `vllm/v1/attention/backends/gdn_attn.py` | consequence |
   |---|---|---|
   | vllm **v0.17.0** `b31e932` | counters at `:179-213`, then `assert not (num_decodes > 0 and num_spec_decodes > 0)` at `:310-312` (comment at `:307-308`: "Function code counted on either presency non-spec decode or spec decode, but not both") | any mixed step kills the engine |
   | vllm **v0.22.1** `0decac0d96` | reclassifies instead: `if num_decodes > 0 and num_spec_decodes > 0:` → `num_prefills += num_decodes; num_prefill_tokens += num_decode_tokens; num_decodes = 0`, commented *"the prefill kernel handles 1-token sequences with initial state correctly, producing identical results"* (`:236-247`); the identical `assert` still printed at `:375` is now unreachable | mixed step runs as prefill |

   The downstream tree carries the same handling once it ships its own builder: at
   vllm-ascend **v0.22.1rc1** `5f6faa0` `vllm_ascend/ops/gdn_attn_builder.py` has
   `AscendGDNAttentionMetadataBuilder` (`:741`, selected by `:1264-1265`), the same
   reclassification at `:1010-1015` and the same inert assert at `:1145`. At
   v0.17.0rc1 there is **no** downstream builder (`ls vllm_ascend/ops/` has no
   `gdn_attn_builder.py`), so the upstream assert is live. Quote this as the version
   scope: "fixed upstream after v0.17; a vllm-ascend built against vllm ≥ v0.22.1 handles
   it", and say plainly that the snapshot cannot date the exact PR (the reporter's own
   follow-up named one). When the case's snapshot has **no** newer pin to grep, these two
   pins are still the citable evidence — the check is one `rg -n 'num_decodes > 0 and
   num_spec_decodes' <vllm>/vllm/v1/attention/backends/gdn_attn.py`, and the presence of the
   reclassification branch rather than of the assert is the answer.

3. **Where the mask comes from, so the answer can name the owner once.** The runner builds
   a per-request draft count with `-1` as the "not a spec decode" marker:
   `vllm/v1/worker/gpu_model_runner.py:1695-1719` upstream is the same code as
   `vllm_ascend/worker/model_runner_v1.py:818-846` downstream (the only ordering difference
   is the `if` placement), and it is filled only from
   `scheduler_output.scheduled_spec_decode_tokens` **and** only when
   `num_computed_tokens_cpu >= num_prompt_tokens` — so a request still sitting in a
   chunked-prefill tail, or one with no entry in that dict, is `-1` = non-spec. The mask
   is passed only when `use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0`
   (`model_runner_v1.py:798`, `:2122-2128`); with spec decode off the block cannot fire at
   all. Per-request draft absence, at vllm v0.17.0 `vllm/v1/core/sched/scheduler.py`: the
   prefill-chunk branch discards drafts (`:1607-1610`), preemption clears
   `request.spec_token_ids` (`:926`), the scheduled count is clipped by
   `num_new_tokens + num_computed_tokens - num_tokens - num_output_placeholders` (`:496-510`),
   and the async path deliberately skips a request to avoid "partial draft tokens since
   this prevents uniform decode optimizations" (`:363-374`). Treat these as the producer
   *inventory*: do not spend turns proving which one fired.

4. **Do not offer the workload as the fix, and do not offer `--async-scheduling` as the
   mechanism.** With continuous batching any step in which one request enters decode while
   another is mid-MTP produces the composition — the same arithmetic as the mixed
   prefill/decode rule in `../mixed-batch-splits/index.md` (rule 6 there). Dropping
   `--async-scheduling` is one ablation row (the placeholder clip at `:496-510` is one way
   to reach it), never the diagnosis; the ladder is: MTP off (reporter-confirmed clean for
   4 h — it proves the lane, it is not a fix), `--async-scheduling` off, then the version
   correction. Do not propose a patch to the assert on the reporter's image: the upstream
   reclassification is the repair, and a local `assert` deletion leaves the mixed half
   unvalidated.

5. **Precedents that are not this shape.** A *dense* backend refusing a spec state
   (`assert attn_metadata.attn_mask is not None`, `dense-attn-state-gap.md`) is a
   downstream enum gap; `EZ9999 … queryT(N) must equal last(actualSequenceLengthQ)(M)`
   (`fia-tnd-seqlen-mismatch.md`) is a padding mismatch. Both print a spec-decode-launch
   traceback and both say nothing about *which requests* are in the step — the counters in
   this page's message are what distinguish them.

## What the first message owes

- The verdict line: host-side metadata assert in upstream GDN, mixed spec/plain decode
  step, no device involvement; MTP off is why it disappears.
- The version statement from §2 with the pin its `path:line` came from, and the upgrade /
  backport as rung 1 (the recorded reporter pointed at an upstream vLLM fix of their own).
- The ladder (MTP off → `--async-scheduling` off → version), each row labelled as an
  ablation rather than a cause.
- What to send back if they want the composition confirmed rather than just believed: the
  `--speculative-config` / `--async-scheduling` / `--max-num-batched-tokens` lines, and the
  `Running: n reqs` line printed just before the crash (a request that had *just* finished
  its prefill is the usual neighbour). Label the producer story a hypothesis; the
  counter pair in the paste is the evidence.
