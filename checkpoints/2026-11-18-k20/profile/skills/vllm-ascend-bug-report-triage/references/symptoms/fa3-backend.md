# The FA3 (`FLASH_ATTN`) attention backend: mixed prefill + spec-decode batches and the `507015` they produce

## Scope

Reports where the argv **selected the FA3 backend** — `attention_config={"backend": "FLASH_ATTN"}` (the
batch-invariant / RL-rollout route, usually beside `VLLM_BATCH_INVARIANT=1`), dependency
`flash_attn_npu_v3` — and the run dies with an aicore error or answers wrong. The owner is
`vllm_ascend/attention/fa3_v1.py` and the decision is a Python slice of attention metadata, not CANN, not
the graph path and not the model.

Not this page: FA3 not selected at all (see "Selection" below — the gate is easy to miss), and a
`507015` whose log carries a DFX `fault kernel_name=…, hash=…` or an `acl_graph.py` / replay-sync frame
(`graph-capture-and-custom-ops.md` §C, `device-kernel-faults.md`). This page owns the third owner of the
same code: **an eager FA3 run with `num_decodes > 0` and `num_prefills > 0`.**

## Selection (`verified` at `v0.23.0rc1`, `platform.py:804-857`)

`get_attn_backend_cls` returns `vllm_ascend.attention.fa3_v1.AscendFABackend` only when the user-selected
backend is `FLASH_ATTN` **and** `_validate_fa3_backend` passes (`:808-809`). That validator (`:833-857`):

- returns `False` with `logger.info("FA3 will not be enabled when not in training-inference consistency
  scenario. Note that Ascend NPU will use its registered plugin backend instead.")` unless
  `attn_selector_config.use_batch_invariant` — i.e. **`VLLM_BATCH_INVARIANT=1`**. A report that sets
  `attention_config={"backend": "FLASH_ATTN"}` without that env var ran the ordinary FIA plugin backend
  and its crash belongs elsewhere; it is a silent non-selection, not a warning the reporter will notice.
- raises `ValueError("FA3 backend does not support MLA and SFA.")` for MLA/sparse layers, then
  `flash_attn_npu_v3 is not installed but FA3 backend is requested. Please install flash_attn_npu_v3 to
  enable FA3.` (or `… does not provide flash_attn_with_kvcache`) when the dependency is missing.
- on success logs `In training-inference consistency scenario, FA3 will be enabled, which may cause
  performance degradation.` — that line in the reporter's log is the proof FA3 really ran.

## Common rules

- **`507015` has three owners and the argv decides which.** (a) graph capture/replay sync surfacing an
  earlier async failure — `aclgraph-mode-and-capture.md` §C; (b) a CANN AICore kernel (`DFX kernel_name`,
  `L0C`/`EL0004`/aivec) — `device-kernel-faults.md`; (c) this page: `--enforce-eager`, FA3 backend, and a
  batch that mixes decode with prefill. A run that is eager has no capture to blame, so (c) is the default
  once the backend is FA3.
- **FA3 indexes the batch twice.** `query` is **token**-indexed (a decode request with `n` speculative
  tokens contributes `1 + n` rows); `block_tables`, `query_start_loc` and `seq_lens` are **request**-indexed
  (one row per sequence). Any branch that slices the request-indexed arrays with a token count is wrong;
  without speculative decoding `num_decode_tokens == num_decodes`, so the defect is invisible — that
  coincidence is why a spec-decode A/B is the discriminator.
- **The fault surfaces one sync later.** The device fault is raised at the next `.npu()`/stream sync, so
  the deepest frame is usually the *decode* branch's `seq_lens[…].npu()`, not the prefill call that read out
  of range. Do not read the frame as the culprit.
- **The default FIA path does not contain this code.** Removing the backend pin is a complete workaround;
  keeping FA3 is a decision about the speed, not about the model.
- **A mixed-batch crash on this file is a metadata-indexing fault in the repo, so a local patch is a
  legitimate answer.** The merged fix is three lines in one Python function (below), and no release tag in
  the clone carries it — "carry the patch" beats waiting for a release, and it is not a workaround around
  a vendor limitation.

## Worked case #17991 (`v0.23.0rc1`, 8 × 910B3, Qwen3-0.6B, ngram spec decode)

Environment (body + `collect_env`): vLLM `0.23.0`, vllm-ascend `0.23.0rc1`, torch `2.10.0+cpu`,
torch_npu `2.10.0.post2`, CANN `9.0.1`, transformers `5.5.4`; `VLLM_BATCH_INVARIANT=1`,
`VLLM_ENABLE_V1_MULTIPROCESSING=0`; `Qwen/Qwen3-0.6B`, bf16, `max_model_len=2048`, `max_num_seqs=8`,
`max_num_batched_tokens=256`, `block_size=128`, `enforce_eager=True`, `gpu_memory_utilization=0.5`,
`attention_config={"backend": "FLASH_ATTN"}`, `speculative_config={"method": "ngram",
"num_speculative_tokens": 2, "prompt_lookup_min": 2, "prompt_lookup_max": 4}`; eight prompts, two short
and six long, `max_tokens=24`. The small token budget is what builds the mixed batch: short prompts finish
prefill and start decoding while a long prompt is still being chunked.

```
File ".../vllm_ascend/attention/fa3_v1.py", line 122, in forward_impl
    attn_metadata.seq_lens[:num_decodes].npu(),
RuntimeError: ACL stream synchronize failed, error code:507015
[Error]: The aicore execution is abnormal.
EE9999: rtDeviceSynchronizeWithTimeout execution failed, reason=aicore exception
Kernel task happen error, retCode=0x26, [aicore exception].
```

Reporter's own control: the **same workload completes with speculative decoding disabled**.

### Verified code at the reported ref (`v0.23.0rc1:fa3_v1.py`)

- `:108-109` `num_tokens = attn_metadata.actual_seq_lengths_q[-1]`; `query = query[:num_tokens]`.
- `:116-126` decode branch: `query[:num_decode_tokens]`, but
  `block_tables[:num_decodes, :]`, `query_start_loc[:num_decodes + 1]`, `seq_lens[:num_decodes]` — and the
  causal flag is the literal `False` (`:123`).
- `:128-138` prefill branch: `query[num_decode_tokens:]` (correct, token-indexed) beside
  `block_tables[num_decode_tokens:, :]` (`:132`), `query_start_loc[num_decodes:]` (`:133`) and
  `seq_lens[num_decodes:]` (`:134`) — **one token count and two request counts in the same call.**

### Mechanism (`verified` at the ref, and by the fix's own diff)

With `num_speculative_tokens: 2` an ngram decode request carries three tokens, so
`num_decode_tokens = 3 × num_decodes`. At `:132` the prefill slice starts at row `3 × num_decodes` of a
request-indexed table whose prefill rows begin at `num_decodes` — the slice therefore starts past the real
prefill requests (and past the table's end once prefills are few). At `:133` the offsets are still measured
against the *whole* token buffer while `query` was already cut at `num_decode_tokens`, so every prefill row
is offset by the decode token count. The kernel is handed page ids and offsets that do not belong to those
rows; an out-of-range paged read faults on the device and the next sync reports `507015` — which is why the
frame is the decode branch's `:122` while the bad call is the prefill one. `:123`'s `is_causal=False` is the
correctness half: multi-token verification needs causal masking, so a "fix" that only repairs the row index
still produces wrong tokens.

### Fix and status

`5e1647cf9` "[BugFix][Attention] Fix FA3 mixed speculative decode semantics (#18068)", merged to `main`
2026-10-15. Three hunks, all in `fa3_v1.py`: `block_tables[num_decodes:, :]`, a rebased
`prefill_query_start_loc = query_start_loc[num_decodes:] - num_decode_tokens` when
`num_decode_tokens > 0`, and the decode branch's causal flag becomes `num_decode_tokens > num_decodes`
(`main` `:125-146`, `:128`, `:134-136`, `:140`). **`git tag --contains 5e1647cf9` is empty for every tag in
the clone (through `v0.30.0`): the fix is `main`-only**, and `v0.23.0`, `v0.23.0.post1`, `v0.24.0rc1`,
`v0.25.1rc1`, `v0.26.0rc2`, `v0.27.1rc1`, `v0.28.0.rc1`, `v0.29.0.rc1`, `v0.30.0rc1`, `v0.30.0rc2` and
`v0.30.0` (2026-10-29, i.e. after the merge — the branches were cut before it and nothing was
cherry-picked) all still carry `block_tables[num_decode_tokens:, :]` at `:131`/`:132`. The physical-page
addressing half (`_as_fa3_paged_cache`, page-ID remapping) is a **separate** PR (#18729), not in #18068 —
do not promise it. Overlapping PRs #17774 (the request-based slice) and #18070 (an independent mixed-batch
attempt) were credited/superseded; do not report them as separate fixes.

### Workaround ladder (relay in this order)

1. Remove the backend pin (serve the default FIA path): complete, and the defect exists only in
   `fa3_v1.py`; cost = whatever FA3 was selected for (batch-invariant rollouts lose the backend).
2. Keep FA3 without the trigger: no speculative decoding (reporter-confirmed) or a `max_num_batched_tokens`
   large enough that a decode batch never shares with a prefilling chunk; cost = the spec-decode speedup or
   the scheduler shape the workload wanted.
3. Local patch on the reporter's build — the image installs vllm-ascend **editable** at
   `/vllm-workspace/vllm-ascend` (`../vllm-ascend/install-and-image.md` §4): apply the three hunks above.
   Cost = a carried patch, lost on rebuild.
4. A build containing #18068: `main` only at the time of writing. Say "fixed on `main`, in no released tag"
   rather than naming the next release.

### Limits to state

Which of the two defects the reporter's crash needed is `verified` for the metadata slice and `verified
read` for the causal flag (the fix's diff), but the run was never re-executed here; a physical-page
(strided KV) layout can still fail after the metadata fix (that is #18729's scope). No NPU check was done.

### Falsifiers to hand the reporter

- A crash with spec decode **off** on the same FA3 path: not this class.
- A crash with the three hunks applied: go to #18729 (physical page addressing) or to a genuine device
  fault (`device-kernel-faults.md`: a stable kernel name/hash in the plog, no dependence on the batch mix).
- A crash under FA3 with a single-token-per-request decode-only batch: this class predicts it passes.
- A run whose startup log printed *"FA3 will not be enabled …"*: the ordinary plugin backend ran, and the
  crash belongs to `attention-mask/` or `graph-capture-and-custom-ops.md`, not here.

### The reproduction is already in the thread

The reporter's 25-line script is the cheapest repro: Qwen3-0.6B, bf16, `max_model_len=2048`,
`max_num_seqs=8`, `max_num_batched_tokens=256`, `block_size=128`, `enforce_eager=True`,
`VLLM_BATCH_INVARIANT=1`, `attention_config={"backend": "FLASH_ATTN"}`,
`speculative_config={"method": "ngram", "num_speculative_tokens": 2, "prompt_lookup_min": 2,
"prompt_lookup_max": 4}`, eight short/long prompts, `max_tokens=24`, `ignore_eos=True` — it dies at the
first mixed batch and passes with speculation off. #18068 ships the durable version:
`tests/e2e/pull_request/one_card/rlhf/consistency/test_fa3_mixed_batches.py` (registered in
`.github/workflows/scripts/estimated_times.yaml`) plus `test_fa3_mixed_speculative_decoding` in
`tests/e2e/pull_request/one_card/rlhf/consistency/test_attention_fa3.py`.

## Check before reuse

```bash
git -C $R show <reported-ref>:vllm_ascend/attention/fa3_v1.py | grep -n 'block_tables\['   # num_decode_tokens = defect
git -C $R log --oneline -1 <fix-sha>; git -C $R tag --contains <fix-sha>                     # empty = main-only
```

and confirm from the argv that `attention_config` really selected `FLASH_ATTN` (a `Qwen3-0.6B` report
without that key is the FIA path and a different owner).
