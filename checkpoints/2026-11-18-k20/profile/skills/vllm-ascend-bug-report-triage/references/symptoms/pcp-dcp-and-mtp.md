# Context parallelism (DCP/PCP) with MTP: cache-unit misses, missing slot metadata, blanked CPU mirrors

Five failures that only exist once `--decode-context-parallel-size` (DCP) or
`--prefill-context-parallel-size` (PCP) is above 1, and that are routinely mis-merged: a **silent** one
(the prefix cache never hits, no error), three **crashing** ones (`mtp_slot_pad` and `seq_lens_cpu` are
metadata fields produced on one branch and read on another; §E is a numpy broadcast in the same file), and
§D, a **throughput** cliff with no error at all: it is here because the only localised frame is the DSA-CP
all-gather, and it is still open. One exception to the CP gate: with
**no CP at all** on DeepSeek-V4 the compressed-MLA groups multiply the same unit by `compress_ratio`
and the write/lookup keys disagreed until `v0.21.0rc1` — that member is
`prefix-cache-granularity.md` §1, not this page. Every member here is bookkeeping in
`vllm_ascend/core/`, `vllm_ascend/worker/pcp_utils.py` and `vllm_ascend/spec_decode/eagle_proposer.py`,
not kernels.

Read the shape first: no traceback plus a 0 % hit rate → §A; a dead worker whose traceback names
`mtp_slot_pad` → §B; a `NoneType` assignment at `cad.seq_lens_cpu[...]` with `--async-scheduling` → §C;
successful requests that get 2.5–3× slower as the per-engine sequence count crosses a power of two → §D;
`ValueError: could not broadcast input array from shape (29,) into shape (31,)` at
`pcp_utils.py` `update_tokens_for_pcp` — MTP on **and** a hybrid-attention model, on a mixed
prefill+decode step → §E.

## Scope and first reads

Classify before any code read — every member needs CP on:

```bash
cd /workspace/repos/vllm-project/vllm-ascend && R=<tag>
git rev-parse -q --verify $R && git grep -n 'dcp_world_size \* pcp_world_size\|block_size \* dcp' $R -- vllm_ascend | head
```

1. `--decode-context-parallel-size` / `--prefill-context-parallel-size` / `--cp-kv-cache-interleave-size`
   in the start command, plus `--speculative-config` (`num_speculative_tokens`) and
   `--async-scheduling`.
2. `block_size` in the same command (`--block-size 128` is the tutorial default).
3. Whether a metric was measured at all: §A is invisible in the log unless `--enable-prefix-caching`
   was set and the reporter's harness prints hit rate/per-request TTFT; §B and §C are dead workers.

Common rules:

- **CP multiplies the scheduler's block size.** Any count the reporter gives (characters, "I sent the
  same 1000 words twice") must be recomputed as *tokens after that model's tokenizer* and divided by
  `block_size × dcp × pcp`. A claim of "no hit" is consistent with "no complete hash unit exists".
- **Do not merge the symptoms.** No traceback + constant throughput is §A; a worker death with a frame
  in `spec_decode/eagle_proposer.py` is §B or §C, decided by the field named in the traceback. One
  recorded run spent part of two messages blaming MTP for a prefix-cache miss that the reporter had
  already shown persists with `--speculative-config` removed.
- **For §B/§C, find the producer's assignment and compare its guard with the reader's.** A traceback that
  names a metadata attribute rather than a shape is a branch-disagreement question, not a shape bug.

## A. Prefix cache never hits while DCP/PCP > 1

### The same unit with CP off: DeepSeek-V4's compressed-MLA groups

A DSv4 hit-rate report needs the granularity page even when the command has no CP: the compressed-MLA
group's unit is `block_size × compress_ratio` original tokens, `block_size` is pinned at 128 for
`model_type == "deepseek_v4"`, and the group's write and lookup paths disagreed until `v0.21.0rc1`. All
of that, with the fix commits, the measured block-size table and the PD fan-out question, is
`prefix-cache-granularity.md` §1–§3 — do not re-derive it here; only the `× dcp × pcp` multiplier above
is this section's.

Worked case: vllm-ascend #13750 — `quay.io/ascend/vllm-ascend:v0.23.0rc1`, A2/A3, GLM-5.2 w4a8c8,
TP8 + EP, `--decode-context-parallel-size 8`, `--enable-prefix-caching`,
`--max-num-batched-tokens 8192`. Repeated identical `curl` requests; `prefix cache hit rate` stays 0; no
error; the reporter also re-tested with 2048-token and ~2000-token inputs and **with
`--speculative-config` removed**.

### Verified code at `v0.23.0rc1`

- `vllm_ascend/patch/platform/patch_kv_cache_utils.py:45-47` —
  `if len(groups) <= 1: bs = cache_config.block_size * dcp * pcp; return bs, bs`
  (the single-KV-cache-group case, i.e. most dense/MLA deployments).
- Same file `:55-58` — multi-group: `scheduler_block_size = lcm(group_block_sizes) * dcp * pcp`, and
  with prefix caching on it returns `(scheduler_block_size, math.gcd(*group_block_sizes))` as
  `(block_size, hash_block_size)`.
- `vllm_ascend/core/single_type_kv_cache_manager.py:215-216` — `find_longest_cache_hit` repeats the
  inflation (`block_size *= dcp_world_size * pcp_world_size`), then `:218-220`
  `logical_block_size = block_size * compress_ratio`, `max_num_blocks = max_length // logical_block_size`,
  so a request shorter than one logical block produces **zero** lookups rather than a miss;
  `:234-237` `if eagle_drop and computed_blocks[0]: computed.pop()` drops the last matched block when
  speculation is on; `:239-243` pops matched blocks until the hit length is a multiple of
  `alignment_tokens`.

### Mechanism

The block-hash chain is built at the *scheduler* block size, which CP multiplies. With `--block-size 128`
and `dcp=8` one hash unit is 1024 tokens (`× compress_ratio` for the DSv4/DSA compressed specs); a prompt
below that cannot produce a single hit whatever the request count. Above it, hits land in 1024-token
steps and `eagle_drop`/alignment can still discard the last one.

### Workaround (reporter-confirmed) and cost

- Send a prompt whose **tokenized** length is a multiple of `block_size × dcp × pcp` and at least one
  unit (≥ 1024 tokens for `dcp=8`, `--block-size 128`). The reporter's closing comment on #13750:
  "开启DCP PCP，block_size会按倍数放大 … 比如128会变成128*8=1024才能命中；而且输入不能只看字符长度，要看tokenization之后是否满足bs".
- Or lower `dcp × pcp`, or `--cp-kv-cache-interleave-size`, to shrink the unit — at the cost of the
  parallelism the deployment was sized for.
- Nothing here is a code defect to file: it is the CP design. Report it as a **doc gap** (the tutorial
  does not state the inflated hit unit) rather than an engine bug.

### Evidence caution

The "Automated Response" comment on #13750 ("DCP-inflated prefix-cache units combined with MTP
last-block drop", citing upstream vllm-project/vllm#50438 and #50897) was **retracted by its own author**
two hours later ("If a long curl request still doesn't hit the prefix, then that's not the issue I was
talking about"). Treat bot/automated triage comments as routing notes, and read the reporter's own
closing comment, which here is the answer.

### Check before reuse

`git grep -n 'block_size \* dcp\|scheduler_block_size' <ref> -- vllm_ascend/patch/platform` and the
`dcp_world_size * pcp_world_size` sites in `vllm_ascend/core/`. `BlockHashListWithBlockSize` itself is
**upstream** vLLM (`vllm/v1/core/kv_cache_utils.py:2093` at `v0.23.0`), imported by
`vllm_ascend/core/single_type_kv_cache_manager.py:11`; an empty grep for it in the vllm-ascend clone is
not evidence of absence. If the tag returns `hash_block_size = gcd(group_block_sizes)` with prefix
caching on, hashing is finer than the scheduler block and §A's arithmetic changes — re-derive it at the
ref instead of quoting 1024.

## B. `PCPManager` has no `mtp_slot_pad` on inputs shorter than k+1

Worked case: vllm-ascend #8578, twin (more detailed, `wontfix`, open) #8402 — "PCP and eagle3
overlaying problem". DeepSeek-V3.2-W8A8, 2 nodes, TP8 + EP, `--decode-context-parallel-size 8`,
`--prefill-context-parallel-size 2`, `--cp-kv-cache-interleave-size 128`, MTP spec decode; the worker
dies after a single short `curl`, or when such a request is swapped out during chunked prefill:

```text
eagle_proposer.py:606, in _propose
    mtp_slot_mapping = self.runner.pcp_manager.mtp_slot_pad
AttributeError: 'PCPManager' object has no attribute 'mtp_slot_pad'
```

### Verified code at `v0.18.0rc1`

- Producer: `vllm_ascend/worker/pcp_utils.py:584` `if self.decode_threshold > 2 and not with_prefill:`
  … `:607 self.mtp_slot_pad = mtp_slot_pad.to(self.device, non_blocking=True)` — the attribute exists only
  after that branch runs (`with_prefill` is a method parameter, default `True`).
- Consumer: `vllm_ascend/spec_decode/eagle_proposer.py:592` `if self.pcp_size * self.dcp_size > 1:`,
  `:593 if self.num_speculative_tokens > 1 and not attn_metadata_i.num_prefills:`,
  `:606 mtp_slot_mapping = self.runner.pcp_manager.mtp_slot_pad` — unguarded attribute read.

### Mechanism

[reported by the twin's author, consistent with the two gates above] A request with fewer than
`num_speculative_tokens + 1` scheduled tokens is classified **decode** by the attention metadata
(`num_prefills == 0`) while the runner's PCP path still considers the batch a prefill (`with_prefill`
true), so the producer branch is skipped and the consumer reads an attribute that was never assigned.

### Version drift (verified — match the exception, not the wording)

The attribute was later pre-declared: at `v0.23.0rc1` `pcp_utils.py:124`
`self.mtp_slot_pad: torch.Tensor | None = None` and `:1341 assert self.mtp_slot_pad is not None` in
`_get_spec_decode_mtp_slot_inputs`, while the producer keeps the same `not with_prefill` gate
(`:1717 if self.decode_threshold > 2 and (not with_prefill or needs_dcp_prefill_slots)`). On those tags
the same trigger surfaces as **`AssertionError`**, not `AttributeError`. Still unguarded at `v0.21.0rc1`
(`spec_decode/llm_base_proposer.py:848`, `pcp_utils.py:955-957`) and gone from `main` only because #12592
(`69f72c642`) removed PCP from MRV1 — do not read its absence at `main` as a fix.

### Status

Maintainer comment on #8402 (COLLABORATOR): "There's a known issue while enabling PCP and EAGLE3 which
could not be resolved", labels `bug` + `wontfix`, still open. The symbol's presence across tags is not
monotone — measured with `git grep -c mtp_slot_pad <tag> -- vllm_ascend`: present at `v0.18.0`,
`v0.20.2rc1`, `v0.21.0rc1`, `v0.23.0rc1`; absent at `v0.19.0rc1`, `v0.20.0rc1`, `v0.22.0rc1` (PCP code
moved between MRv1 and MRv2 paths and release branches carry different cherry-picks). Never infer
"fixed" from one tag's empty grep: check the tag the report names, and read the *guard* around the read.

### Workaround (proposed, not reporter-confirmed)

- Drop PCP, keep MTP: `--prefill-context-parallel-size 1` (the failing block is gated by
  `pcp_size * dcp_size > 1`). Largest capability loss, no crash.
- Or keep PCP and drop speculation (`--speculative-config` removed / `num_speculative_tokens: 1`, which
  also skips the `> 1` guard).
- Or keep both and enforce `tokenized_len ≥ num_speculative_tokens + 1` at the gateway, plus enough KV
  headroom that such requests are not preempted mid-chunked-prefill.
- Disabling a feature is a bisect tool, not a fix: say which of these the reporter confirmed.

### Check before reuse

`git show <ref>:vllm_ascend/worker/pcp_utils.py | grep -n 'mtp_slot_pad'` and the same for
`spec_decode/eagle_proposer.py`; if the read is inside a `try/except AttributeError`, or the producer gate
is now unconditional, the class is closed at that ref. A missing attribute from another manager
(`pcp_manager.<x>`) is a different field — read the name in the traceback.

## C. Async scheduling blanks `seq_lens_cpu` and the PCP splice writes into it

Worked case: vllm-ascend #8926 — title `0.19.0rc1`, image `quay.io/ascend/vllm-ascend:v0.19.1rc1`, 2×A2,
DeepSeek-V3.1-w4a8-mtp-QuaRot, TP8 + EP + `--prefill-context-parallel-size 2`, `--async-scheduling`,
`{"num_speculative_tokens": 3, "method": "deepseek_mtp"}`, MLAPO/FLASHCOMM1 on,
`--max-num-seqs 512 --max-num-batched-tokens 32768`, `FULL_DECODE_ONLY`. The worker dies on a mixed
prefill+decode draft step:

```text
TypeError: 'NoneType' object does not support item assignment
  vllm_ascend/spec_decode/eagle_proposer.py:1158 in set_inputs_first_pass
      cad.seq_lens_cpu[-num_prefill_reqs:] = seq_lens_p
```

- Producer: `worker/model_runner_v1.py:2464-2480` (the `use_async_spec_decode` block at `:2466-2470`)
  deliberately sets `seq_lens_cpu = None; num_computed_tokens_cpu = None` — "GPU tensors are
  authoritative in async mode" — and passes the populated CPU view only as `_seq_lens_cpu`.
- Consumer: `set_inputs_first_pass`, in the PCP splice branch (`pcp_size > 1` and `num_prefill_reqs > 0`,
  i.e. a new request chunk-prefilling next to decodes), writes both `cad.seq_lens[-num_prefill_reqs:]`
  and `cad.seq_lens_cpu[-num_prefill_reqs:]` unconditionally. `_seq_lens_cpu` (upstream-canonical,
  always filled) is not the field that crashes; `seq_lens_cpu` (the Ascend subclass field) is.
- Gating: PCP (or DCP) **and** MTP **and** `--async-scheduling`; removing any one avoids it.

Version note: the issue **title** said `v0.19.0rc1` while the image was `v0.19.1rc1`. The traceback's
`:1158` exists only at `v0.19.1rc1` (`v0.19.0rc1` has no such line), so the image tag is the built ref —
one line fingerprint settled a prose conflict (`../workflow/version-pinning.md`).

Fix (verified): PR #8749 = `931604054` *"[BugFix] fix dcp async bug"* (2026-05-08) guards both writes,
keeps the parent and subclass mirrors aligned in the extend path (`new_cad._seq_lens_cpu =
cad._seq_lens_cpu + N`) instead of rebuilding the mirror from GPU, and adds
`test_qwen3_eagle3_pcp2_tp1`. First tag containing it: `v0.20.2rc1` — so it is **not** in `v0.19.1rc1`,
the reporter's image. Thread: maintainers named #8749 twice and the reporter confirmed "Verified, it
resolves my issue".

Workaround on the reported build: drop `--async-scheduling` (keeps PCP + MTP, loses the scheduler/model
overlap) or drop `--prefill-context-parallel-size` (keeps async + MTP); removing spec decoding is a third,
purely diagnostic, option. Durable: a build containing #8749.

### Check before reuse

```bash
git grep -n 'seq_lens_cpu\[-num_prefill_reqs:\]' <tag> -- vllm_ascend/spec_decode/eagle_proposer.py
git grep -n -A3 'use_async_spec_decode' <tag> -- vllm_ascend/worker/model_runner_v1.py
```

Unguarded at the tag plus `seq_lens_cpu = None` under async ⇒ this class. Falsifier: the same `TypeError`
with `--async-scheduling` absent means a different `None` producer — read the metadata constructor the
traceback names, do not assume this branch.

## D. A silent throughput cliff as the *running-seq count* grows (DSA-CP, open — #14440)

Not bookkeeping like A–C and not a crash: the service is correct, and decode throughput falls 2.5–3× the
moment the running sequences per engine cross a power of two. Filed at `v0.23.0rc1` on A2 (910B2),
DeepSeek-V4-Flash-w8a8-mtp, TP8 + EP, `--data-parallel-size 1 --max-num-seqs 264`, DSA-CP and
flashcomm1 on, `FULL_DECODE_ONLY`, MTP 1. **Open, no mechanism named by the maintainers**; the value of
this section is the measurements already taken and the one search left, not a cause.

- The reporter's own table at N=264 (`--max-num-seqs` set equal to N, so nothing is admission-queued and
  `vllm:*_waiting` stayed 0) rules out most of the obvious owners, each with its own A/B: MTP off
  (712 tok/s), `--enforce-eager` (805), `enable_npugraph_ex: false` (1062), `--async-scheduling` off (795),
  `multistream_overlap_shared_expert: false` (1135), `VLLM_ASCEND_ENABLE_FLASHCOMM1` unset **with
  `enable_dsa_cp: false`** (1277), `--max-model-len 1024` (cliff still at 256→258). All still collapse
  against 3 048 tok/s at N=256. So: not graph capture, not MTP, not async scheduling, and **not DSA-CP
  either** — the run with `enable_dsa_cp: false` collapsed too, which is the fact that kills the tidiest
  story (see the next bullet).
- What py-spy does show is a worker parked in `dist.all_gather_into_tensor` reached through
  `_maybe_all_gather_and_maybe_unpad_impl` / `dsa_cp.py:1296` (at `v0.23.0rc1` that line is
  `torch.ops.vllm.maybe_all_gather_and_maybe_unpad(hidden_states_local, need_gather_q_kv)`; the `_impl` is
  upstream vLLM). On NPU, host and device streams execute in order, so that frame marks **where the stream
  waits**, not what made it slow — and the reporter's bare-HCCL microbenchmark (`all_gather_into_tensor`,
  torchrun ×8, bf16, hidden 5120) is flat at ~0.3 ms/op from 256 to 640 tokens, so the collective itself is
  excluded.
- The shape that still points somewhere: the cliff moves with TP (TP4 collapses between 224 and 256 per
  engine, TP8 exactly at 256) and disappears at `dp=2/tp=4` with ≤ 224 sequences per engine (peak
  4 561 tok/s). That is a host-side structure indexed by the running-seq count — worth **one** grep
  (`git grep -nE '(>|>=) *256|[^0-9]256[^0-9]' <ref> -- vllm_ascend`) and one read of whatever it names,
  not another round of feature toggles.
- Answer shape while it is open: the working deployment (`--data-parallel-size 2/`TP4`, i.e. ≤ 224
  sequences per engine) as the workaround, `--max-num-seqs`/client concurrency capped below the cliff as
  the cheaper one, and the mechanism as `hypothesis` with the falsifier "a candidate structure whose
  threshold is not 256, or a cliff that does not move with TP, kills it".
- Do not spend a run re-deriving the reporter's A/B table or re-reading the DSA-CP attention body: the
  class is measured, the collective is excluded, and the remaining question is a constant in the host path.

## E. `ValueError: could not broadcast input array from shape (29,) into shape (31,)` — a token count used as a request index (MTP + PCP, #11231)

Reported on `v0.21.0rc1` and again on `v0.22.1rc1`: Qwen3.5-27B (hybrid GDN, `VLLM_ASCEND_GDN_FAST_PATH=1`) on
A3, TP2, `--prefill-context-parallel-size 2 --decode-context-parallel-size 1`, `--max-num-seqs 32`,
`FULL_DECODE_ONLY`, `--async-scheduling`, `--speculative-config '{"num_speculative_tokens": 2,
"method":"mtp"}'`. The service starts and serves; under an aisbench mixed load every worker dies:

```text
vllm_ascend/worker/model_runner_v1.py:839 in _prepare_inputs
  → vllm_ascend/worker/pcp_utils.py:667 in update_tokens_for_pcp
      num_padded_scheduled_tokens[self.num_decode_reqs :] = num_prefill_scheduled_tokens_linear
ValueError: could not broadcast input array from shape (29,) into shape (31,)
```

### Verified code at `v0.22.1rc1` (identical at `v0.21.0rc1`)

- `:194` `self.num_decode_tokens = num_scheduled_tokens[: self.num_decode_reqs].sum()` — a **token** count.
- `:660` `prefill_tokens_tensor = torch.Tensor(num_scheduled_tokens[self.num_decode_tokens :])` — that token
  count used as an **index into a per-request array**; the docstring at `:530` says the array is "1D numpy
  array of length num_reqs containing the number of new tokens scheduled per request".
- `:667` assigns the result into `num_padded_scheduled_tokens[self.num_decode_reqs :]`, a target of
  `num_reqs - num_decode_reqs` rows.
- Gate: `if self.pcp_use_hybrid_attn:` (`:650`), which is a **model-type** test —
  `hf_config.model_type in ("qwen3_next", "qwen3_5", "qwen3_5_moe")` (`:131-135`) — plus a mixed
  prefill+decode step (`:659 num_reqs - num_decode_reqs > 0`), MTP with `k > 0`, and a `PCPManager` at all
  (`pcp_size * dcp_size > 1`, `model_runner_v1.py:417-419`).
- The same wrong index is at `:759` (`ori_tokens_start_loc`); `:684` uses `num_decode_reqs`, the shape the
  other two sites should have had. One site mixes the two spaces — it is not the variable that is wrong.

### Mechanism

With speculation each decode request schedules `1 + num_speculative_tokens` tokens, so
`num_decode_tokens = num_decode_reqs × (1 + k)` and the source is short by `num_decode_reqs × k` rows while
the target stays `num_reqs - num_decode_reqs`. Both halves are in the reporter's own log: `1 generation
requests, 3 generation tokens` with `num_speculative_tokens: 2`, and the reported shapes
`29 = 32 − 3` / `31 = 32 − 1` for `Running: 32 reqs` with one decode request. With `k = 0` the index happens
to equal the count, which is why the reporter sees "MTP alone works, PCP alone works" — the code predicts
exactly that. Any `k > 0` on a **mixed** batch fails; `num_speculative_tokens: 1` is not a lever here (it
still schedules 2 tokens), unlike §B where the guard is `> 1`. [reported] the thread's only diagnosis is the
maintainer's "open only MTP, close PCP" ask, and the reporter answers that MTP alone was already clean.

### Fix and status

`c6343b13b` (#10487, "qwen3.5 support pcp+mtp", 2026-06-23) rewrites both sites to
`num_scheduled_tokens[self.num_decode_reqs :]` (and the `ori_tokens_start_loc` roll), moving several sibling
sites the other way; first tag containing it is **`v0.23.0rc1`** (also `v0.23.0`). It is **not** in
`v0.21.0rc1` or `v0.22.1rc1` — release-branch tags that carry the pre-#10487 file. The reporter's "PR #10172
is already merged in v0.22.1rc1 and it still fails" is correct and unrelated: #10172 (`2221a9e71`, "fix
qwen3.5+pcp+mtp single batch error") is an ancestor of both tags but its `--stat` is
`attention/context_parallel/attention_cp.py`, `ops/triton/mamba/causal_conv1d.py`,
`spec_decode/llm_base_proposer.py` — never `pcp_utils.py`. **#10907** `709bd7bd2` (2026-06-26, same first
tag) is the accuracy follow-up on `adjust_cu_num_scheduled_tokens_for_pcp` and the all-gather lengths for the
same pairing. A PR *title* that matches the symptom is not the fix; `git show --stat <sha>` names the files
in one command.

### Levers

| tier | change | cost / status |
| --- | --- | --- |
| keep PCP, drop speculation | remove `--speculative-config` (`k = 0` makes the index equal the count) | loses the MTP speedup; `proposed`, and the reporter's own "MTP only = no error" says the other half is PCP |
| keep MTP, drop PCP | `--prefill-context-parallel-size 1` (with `dcp_size` also 1 the `PCPManager` is never constructed, `model_runner_v1.py:417-419`) | loses PCP; `proposed` |
| port the fix | change `:660` and `:759` to `num_scheduled_tokens[self.num_decode_reqs :]`, exactly as `c6343b13b` does | local patch on a 0.21/0.22 build; `proposed`, one hunk |
| upgrade | a build with #10487: `v0.23.0rc1`+ | version jump; re-check the other 0.22→0.23 deltas |

### Check before reuse

```bash
git show <ref>:vllm_ascend/worker/pcp_utils.py | grep -n 'num_scheduled_tokens\[self.num_decode'
git tag --contains c6343b13b | sort -V | head -3
```

`_tokens` at the `prefill_tokens_tensor` site is the bug, `_reqs` is the fix. At `v0.23.0rc1` the hybrid
branch is gone and the same routine works in `cu_num_scheduled_tokens` / `num_decode_reqs` space
(`:352-357`), so the class is closed there. A `merge-base --is-ancestor` **NO** on a main-line sha does not
by itself prove a release lacks the change (release branches cherry-pick under new shas), so read the tag's
tree as well — here it agrees. Falsifier: the same broadcast with `--speculative-config` removed ⇒ a
different producer (then read how `num_decode_reqs` itself is derived — the 0.23+ reorder logic is
`5bd4e3950`/#10580's `decode_req_mask`), and the same message with `pcp_use_hybrid_attn` false (a non-hybrid
model) means a different line. `pcp_utils.py` is absent from `main` and from `v0.25.1rc1`+ because PCP was
removed from MRV1 (#12592, `69f72c642`) — that absence is a removal, not a fix.
