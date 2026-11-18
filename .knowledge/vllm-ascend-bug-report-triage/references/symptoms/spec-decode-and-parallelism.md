# Speculative decoding and parallel-feature accuracy

Two families that look alike ("throughput collapses", "accuracy is wrong") but have different
mechanisms and different evidence: the MTP placeholder bug is *verified at the commit level*, while
the parallel-feature accuracy cases are *open with a suspect set*. Keep the label honest. A third member
of the spec-decode family sits under §2 and is **resolved**: the `507034` + `507035` device death of
DeepSeek-V4-Flash/Pro on A3 PD under DSpark, fixed in `v0.26.0rc2` (case #15985) — read it when the
symptom is a crash rather than a score. A fourth (§1b) is neither accuracy nor a score: a **stall**
on a P/D + MTP engine, whose one verified fact is a fork override that missed its upstream fix.

Not on this page: the other PCP/DCP + MTP failures — a crash inside the draft proposer's metadata
plumbing (`mtp_slot_pad` never created, `cad.seq_lens_cpu` blanked by async scheduling) and a
prefix-cache **miss** that is block-size arithmetic — because their evidence is a producer/consumer
guard mismatch or a unit calculation rather than an accuracy delta. Read the shape first: exception in
`eagle_proposer.py` → `pcp-dcp-and-mtp.md` §B/§C; healthy service with 0 % cache hits under DCP/PCP →
`pcp-dcp-and-mtp.md` §A; wrong numbers, no exception → §2 here.

## 1. Spec-decode acceptance rate collapsing to zero (MTP placeholder)

### Symptom

With MTP speculative decoding (`qwen3_5_mtp`, `num_speculative_tokens: 3`), long contexts and high KV
occupancy, acceptance rate drops to 0 after several rounds and throughput goes to 0 — the server
appears stuck while requests remain queued.

### Code

`vllm_ascend/sample/rejection_sampler.py` (placeholder masking and rejection paths),
`vllm_ascend/worker/v1/model_runner.py` (`PLACEHOLDER_TOKEN_ID` bookkeeping, masked fill of padded
input ids), `vllm_ascend/core/recompute_scheduler.py` (at `v0.20.2rc1:152` it injects
`[PLACEHOLDER_TOKEN_ID] * num_spec_tokens` into `spec_token_ids`). The upstream constant is imported
from `vllm.v1.sample.rejection_sampler`.

### Mechanism (from commit `7cb5d6180`, verified text)

On the first decode step of a PD KV-consumer node, MTP keeps `PLACEHOLDER_TOKEN_ID` (`-1`) in
`spec_token_ids` to preserve the full-graph decode shape. It is a scheduling placeholder, not a
vocabulary id. A sampler that uses it as an index into target/draft probability tables reads the last
vocabulary row (Python negative indexing) or out-of-range memory, so every draft token is rejected
and acceptance goes to zero; Triton kernels hit the same bad index.

### Version status (verified)

For `v0.20.2rc1` the backport `7cb5d6180` (#10259) and the follow-up `91e6b539d` (#10308) are **not**
ancestors of the tag, while `367b8e62d` (#9946) is. Reports against `0.20.2rc1` therefore predate the
rejection-sampling guard. The mainline originals `a3b10cb6f` (#10043: reject placeholder draft
tokens, never index probability tables with `-1`, no Python negative indexing) and `1a44dc5f2`
(#10307: sanitise placeholders before model forward) are in `v0.23.0rc1`+.

### Workaround

Move to a build containing the guard (the release-branch commit or later), or temporarily disable
speculation (`--speculative_config` removed / `num_speculative_tokens: 0`) to confirm the diagnosis.
Disabling is a bisect tool, not a fix.

### Check before reuse

`git show <ref>:vllm_ascend/sample/rejection_sampler.py | grep -n PLACEHOLDER_TOKEN_ID`. An
acceptance drop with no PD/placeholder involvement is a different bug (draft-model weights,
block-table or position errors) — do not force it into this class.

## 1b. A P/D + MTP engine that stops logging after an hour and is dead hours later (#8916)

Symptom, verbatim from the report: 「the log will stuck, and many hours later, it shows the server shot
down」 — no traceback, no plog, one serve block on `base:0.18.0rc1` (Qwen3.5-MoE, PD disaggregation,
MTP). It reads like a collective or a KV-transfer wait, and it is **not** localisable from the body;
the lever was already in the thread (the reporter's own 「this bug fixed with these three pr」).

### Verified at the tag

`v0.18.0rc1:vllm_ascend/worker/model_runner_v1.py:2045-2048`, in `_build_attention_metadata` under
`use_spec_decode and self.need_accepted_tokens`, copies the CPU mirror
`input_batch.num_accepted_tokens_cpu` into `self.num_accepted_tokens.np[:num_reqs]` and pushes it back
with `copy_to_gpu()` with **no event and no synchronize**; `git grep -n num_accepted_tokens_event
v0.18.0rc1 -- vllm_ascend` is empty. `9a0b786f2` (#8764, base `releases/v0.18.0`, 2026-04-27) adds
two lines at `:2044-2045`:

```python
if self.num_accepted_tokens_event is not None:
    self.num_accepted_tokens_event.synchronize()
```

`merge-base --is-ancestor 9a0b786f2 v0.18.0rc1` → no; `git tag --contains 9a0b786f2 --list 'v0.*'`
→ **`v0.18.0` only**. The same build predates `696dcc926` (#8531, sampler mask / invalid
`draft_token_id` guards in `ops/triton/reject_sample.py` + `ops/triton/spec_decode/utils.py`), also
first tag `v0.18.0`.

### The reusable shape: a paired-upstream fix does not reach a fork override

The identical fix is upstream vLLM's `be0a3f7570` (#36013) and it **is** in `vLLM v0.18.0rc0`+ — where
`num_accepted_tokens_event` is created (`v0.18.0:vllm/v1/worker/gpu_model_runner.py:794-797`), recorded
(`:1379`) and synchronised (`:1925-1926`). The paired upstream therefore had the guard while the fork
did not, because `vllm_ascend/worker/model_runner_v1.py` **overrides** `_build_attention_metadata`.
When a report's symptom is a race, a missing sync or a stale CPU mirror and the paired upstream looks
fixed, check whether the failing method is a fork override before blaming dependency pins.

### Thread-reading rule for this case

The reporter listed **three** PRs. Two are merged into `releases/v0.18.0` and inside the `v0.18.0` tag;
#7904 ("Replace decoder MTP padding token_id") is `merged: false`, base `main`, and touches only
`core/recompute_scheduler.py` — which still injects `[PLACEHOLDER_TOKEN_ID] * num_spec_tokens` at
`v0.18.0:152`. Relay the merged pair and the tag; say that the third is unmerged rather than repeating
the list as a fix set (the same rule as `../workflow/fix-discovery.md`'s "merged: true **and** an
ancestor of a tag").

### Lever, status, falsifier

Move to `v0.18.0` (both merged fixes inside) or cherry-pick `9a0b786f2` + `696dcc926`; on rc1, drop
MTP on the **decode** node first (`num_speculative_tokens` → off) as the bisect, at the cost of the
speculation speedup. Status: reporter-confirmed lever, `verified` missing sync, and the *hang* mechanism
(stale accepted-token count → mis-sized speculative step → a decode side waiting on a transfer that
never completes) stays `hypothesis` with the plog/`npu-smi` print at hang time as the falsifier: a
stall that survives MTP off, or a device/collective error in the plog, is a different class. The
accuracy sibling on the same build is #8932 (`../issue-ledger/reports-later.md`).

## 2. Accuracy loss that appears only with a parallel/KV feature

Family: the service answers, no garbage characters, but a benchmark score drops or reasoning
arithmetic goes wrong, and only when PCP/DCP/EP-off/PD/spec decode/prefix caching is in the mix.

### Triage order

1. **Keep the reporter's control measurement.** "94 with EP, 73 without" (#8579) is a differential
   experiment; it localises the fault to the configuration delta, which is far more than "accuracy
   is bad". Name the differential (feature on vs off, numbers, version) before naming a cause.
2. **Check the feature exists at the reported ref and that its flags are honoured.**
   `git grep -n "<flag or config field>" <ref> -- vllm_ascend`. A flag absent at that tag was ignored
   by that build.
3. **Read the config gates that silently rewrite the user's settings.** At `v0.18.0rc1`
   `vllm_ascend/platform.py` (~lines 448–472):
   - PD (`kv_transfer_config` set) + `cp_size>1` + `block_size != cp_kv_cache_interleave_size` →
     `AssertionError` ("needs to be equal if use pcp or dcp > 1 in P/D disaggregate and kv pool
     scenario");
   - sparse model (`index_topk` in `hf_text_config`) + `cp_size>1` + mismatch → warning and
     `cp_kv_cache_interleave_size` is **overridden** to `block_size`. A user's
     `--cp-kv-cache-interleave-size 128` is not necessarily what the kernel ran with.
4. **Map the feature to its modules and diff the on/off paths.** At `v0.18.0rc1`:
   `vllm_ascend/attention/context_parallel/{sfa_cp,mla_cp,common_cp,attention_cp}.py` (SFA/MLA CP,
   `cp_local_block_size`, LSE merge in `common_cp.py`), `vllm_ascend/worker/pcp_utils.py`
   (`PCPManager`), and `vllm_ascend/ops/fused_moe/prepare_finalize.py` (~397–410 all-gather of
   `hidden_states`/`router_logits` across the PCP group, ~469–470 reduce-scatter). Note:
   `dsa_v1.py`, `dsa_cp.py`, `indexer.py` do **not** exist at `v0.18.0rc1` — do not cite them for a
   0.18 issue.
5. **If the symptom is intermittent, change the hypothesis class.** "Sometimes right, sometimes
   wrong" points at a race or a cache-state mismatch (KV readiness, prefix-cache reuse, async
   scheduling), not at a deterministic arithmetic bug. Search later fixes before theorising:
   `git -C <repo> log --oneline <ref>..HEAD -i --grep="pd\|prefix\|dspark\|accuracy"`.

### Case #8579 — PCP without EP (open)

Report: DeepSeek-V3.2-W8A8, vLLM 0.18.0 / vllm-ascend `v0.18.0rc1`, 2 nodes × TP8,
`--prefill-context-parallel-size 2`, `--decode-context-parallel-size 1`,
`--no-enable-expert-parallel`, `--cp-kv-cache-interleave-size 128`, `--max-model-len 64k`. Score 94
with EP enabled, 73 without; the model produces wrong arithmetic rather than garbled text.

Confirmed workaround (maintainer, 2026-04-23): "Yes, known issue. Temporarily enable EP parallel
processing to work." Labels: `bug`, `triaged`, `advanced-features`, `long-seq`, `llm-model`,
`deepseek`. `[thread]` — evidence of the workaround, not of the mechanism.

Root cause: **not established** in the thread. Code facts relevant to the class `[code]`:

- `vllm_ascend/ops/fused_moe/prepare_finalize.py` at `v0.18.0rc1`: the MoE prepare pads to
  `_EXTRA_CTX.max_tokens_across_pcp`, records `self.num_tokens_pcp`, then `all_gather`s hidden states
  and router logits over the PCP group; `_finalize_with_dp_group` reduce-scatters over PCP **without**
  slicing back to `num_tokens_pcp`, while the neighbouring DP branch does slice
  (`hidden_states[: self.num_tokens]`).
- That missing slice was added later by `5e45ac996` — "[BugFix] fix qwen3.5 accuracy bug when sett
  cp-kv-cache-interleave-si… (#10442)", 2026-06-16, first tag containing it `v0.22.1rc1`; the same PR
  also rewrote `worker/block_table.py` slot-mapping/block-size handling under CP and touched
  `attention/context_parallel/attention_cp.py` and `worker/model_runner_v1.py`. Its own description:
  with `block_size > 1024` the block table stored the kernel block size, breaking slot mapping so
  allocation was not evenly distributed across cards, causing precision problems.

So for any "CP/PCP accuracy" report below `v0.22.1rc1`, first check whether the PCP finalize slice and
the block-table slot mapping carry that fix. `[lead]` — the #10442 PR text names qwen3.5 and
interleave size 1024, while #8579 is DeepSeek-V3.2 with 128; the code overlap is real but the
identification is not confirmed by the thread. Report the differing regions as the suspect set, not
as the cause; any mechanism invented from reading alone (e.g. padded rows surviving the PCP
reduce-scatter) must be labelled a hypothesis.

### Case #15985 — DeepSeek-V4-Flash/Pro on A3 PD: `507034` + `507035`, fixed in `v0.26.0rc2` (verified)

Report: vllm-ascend `v0.26.0rc1`, DeepSeek-V4-Flash/Pro, A3, PD deployment; the engine dies with
`MoeDistributeDispatchV2` timeout `507034` and an `IndexCheck_int64_high_performance_0` `507035`, then
`The vector core execution is abnormal` / `rtStreamSynchronizeWithTimeout execution failed`. **This is the
one case in this family with a maintainer-named, tag-verified fix**, and it is worth reading before the
open cases above, because the two commits show what a DSpark-boundary defect looks like in code.

Maintainer `underfituu` 2026-09-18: 「The issues described above have been resolved in `v0.26.0rc2`」 with
three PRs — #16712 (empty DSpark context), #16713 (skip over-limit DSpark drafting with DP alignment),
#16714 (validate `stop_token_ids` / `allowed_token_ids` before device-side logits indexing). All three are
cherry-picks onto `rfc/0.26.0rc1-bugfix`, so **the released commits carry different PR numbers and a tag
prefix in the subject** (`#16712`→`155f68974`, `#16714`→`69e54c8b7`); find them with
`git log --all --oneline -i --grep='<PR title words>'`, never by ancestry of the mainline PR
(`../workflow/fix-discovery.md`).

**Verified hunk, and the mechanism the report's `507034` needs.** At `v0.26.0rc1`,
`vllm_ascend/ops/triton/spec_decode/utils.py:120` in
`copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid`:

```python
last_pos = tl.load(target_positions_ptr + valid_ctx_end - 1)     # :120 at v0.26.0rc1
```

It assumes a request keeps at least one context token. When an asynchronous speculative step is
**discarded** (every context token rejected), `valid_ctx_end == ctx_start`, so `valid_ctx_end - 1` reads
the *previous* request's position — or, for the first request, before the tensor — and
`query_pos = last_pos + 1 + q_idx` derives every drafted position from it; the bogus positions reach the
MoE dispatch kernel and the device faults. The fix (`155f68974`, in `v0.26.0rc2` by
`git tag --contains`) replaces the read with `has_valid_context = valid_ctx_end > ctx_start` plus a
`tl.where` clamp to `ctx_start` and a `last_pos - 1` correction. The `507035` half is separate and
server-side: unvalidated `stop_token_ids`/`allowed_token_ids` were indexed into logits on device, so an
out-of-vocab id traps the CANN `IndexCheck` kernel (#16714 adds
`vllm_ascend/patch/platform/patch_stop_token_ids_validation.py`, which rejects the request with HTTP 400
and self-disables once the bundled vLLM ships `_validate_stop_token_ids`, upstream vllm#54196). Which of
the two a given log shows is decidable: an `IndexCheck…_0` trap with a client that sends token-id filters
is the validation half; a vector-core timeout on `MoeDistributeDispatchV2` in a DSpark run is the
discarded-step half.

- **Workaround**: `vllm-ascend >= v0.26.0rc2` for DeepSeek-V4-Flash/Pro on A3 PD (maintainer-confirmed);
  cost is a full image bump plus re-validation of the DSpark/Mooncake config. On rc1 the only mitigations
  are to keep requests clear of the model-length limit (so no step is discarded) and to send no
  out-of-vocab token ids — both `proposed`, not verified.
- **Third fix not read**: #16713 (over-limit DSpark drafting with DP alignment, `1653ec76b`/`6f9e86a71`
  lineage) was listed by the thread but its diff was not opened; cite it as one of the three, not as a
  mechanism.
- **Falsifier**: after the upgrade, the plog must show no `IndexCheck_int64_high_performance_0` trap and
  an out-of-vocab `stop_token_ids` request must return 400 instead of killing the engine.

### Case #15939 — PD + DSpark + prefix caching (open, leads only)

Report: `deepseek-v4-flash-073-w4a8`, vllm-ascend `0.26.rc1`, 1P1D disaggregation,
`--speculative-config '{"num_speculative_tokens": 7, "method": "dspark", "enforce_eager": true}'`,
accuracy distorted, suspiciously low TTOT; disabling prefix caching still failed intermittently; the
default MTP path also failed.

Its `collect_env.py` dump reports vLLM 0.26.0 / vllm-ascend 0.26.0rc1 with `torch_npu==2.10.0.post4`
and CANN 9.1.0, which matches that tag's pins exactly, so the paired upstream tag (`v0.26.0`) is the
right upstream to read. Its attachments are the P and D launch scripts (`run_dp_template_prefill.sh`,
`launch_online_dp_*.py`, proxy script) — the cheapest way to see the real flags.

What the version correction buys: at `v0.26.0rc1` the PD connector package
`vllm_ascend/distributed/kv_transfer/kv_p2p/sfa_pd_rd2h/` exists (`connector.py`, `protocol.py`,
`read_thread.py`), while the DSpark draft-KV readiness helper `sfa_pd_rd2h/dspark_context.py` exists
only at HEAD, so HEAD notes and HEAD-only modules must not be attributed to the reporter's build.

Candidate fixes found by scanning `git log v0.26.0rc1..HEAD` (all dated after the release;
`git merge-base --is-ancestor <c> v0.26.0rc1` → no, i.e. none is in the reporter's build):

- `76c396813` [BugFix][Model] Preserve private cache groups with prefix caching disabled (#17569)
- `6f8c631cc` [BugFix][KV Cache] Propagate resolved prefix retention config (#17480) — later reverted
  by `42ef1c363` (#18029)
- `fa60e23c1` [BugFix] Truncate Mooncake hybrid prefill before prefix-cache lookup (#17407) — a P-side
  scheduler crash, not this accuracy symptom
- `dca1ad85d` [BugFix][PD] Support single-head NHD GQA caches in MooncakeConnectorV2 (#19824),
  `23b76ca20` [BugFix][PD] TP-replicated Mamba state

None was confirmed to fix #15939; treat this as a candidate list to verify against the reporter's
exact config, and say the issue was open when read.

### Case #17593 — DSA model, garbage **only** in graph mode, DP4×TP4 over two nodes (open)

Report: vllm-ascend `0.20.2rc1` (vLLM `0.20.2`, torch_npu `2.10.0`, CANN `9.0.0`,
`transformers 5.5.3`), 910B, customer 30B-bf16 **DSA** (sparse-attention) model, `data_parallel.py`
with `--dp-size 4 --tp-size 4 --node-size 2 --num-speculative-tokens 4 --max-num-batched-tokens 1024
--max-model-len 65536 --compilation-config FULL_DECODE_ONLY`; output is 乱码 in graph mode and fine
with graphs off (the reporter's own control, and the lever to relay).

Maintainers: `0.20.2rc1` is an old historical release for an unvalidated custom model — try the latest
stable first (`kunpengW-code`); then `bowgneo` named **#15362** and **#15409** as the candidate fix. The
reporter patched `model_runner_v1.py`/`test_model_runner_v1.py` per #15362 on `0.20.2rc1` and reported
the same garbage on both nodes, so on this build that fix is **not** the answer — at `v0.20.2rc1` the
dummy-run slot clear the fix relocates does not exist at all, so a hand-port is a different change.

The full treatment — the `_dummy_run` slot-invalidation mechanism, the three lane shas with their first
tags (`fe8f6b0d1`/`v0.27.1rc1`, `04a4549f9`/`v0.23.0.post1`, `23dd2e411`/`v0.26.0rc1`), the DSA builder
sizing at the ref and the answer shape — is `aclgraph-mode-and-capture.md` §D. Do not re-derive #15362
as this issue's fix. If the class does not fit, the hypothesis set is (a) dummy/prefill-batch padding
metadata on a DSA backend, (b) the DSA top-k/indexer path under `UNIFORM_BATCH` capture with 4 spec tokens,
(c) DP4 cross-rank `num_input_tokens` padding; falsifier for each = the same prompt at `temperature=0`,
eager vs graph, then graph with speculation off — a clean eager-vs-graph delta with spec off removes (c)
and leaves the capture path.

### Case #8920 — hybrid (linear-attention/MoE) model under PD: fluent but wrong text

A hybrid model's linear-attention layers carry a separate recurrent state (`MambaSpec`), and at
`v0.18.0` only the layerwise Mooncake connector transfers it; `MooncakeConnectorV1` and the
`kv_pool/ascend_store` connectors do not. So the deciding question for such a report is *which connector
the `--kv-transfer-config` names*, and the trap is that a grep for `gdn`/`linear_attention` returns
nothing at a tag that does support the feature. Full treatment, the per-tag `mamba` counts for
`v0.18.0`/`v0.23.0` and the falsifying checks: `hybrid-gdn-pd.md` (#8920, twins #8916/#8932).

## 3. Collective-reduction faults: FC1/SP warm-up crashes and doubled MoE all-reduce

Two further families belong to the parallelism axis but have their own page, because the evidence is a
shape ratio and a reduction count rather than an accuracy delta: a FlashComm1/SP crash during
cudagraph capture where `src_rows / dst_rows` equals `tensor_parallel_size` (#8990), and a MoE model
with TP>1 that answers fluent nonsense because the routed output is all-reduced twice (#10466, fix
PR #10443 open), plus the deployment whose own shipped config enables FC1 and the reporter's does not
(#10503, mechanism open). Read `flashcomm1-and-moe-reduce.md`; the ledger rows point there.

## Reporting rules for this family

- A maintainer statement is evidence of the workaround, not of the mechanism.
- If the issue is open and no commit or test confirms a cause, deliver the workaround plus the
  suspect set and say the root cause is unresolved.
- Do not present a UT-level check as hardware validation, and do not let a plausible mechanism
  inherit the confidence of a verified one.
