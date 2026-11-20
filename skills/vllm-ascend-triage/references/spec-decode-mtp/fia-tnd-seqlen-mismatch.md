# FIA TND seqlen mismatch: `queryT(N)` vs `last(actualSequenceLengthQ)(M)`

## Report shape

The workers are up and the first decode step dies with an **op-level message and
no Python frame** (often the whole paste is a screenshot; read the OCR, then the
two numbers in it):

```text
ERR00100 PTA call acl api failed
EZ9999: Inner Error!
EZ9999 [PID: 69963] ... (EZ9999): When layout is TND, queryT(16) must be equal to
  the last element of actualSequenceLengthQ(4)
  [FUNC: CheckFASeqlenDataInTND] [FILE: fused_infer_attention_score.cpp] [LINE:904]
```

Launch that produces it: `--compilation-config '{"cudagraph_mode":
"FULL_DECODE_ONLY"}'` together with `--speculative-config '{"method": "mtp",
"num_speculative_tokens": <K>}'` (recorded with `K = 3`), an MLA-family model (GLM-4.x
`Glm4MoeForCausalLM`, DeepSeek-V3.x), often `--data-parallel-size > 1` and
`--enable-expert-parallel`. Nothing crashes at capture; the refusal arrives with
real requests, sometimes on the first one.

The constraint is old and stated in the tree's own RFC #3003 (*Support MTP Graph
Mode*, `history/<case>/3003.md:11-18`): TND handles more than one token per
request in decode, **each request can handle fewer than 16 tokens** in TND layout
(that bound is the `FIA_SEQ_LEN_LIMIT = 16` constant the padding decision uses at
`attention/mla_v1.py:346`), and
`npu_fused_infer_attention_score` requires `actual_seq_lengths[-1] ==
batch_size(num_tokens)`; in Torchair/graph mode the lengths must be padded to the
same values every iteration. #4300 (*MTP in full_decode_only graph mode*) is the
RFC that built the padding machinery this report trips over, and says so plainly:
"MTP>1 … in piecewise scenarios … the capability for MTP to operate in
full_decode_only graph mode remains unsupported" before that change.

## Read the two numbers first — the arithmetic is the classification

- `queryT(N)` = rows of the query tensor actually handed to the op = the **graph
  bucket** the step runs at, i.e. `num_input_tokens` *after* the local
  `pad_for_cudagraph` and *after* the DP-wide maximum. It is not the real
  schedule.
- `last(actualSequenceLengthQ)(M)` = the last entry of the per-request cumulative
  query lengths the metadata builder produced, i.e. the **real** token count of
  that rank's batch (or, for the drafter, its request count — see below).

They must be equal. When they are not, exactly one of the padding sites ran.

```bash
bash <skill_root>/scripts/fia-tnd-seqlen-check.sh --query-t 16 --seqlen-last 4 \
  --max-num-seqs 16 --num-spec 3 --dp 2
```

One call prints the capture-size list the launch resolves to, whether `N` is one
of those buckets, every decomposition of `M` that fits the launch, and the
ordered ablations.

## The padding sites that disagree (verified at v0.14.0rc1, 52d4acf)

| site | pads the query rows (`N`) | pads `actual_seq_lengths_q` (`M`) |
|---|---|---|
| `worker/model_runner_v1.py:588-591` `vllm_config.pad_for_cudagraph(total_num_scheduled_tokens)` | any batch whose tokens fit under the largest capture size | — |
| `:606` → `_sync_metadata_across_dp` (`:439-487`, `max_tokens_across_dp` at `:480`) | **DP>1**: `num_input_tokens` becomes the maximum over DP ranks, i.e. the peer's workload, not a local bucket | — |
| `:856-881` (`decode_mode() == FULL and uniform_decode`) | (bucket already chosen) | **only here** for the target model: fills `query_start_loc`/`seq_lens` up to `num_reqs_padded = num_input_tokens // uniform_decode_query_len`, then `num_reqs = num_reqs_padded` |
| `attention/mla_v1.py:565-605` (`self.graph_pad_size > num_reqs`) | (drafter only) | `pad_actual_seq_len_q_mtp_disable_pad` / `…_enable_pad` (`:325-368`, `FIA_SEQ_LEN_LIMIT = 16`) — and only when `graph_pad_size` is set, which happens **only** in `spec_decode/mtp_proposer.py:284` (`graph_pad_size = num_input_tokens` under FULL, else `-1`) |
| `spec_decode/mtp_proposer.py:434-448` | — | rewrites the drafter's list to `arange(1..batch_size)` = **one token per request**; skipped when `disable_padded_drafter_batch` is False (the default) and the mode is FULL, because the builder is then supposed to have padded |

So the invariant holds exactly for a **uniform decode batch on a single DP rank**
(`uniform_decode_query_len <= num_input_tokens <= max_decode_tokens`). It breaks
when the step was padded in rows and one of these held instead:

1. **The DP-wide max raised `num_input_tokens`** above the local schedule (DP>1
   with uneven ranks, which is the normal case when two DP engines serve
   different prompts — and the shape of the recorded report, `DP2 × TP8`). Then
   `N` is a peer's token count, not even a local capture size.
2. **The batch was not uniform decode** (mixed decode+prefill in one step,
   `--async-scheduling` bookkeeping, or a request whose next-token budget allows
   fewer than `1+K` tokens), so the `:856` gate is False while `:590` already
   padded.
3. **The MTP drafter** ran with the default padded drafter batch: its list is
   `arange(1..batch_size)` (1 token per request) unless `graph_pad_size >
   num_reqs` and the `mtp_enable_pad` branch replaced it — and that branch's
   `need_padding` needs `static[num_reqs] - actual[-1] > 16` (`mla_v1.py:346-368`),
   whose else-branch appends a slice of the **static** `actual_seq_lengths_q`
   list at index `num_reqs`, so it only lands on `N` by coincidence.

The numbers usually separate (1)/(2) from (3): in (3) `M` equals the number of
requests in the draft batch (1 token each), while in (1)/(2) `M` equals the real
scheduled tokens, i.e. `requests × (1+K)` for pure decodes. When both readings fit
the two numbers — `16` vs `4` fits `4 requests × 4 tokens` **and** a padded
bucket around `4` one-token requests — say so and ask for the `Running/Waiting`
line of the failing step instead of picking one by preference.

Related but distinct: `../mixed-batch-splits/index.md` is the same
requests-vs-tokens confusion where nothing checks it, so it reads out of the
tensors' own storage and surfaces as `507015`/`aclnnX failed`; here the operator
checks the pair and refuses before launching.

## The tutorial diff that is the first half of the answer

The tree's own GLM-4.x page at this ref (`docs/source/tutorials/GLM4.x.md:89-104`)
runs the same feature set, and differs from the recorded report in every row that
matters:

| knob | tutorial at v0.14.0rc1 | recorded #6598 report |
|---|---|---|
| `num_speculative_tokens` | `1` | `3` |
| capture sizes | `"cudagraph_capture_sizes": [1,2,4,8,16,32]` pinned | default (upstream derives `[1,2,4] + 8..min(128, …)` step 8 for `max_num_seqs=16`, `K=3`) |
| parallelism | `--data-parallel-size 1 --tensor-parallel-size 16`, single node | `DP2 × TP8` across two nodes |
| expert parallel | off ("turn off expert parallel in low-latency scenarios") | `--enable-expert-parallel` |
| `--async-scheduling`, `--max-num-batched-tokens 16384`, `--max-num-seqs 16` | present | present |
| weights | `glm4.5_w8a8_with_float_mtp` (msmodelslim W8A8 with float MTP) | BF16 GLM-4.7 dir + `--quantization ascend` |

and the same page carries `Multi-node Deployment: Not recommended to deploy
multi-node on Atlas 800 A2 (64G * 8)` (`:112`) — while validating a different
weight set, so its command is a diff target, not something to paste verbatim. A2 two-node + `K=3` + FULL
decode graph is outside every validated row of that page, which is why the ladder
starts there rather than in the kernel.

## Ladder (cheapest first)

1. `--compilation-config '{"cudagraph_mode": "PIECEWISE"}'` (or `--enforce-eager`):
   removes the FULL bucket entirely, so `N` and `M` are the same number again.
   One run confirms the padding bookkeeping and unblocks the reporter; the cost is
   decode throughput, not correctness.
2. `num_speculative_tokens: 3 → 1` (the tutorial value). Survives ⇒ the
   padded-drafter/FULL path with `K > 1` is the owner (RFC #4300 territory).
3. Pin `"cudagraph_capture_sizes": [1,2,4,8,16,32]` (tutorial value) so the bucket
   the step pads to is the one the metadata was captured with.
4. Drop DP to 1 (`--data-parallel-size 1 --tensor-parallel-size 16`, one node) —
   separates the DP-wide max from the local batch. Or make both DP ranks carry
   the same in-flight count, which is not something a reporter can do.
5. `--enable-expert-parallel` off (tutorial has it off; `DP2 × TP8 = 16` is also
   the MC2 boundary, so this may change the MoE path as well).
6. `--async-scheduling` off **last**: the tutorial keeps it on, so it is not a
   first suspect, but it changes what the scheduler knows about accepted tokens
   and therefore whether a decode step is uniform.

## Answer shape and scope

- First sentence: not a card, CANN or collective fault — the attention operator
  refused two numbers that must be identical: the query rows were padded to a
  graph bucket (`queryT`) while the per-request cumulative lengths stayed at the
  real schedule; then rung 1.
- Classification: lane C is wrong here — the fault is host-side metadata
  bookkeeping, detected by the op before launch; the reported line has no device
  dump, no `aicore`, no plog content to read.
- Version scope: source-verified at vllm-ascend v0.14.0rc1 (`52d4acf`, snapshot;
  reporter ran `d1dcdfc`) with vllm v0.14.1 (`d7de043`) in the same snapshot. The
  recorded paste was a 3-line OCR with **no Python frame**, so which padding site
  fired is derived from the two numbers plus the launch shape, not from a stack —
  label it hypothesis-plus-check, and re-read `:588-591`, `:606`, `:856-881` and
  `mla_v1.py:565-605` before quoting them at another release (the whole
  padded-drafter machinery is ~v0.13-v0.14 code).
- Ask for: the full per-rank log tail with the `Running/Waiting` line of the
  failing step, `--compilation-config` and `--speculative-config` verbatim, the
  resolved capture-size list, how many requests were in flight, and whether the
  first request or only concurrency triggers it.
