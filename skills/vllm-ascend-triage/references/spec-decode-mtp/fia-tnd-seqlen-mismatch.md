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

A **second launch reaches the same refusal through a different lane**:
`cudagraph_mode: "FULL"` with `--speculative-config '{"method": "ngram",
"prompt_lookup_max": 2, "prompt_lookup_min": 2, "num_speculative_tokens": 3}'`
(recorded #9088 on vllm-ascend v0.19.1rc1 + vLLM 0.19.1, MiniMax-M2.7-w8a8,
`--tensor-parallel-size 4 --data-parallel-size 2 --enable-expert-parallel
--max-num-seqs 200 --max-model-len 40960`, `--no-enable-prefix-caching`), where the
paste reads `npu_fused_infer_attention_score_out_symint … call
aclnnFusedInferAttentionScoreV3 failed, error code is 561002` and then
`E89999 … When layout is TND, queryT(16) must be equal to the last element of
actualSequenceLengthQ(13)[FUNC:CheckFAISeqlenDataInTND][FILE:
fused_infer_attention_score_tiling.cpp][LINE:904]`. Its own ablation is the
discriminator and belongs in the answer verbatim: **ngram + `--enforce-eager` fine,
FULL without spec decode fine, both together crash** — that shape is a property of the
padding *branch*, not of the kernel (`## Lane B` below).

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

They must be equal. When they are not, exactly one of the padding sites ran. At a
v0.19-or-later ref the pair also *implies the request count* — `r = N / (1+K)` — and
the branch that produced it is decidable without a log (`## Lane B`).

```bash
bash <skill_root>/scripts/fia-tnd-seqlen-check.sh --query-t 16 --seqlen-last 4 \
  --max-num-seqs 16 --num-spec 3 --dp 2
```

One call prints the capture-size list the launch resolves to, whether `N` is one
of those buckets, the request count and accepted-draft count the pair implies under
the v0.19 branch rule (stating the `M == r` ambiguity instead of resolving it), the
Lane A decompositions of `M`, and the ordered ablations.

## Third reading: `queryT` *smaller* than every plausible `M`

The two lanes above both assume `N` is a padded bucket above the real schedule. A paste
can report the opposite direction, and then the padding story is not just uncertain — it
is excluded. Recorded #4771 (vllm-ascend **v0.11.0rc2** `a2e4c3f` + vllm **v0.11.0**
`b8b302c`, Qwen2.5-32B-Instruct TP2 with its own trained **eagle3** draft,
`draft_tensor_parallel_size: 1`, `n=5`, `--max-num-batched-tokens 4096`, no graph-mode
flag on the launch): the first request completes normally (the metrics line even reads
`Drafted: 15, Accepted: 2`), the **second** request dies ~8 s later, and the paste carries
three things at once:

```text
(Worker_TP1) File "vllm_ascend/sample/sampler.py", line 28, in _apply_top_k_top_p
(Worker_TP1) RuntimeError: operator():…/LocalScalarDenseNpu.cpp:47 NPU function error:
  c10_npu::acl::AclrtSynchronizeStreamWithTimeout(copy_stream), error code is 507018
E89999 … When layout is TND, queryT(1) must be equal to the last element of
  actualSequenceLengthQ(33)[FUNC:CheckFAISeqlenDataInTND]…LINE:880]
        When layout is TND, queryT(1) must be equal to the last element of
  actualSequenceLengthQ(2)…                       # a second M in the same record
        Aicpu kernel execute failed, device_id=1, stream_id=2, task_id=7315,
  soName=libcpu_kernels.so, funcName=RunCpuKernel, kernelName=ScatterElements,
  errorCode=0x2a.[FUNC:PrintAicpuErrorInfo][FILE:davinci_kernel_task.cc][LINE:1308]
```

- `sample/sampler.py:28` at that pin is the **fast-path condition** of
  `_apply_top_k_top_p` (`if not is_310p() and p is not None and k is not None and
  1 <= int(k.max()) <= 1024:`). `int(k.max())` is a `LocalScalarDenseNpu` D2H copy, i.e.
  a host sync point: it is where an earlier async op's failure surfaces, not the owner
  (SKILL.md rule 11). Say so in the first paragraph or the reporter will chase the
  sampler.
- The `v0.11`-era padding branch cannot produce this pair. In
  `attention/attention_v1.py` `build()`, the `DecodeOnly and num_input_tokens >
  num_actual_tokens` branch (`:228-247`) pads `seq_lens` and `block_table` by
  `padded_num_tokens` but extends `query_start_loc_cpu` with
  `torch.arange(query_start_loc_cpu[-1] + 1, query_start_loc_cpu[-1] + padded_num_tokens)`
  (`:244-247`) — an **exclusive** stop, so it appends `padded_num_tokens − 1` entries and
  `actual_seq_lengths_q[-1]` (`:271`) ends at `num_input_tokens − 1`. That branch's own
  refusal is therefore **`N = M + 1`**, never `M ≫ N`; it is worth quoting as a
  source-verified off-by-one at this ref (recheck before citing at another), and it is
  enough to retire the padded-bucket reading of a `queryT(1)` paste.
- On the `SpecDecoding`/`ChunkedPrefill` path the two numbers are read from the same
  list: `_forward_v1_style` passes `actual_seq_lengths=attn_metadata.actual_seq_lengths_q`
  (`:553-555`) after the caller unpads `query = query[:query_start_loc[-1]]`
  (`:674-675`), so a *stable* mismatch there needs the metadata itself to be wrong — and
  a paste with **two different `M` in one record** (33 then 2) shows more than one
  refusal was recorded, so the pair does not describe a single step's bookkeeping.

What to do with that shape: treat the **device record as the primary evidence** — the
kernel name, `task_id`, `stream_id`, `chipId` and `retCode=0x2a` belong to the AICPU
task, so the answer is `--enforce-eager` (or `PIECEWISE`) first, `n: 5 → 1` second, and
a request for the plog's `[AIC_INFO] fault kernel_name=` segment plus an
`ASCEND_LAUNCH_BLOCKING=1` run — not a further derivation of which padding site fired.
The recorded run spent ~40 K characters of reasoning on that decomposition and shipped
the mechanism as an open item, which is the correct label: **arithmetic verified, owner
not established.** Do not publish a candidate branch as the cause here; the pair rules
the branches out rather than in.

## Lane A — the padding sites that disagree at v0.14 refs (verified 52d4acf, MTP launches)

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

## Lane B — from v0.19 the sites collapse into one helper whose guard is a coincidence test

Source-verified at vllm-ascend **v0.19.1rc1** (`da421af`, #9088) with vllm **v0.19.1**
(`b1388b1`), i.e. *after* the refactor the helper's own `TODO` names (upstream PR #34043
deleted `relax_for_mixed_batch_cudagraphs`). Everything under Lane A is v0.14-era code:
**do not quote `:856-881` or `mla_v1.py:565-605` at a 0.19 ref.** The gate moved into
`_pad_query_start_loc_for_fia` in *both* runners
(`worker/model_runner_v1.py:566-604`, `worker/v2/model_runner.py:395-431`), a third copy
exists for 310P (`_310p/model_runner_310p.py:140-160`), and the docstring states this
report's invariant verbatim — "when the layout is TND, the first dimension of
`hidden_states` must equal the last element of `actual_seq_lengths_q`".

| v0.19 site | what it decides |
|---|---|
| `model_runner_v1.py:580-582` (`if cudagraph_runtime_mode == FULL and self.compilation_config.cudagraph_mode == FULL: num_reqs_padded = num_reqs`) | under FULL the *padded* request count is forced back to the real one, so the guard below can never see the bucket as extra requests. The v2 twin (`worker/v2/model_runner.py:410-413`) drops the second condition. |
| `:586` (`if num_tokens_padded == num_reqs_padded * self.uniform_decode_query_len:`) | **the guard is a coincidence product test**: it compares the *padded bucket* with `num_reqs × (1+K)` and never reads the real token count, so it is true whenever the bucket happens to be an exact multiple of `1+K` at that request count — including for a batch whose drafts came out short. |
| `:587-593` uniform branch | fills only indices `num_reqs+1 … num_reqs_padded`; FULL already set `num_reqs_padded == num_reqs`, so that slice is empty and `query_start_loc[num_reqs]` keeps the **real** cumsum (`M`) while the rows are the bucket (`N`). |
| `:594-600` mixed branch | appends a dummy request and writes `query_start_loc[num_reqs+1] = num_tokens_padded`, so the last element **is** `N`. |

That last row is load-bearing: **under FULL the mixed branch cannot produce a refusal.**
So on a 0.19 ref the pair localises itself, no source walk needed:

```text
N == r × (1+K)                 # else the mixed branch ran and M would equal N
r      = N / (1+K)             # integer, 1 <= r <= max_num_seqs
M - r  = total accepted drafts # 0 <= M - r <= K × r, else the pair is not this lane
```

`16` vs `13` at `K = 3` gives `r = 4` requests and `13 - 4 = 9` accepted drafts, i.e.
drafts `3+3+3+0`. The step's `Running:` count is the confirming line; without it, state
`r = 4` as a prediction, not a reading (the numbers alone do not exclude a request that
scheduled 1 token because it reached `max_model_len`).

**Why `ngram` and not MTP trips it.** MTP pads every decode request to exactly `K`
drafts, so the real total *is* `num_reqs × (1+K)`: the guard is true and also correct.
`ngram` proposes 0…K drafts per request
(`vllm/v1/spec_decode/ngram_proposer.py:170-195`, `valid_ngram_num_drafts[i]`), and a
request at `max_model_len` is skipped and schedules one token
(`vllm_ascend/spec_decode/ngram_proposer.py:47-49`). Either way the real total falls
below `r × (1+K)` while the bucket still equals it. The tree already knows this on 310P:
`_310p/model_runner_310p.py:78-80` forces `uniform_decode_query_len = 1` for
`method == "ngram"` ("310P ngram requires decode-only graph shapes to be built with
q_len=1"; `:46` is the constant, `:83-97` the scoped override) and its
`_determine_batch_execution_and_padding` (`:99-122`, test at `:117-121`) uses the
**correct** uniformity test — `max_num_scheduled_tokens == decode_query_len and
num_tokens == decode_query_len * num_reqs` — which is exactly the condition the shared
guard is missing. Non-310P runners carry no such workaround.

Which number is which, on the v1 path (the default: `VLLM_USE_V2_MODEL_RUNNER` is
`False` at `vllm/envs.py:239`, so a paste with `npu_fused_infer_attention_score` /
`aclnnFusedInferAttentionScoreV3` is `attention_v1.py`, backend `CUSTOM` at `:81`):

- `N` = rows = `num_tokens_padded = batch_desc.num_tokens`
  (`worker/model_runner_v1.py:1574`, passed to the helper at `:1628-1630`): the graph
  bucket.
- `M` = `actual_seq_lengths_q[-1]` = `query_start_loc_cpu[1:].tolist()`
  (`attention/attention_v1.py:322`) over
  `query_start_loc_cpu=self.query_start_loc.cpu[: num_reqs_padded + 1]`
  (`worker/model_runner_v1.py:2473`) — truncated at the *un*-extended `num_reqs`.
- `attention_v1.py:591` reads `attn_metadata.actual_seq_lengths_q[-1]` only as the graph
  workspace key (`graph_params.workspaces.get(num_tokens)`); it does not slice `query`.
  The divergence therefore reaches the op as reported, and a wrong workspace bucket is
  a second, silent symptom if the op ever accepts the pair.

Capture sizes at this ref: defaults are `[1,2,4] + 8..248 step 8 + 256..` with
`max = min(max_num_seqs × (1+K) × 2, 512)` clipped by `max_num_batched_tokens`
(`vllm/config/vllm.py:1343-1435`); an **explicit** `cudagraph_capture_sizes` list is
rounded up to multiples of `1+K` (`vllm/config/compilation.py:1237-1283`). Pinning the
list therefore does not remove the coincidence, it only changes which multiples exist —
say that before passing rung 3 on.

Fix statement for the maintainer (both call sites, `model_runner_v1.py:586` and
`worker/v2/model_runner.py:415`): test the **real** count
(`num_tokens == num_reqs × (1+K)`, as `_310p` does) instead of the padded one, or take
the dummy-request branch whenever `query_start_loc[num_reqs] != num_tokens_padded`. The
`max_num_reqs + 2` allocation that branch needs is already reserved — "For FULL mode we
change +1 to +2 to reserve extra space for padding" (`worker/v2/model_runner.py:239-242`)
— so the fix is the branch condition only.

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
   On a **Lane B** ref the sharper A/B is the *method*: an `ngram` → `mtp` switch
   (fixed `K` drafts per request) keeps FULL on and still survives ⇒ the variable
   draft length is the owner, which names the guard rather than the padding.
3. Pin `"cudagraph_capture_sizes": [1,2,4,8,16,32]` (tutorial value) so the bucket
   the step pads to is the one the metadata was captured with. On a Lane B ref this
   only moves which buckets exist (explicit lists are rounded to multiples of `1+K`,
   `config/compilation.py:1237-1283`) — it does not remove the coincidence, so say
   so rather than selling it as a fix.
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
- Classification: for the `queryT > M` pair lane C is wrong — the fault is host-side
  metadata bookkeeping, detected by the op before launch, and the reported line has no
  device dump, no `aicore`, no plog content to read. When the pair runs the **other**
  way (`queryT` below every `M`) the paste *does* carry a device record, so say so
  instead: host sync point + AICPU task, plog ask first (§*Third reading*).
- Version scope: the third reading is source-verified at vllm-ascend v0.11.0rc2
  (`a2e4c3f`) with vllm v0.11.0 (`b8b302c`) — `sample/sampler.py:28`,
  `attention_v1.py:228-247`, `:271`, `:553-555`, `:674-675`; Lane A is source-verified at vllm-ascend v0.14.0rc1 (`52d4acf`,
  snapshot; reporter ran `d1dcdfc`) with vllm v0.14.1 (`d7de043`); Lane B at
  v0.19.1rc1 (`da421af`) with vllm v0.19.1 (`b1388b1`). The recorded v0.14 paste was
  a 3-line OCR with **no Python frame**, so which padding site fired there is derived
  from the two numbers plus the launch shape — label it hypothesis-plus-check, and
  re-read `:588-591`, `:606`, `:856-881` and `mla_v1.py:565-605` before quoting them
  at another release (that padded-drafter machinery is ~v0.13-v0.14 code). At a 0.19+
  ref do the opposite: `rg -n '_pad_query_start_loc_for_fia' vllm_ascend/` first and
  quote the helper — the two numbers alone then decide the branch.
- Ask for: the full per-rank log tail with the `Running/Waiting` line of the
  failing step, `--compilation-config` and `--speculative-config` verbatim, the
  resolved capture-size list, how many requests were in flight, and whether the
  first request or only concurrency triggers it. On a Lane B ref, name the number
  first — "`Running: 4` (`N/(1+K)`) would confirm it, `Running: 13` would refute it"
  — so the reply cannot be satisfied by a vague log tail.
