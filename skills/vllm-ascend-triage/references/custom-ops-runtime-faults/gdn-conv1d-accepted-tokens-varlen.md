# `numAcceptedTokens[i]=N exceeds varlen segment length=M`: the GDN conv1d op's own accounting check

## Scope

A hybrid GDN model (Qwen3.5 / Qwen3.6-A3B, Qwen3-Next) served with MTP and
`cudagraph_mode: FULL_DECODE_ONLY`, which after hours-to-days of healthy serving
**aborts the service** with one operator sentence and no Python frame:

```text
numAcceptedTokens[0]=4 exceeds varlen segment length=3
```

The text is a host-side tiling validation in the AscendC `causal_conv1d` op — note the
directory, it is **not** under `attention/`:
`csrc/moe/causal_conv1d/op_host/causal_conv1d_tiling_validation.h:358-364` (verified at
vllm-ascend **v0.20.2rc1** `367b8e62`; in that snapshot it is
`version_role: dated_reference_only` with `reported_ref_status: conflicting_versions`, so
treat every line number as the dated reference and re-locate by string at another ref).
The op never launched: `OP_LOGE` + `ge::GRAPH_FAILED` on the host, so there is no device
dump, no `fault kernel_name=`, no plog entry, and no card/driver lane. Reconciling this
pair is a *bookkeeping* job.

Not this page: a device fault (`index.md` rules), the FIA `queryT`/`actualSequenceLengthQ`
refusal (`../spec-decode-mtp/fia-tnd-seqlen-mismatch.md`), `queryStartLoc[last] must equal
cuSeqlen` from the same validator (the padding/`x.shape[0]` half — §3 here names the site),
or the upstream `IndexError` in `get_temporal_copy_spec`
(`../spec-decode-mtp/hybrid-mtp-prefix-cache-unvalidated.md`).

## 1. Read the validator: what the two numbers are

The same file also prints `exceeds seqlen=… in 3D update` and `exceeds decode 2D token
count (1)`; which one fires is decided by `inputMode` (`:93-118`), and **this message is
`inputMode == 0`**, i.e. the 2D varlen form with `queryStartLoc` present:

| Validator fact (v0.20.2rc1) | Line | Consequence |
|---|---|---|
| `inputMode = 0` needs 2D `x` in non-decode shape, or 2D `x` + `queryStartLoc` in decode mode with `batchFromQsl != batch` (`:212-224`) | `:105`, `:212-224` | `batch = qslSize - 1` (`:225-227`) — one **segment per request** |
| `queryStartLoc[0] == 0`, `queryStartLoc[last] == cuSeqlen`, non-decreasing | `:186-208` | `cuSeqlen` is `x.shape[0]`, so the segments *are* the scheduled tokens of this step |
| `numAcceptedTokens.size must equal batch` | `:326-330` | `nat[i]` and segment `i` are the **same request** — a length mismatch is caught before the value check |
| `0 <= a <= qslData[i+1]-qslData[i]` | `:343-365` | the contract at issue: the accepted count for a request can never exceed that request's token segment |
| `numAcceptedTokens is only supported in runMode=1 (decode/update)` | `:321-323` | only the decode/update launch carries the argument at all |

So `4 > 3` is not a tolerance problem: with `num_speculative_tokens: 3` the accepted
count can legitimately reach `1+K = 4`, while the segment (`1 + drafts_scheduled`) came
out 3. Somebody's `4` and somebody's `3` describe different steps.

## 2. The two feeders, and why they live in different worlds (v0.20.2rc1)

```bash
rg -n "npu_causal_conv1d_custom|spec_nat_host" vllm_ascend/ops/gdn.py
rg -n "_build_spec_causal_conv1d_host_meta|_build_spec_query_start_loc_cpu" vllm_ascend/patch/worker/patch_gdn_attn.py
```

| Half of the pair | Where it comes from |
|---|---|
| segment `M` | `_build_spec_query_start_loc_cpu` (`patch_gdn_attn.py:454-480`) takes `common_attn_metadata.query_start_loc_cpu`, keeps the rows the mask `num_decode_draft_tokens_cpu >= 0` selects, and cumsums them. That is the **scheduler's** `num_scheduled_tokens` for this step. |
| accepted `N` | `_build_spec_causal_conv1d_host_meta` (`:651-682`) copies `attn_metadata.num_accepted_tokens[:num_spec_decodes]` through `_copy_to_pinned_cpu` (`:565-586`) into a 2-slot pinned pool (`:519-545`, `:557-563`). `attn_metadata` got it from `model_runner_v1.py:2912` `num_accepted_tokens=self.num_accepted_tokens.gpu[:num_reqs_padded]`, and that GPU buffer is filled **from CPU accounting** at `model_runner_v1.py:926-952` — `input_batch.num_accepted_tokens_cpu`, remapped through `prev_positions` in async mode, tail `fill(1)` — from `_update_states_after_model_execute`. |

The runner's own comment at `:953-960` says the quiet part: *in async spec decode the CPU
values are optimistic (all drafts accepted) and the kernel corrects on GPU using the
previous step's `valid_sampled_token_count_gpu`* — and that device-side correction
(`update_num_computed_tokens_for_batch_change`) updates `num_computed_tokens`, **not** the
host tuple this op is validated with. `nat` is therefore "1 + however many drafts the
*previous* step could have accepted"; `seg` is "this step's scheduled tokens". They agree
only while every spec request is scheduled with a full `1+K` segment.

Then the launch carries the host tuples as ACL-graph arguments
(`vllm_ascend/ops/gdn.py:393-460`, `spec_q_per_seq = spec_state_indices_tensor.size(-1)`
at `:408`), and replay goes through `_pad_conv1d_host_args_to_capture` (`gdn.py:89-130`),
which pads *dummy* requests at `q_per_seq` stride with `nat = 1` until
`qsl[last] == cap_x_dim0`; its docstring states the reason (tiling insists
`qsl[last] == cuSeqlen`, i.e. capture-time `x.shape[0]`). A runtime step whose real
segments are shorter than `q_per_seq` leaves `pad_tokens % q_per_seq != 0`, which this
helper silently truncates.

## 3. Ranked candidates, each with its one check

1. **The segment was *capped* below `1+K` while `nat` stayed optimistic.** The arithmetic
   has one trap: the `num_output_placeholders` term **cancels**.
   `vllm/v1/core/sched/scheduler.py:408-412` sets
   `num_new_tokens = num_tokens_with_spec + num_output_placeholders - num_computed_tokens`,
   and `:526-530` computes
   `num_scheduled_spec_tokens = num_new_tokens + num_computed_tokens - num_tokens -
   num_output_placeholders`; substituting gives `num_tokens_with_spec - num_tokens` =
   `len(spec_token_ids)` (`vllm/v1/request.py:232-233`), so `--async-scheduling` on its own
   does **not** shorten a segment — the placeholder is there *because* `num_tokens_with_spec`
   already ran ahead. (Algebra verified at vllm v0.20.2 `bc150f50`; at an older ref the
   same two lines sit elsewhere — v0.17.0's clip is `:496-510` — so re-do it before
   repeating the cancellation claim.) What shortens it are the three caps applied between those lines:
   `long_prefill_token_threshold` (`:413-414`), `token_budget` (`:415`, chunked prefill
   pressure), and `max_model_len - 1 - num_computed_tokens` (`:418-421`) — a request one or
   two tokens short of `--max-model-len 133000` schedules a 2-3-token segment while the
   op is handed that request's *previous* acceptance, up to `1+K = 4`: exactly `4 vs 3`.
   **One check:** the failing step's `num_computed_tokens` beside `--max-model-len`
   (`npu-smi`-free, from a plog step record or the reporter's own long-context prompt), or
   reproduce by lowering `--max-model-len` so the cap is hit in minutes instead of days.
   `--async-scheduling` off stays rung 1 for the *other* half — it is what makes `nat` a
   CPU-side optimistic value (`model_runner_v1.py:930-952`) rather than a sampled one.
2. **The capture-time padding world.** `_pad_conv1d_host_args_to_capture` assumes each
   runtime spec segment is `q_per_seq = 1+K`; when it is not, the dummy rows it appends no
   longer reconstruct the captured `cuSeqlen`, so the same launch can be self-consistent in
   `qsl[last]` and inconsistent per segment. `--enforce-eager` takes the eager branch
   (`gdn.py:447-476`) and removes capture/replay entirely — expensive (the recorded cost
   elsewhere in this library for spec-scoped eager is several times slower), but it is the
   only clean discriminator for this row. The reporter's launch has
   `"enforce_eager": true` *inside* `--speculative-config`: that is the **drafter's**
   graph mode and does not remove the target model's captured conv1d.
3. **`nat` is not merely mis-paired — it is wrong.** The public follow-up on #10372 says
   the fed value is itself wrong and quotes `spec_nat_host=(3, 6, 9)` for a `K=3` launch.
   Two readings, and they are cheap to separate: `(3,6,9)` is exactly `qsl[1:]` for three
   3-token segments, so the print may be showing the *qsl* tuple rather than `nat`; and if
   those really are `nat` values, 6 and 9 are above `1+K = 4`, which no per-request
   acceptance count can be — the owner is then the CPU copy/`prev_positions` remap
   (`model_runner_v1.py:930-952`) or the pinned-slot pool, not the scheduler.
   **One check:** ask for **both** tuples from the same failing step (`spec_qsl_host` and
   `spec_nat_host`, `gdn.py:393`) plus the plog segment around the abort — `ASCEND_PROCESS_LOG_PATH`'s
   `plog/`, and the reporter's `#10372` thread asks for exactly that ("keep the plog during
   issue time").
4. **A stale tail rather than a wrong row.** `num_accepted_tokens.np[num_reqs:].fill(1)`
   (`:948`) and the `:670-673` builder copy make padding rows harmless *by design*; only
   check this after 3 is refuted, and check it by comparing `len(spec_nat_host)` with
   `num_spec_decodes` in the same record.

Do not answer this signature with "the kernel is wrong" or "HBM": nothing was launched,
and no DCMI/device line is involved. The deliverable is which of the two feeders moved.

## 4. What a repair must satisfy

A clamp (`min(nat[i], segment)`) in the host-meta builder removes the abort and hides a
real bookkeeping error — this library's rule 3 applies in full. The structural repair is
that the count and the segment must come from the same step: either derive `nat` from the
step's sampled result the way `update_num_computed_tokens_for_batch_change` already does
on device, or (if `nat` must stay the *previous* step's acceptance for the state update)
validate it against the segment it actually belongs to, and make the capture-padding
helper reconstruct `cuSeqlen` from real segment lengths. Acceptance check: a long-context
run on the hybrid GDN family with `K=3` + `--async-scheduling` for longer than the
reporter's interval, plus one `K=1` run, with the per-step tuples logged.

## Recorded run (#10372)

Snapshot pin vllm-ascend v0.20.2rc1 (`367b8e62`) / vllm v0.20.2 (`bc150f50`); reporter's launch
`Qwen3.6-35B-A3B`, TP4 + EP, `--max-model-len 133000`, `--max-num-batched-tokens 16384`,
`--max-num-seqs 128`, `FULL_DECODE_ONLY`, `method: qwen3_5_mtp` `K=3`,
`--async-scheduling`, crash "after running for several days". The session found the owning
line in three greps, then spent the rest of its budget deciding *which producer* moved —
it opened the upstream scheduler truncation (`:526-535`) after its answer text was already
typed, re-ranked async first, and was cut by the 16 384-token cap with no revised message;
the follow-up's `(3,6,9)` datum was never asked for, and the arithmetic that decides the
ranking (§3 item 1: the placeholder term cancels, only the caps shorten a segment) was
never completed. The two feeders in §2 plus one printed
pair is the whole answer (`SKILL.md` rule 1, *a refusal that quotes an argument value*).
