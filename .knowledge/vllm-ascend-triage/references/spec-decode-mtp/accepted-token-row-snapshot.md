# Accepted-token counts gathered from a mutable `InputBatch` row: silent repeated/omitted tokens (#13863)

Verified at vllm-ascend `v0.23.0rc1` (`f4a08bdd`) + vllm `v0.23.0` (`0fc695fc`); mechanism
reporter-confirmed on the issue thread (2026-08-09, the row-ownership worked example).
Recorded #13863 — a **silent** member of the `num_accepted_tokens` family: the siblings below
all leave an error behind, this one leaves only a count.

## Signature

A hybrid linear-attention family (Qwen3.5/Qwen3.6, `model_type: qwen3_5`, Mamba/GDN layers)
serves normally and, on **a small fraction of requests only**, returns output that repeats,
omits or corrupts a fixed run of tokens at an accept boundary. No traceback, no device record,
no OOM, nothing in the log; the numbers in the body are the whole evidence:

| arm (reporter's) | differing outputs |
|---|---|
| `--async-scheduling` + `--speculative-config` (MTP) + `--enable-prefix-caching`, KV at 100 % | 2 / 512 |
| same launch, `--async-scheduling` **off** | 0 / 1024 |
| same launch, accepted-count snapshot fix applied | 0 / 2048 |

Read that table as the classification: async scheduling off is clean, so the lane is the async
runner's cross-iteration bookkeeping and **not** sampling, dtype or W8A8 numerics (a numerics
fault would not be removed by turning async scheduling off). The trigger is **batch churn**, not
batch size: only `condense()` moves rows, so a small steady batch can be clean for hours while a
KV-saturated run with long background requests plus a short foreground request corrupts ~0.4 %.

Distinguish from the three siblings that consume the same number and *do* say something:

- `async-spec-bookkeeping-desync.md` (#8724) — same launch flags, but the process dies:
  `KeyError` in `_update_states` + `'NoneType' … sampled_token_ids` + a negative counter.
  That one is the discarded-request **index map**; this one is the **value in a row**.
- `hybrid-mtp-prefix-cache-unvalidated.md` — request-time `AssertionError`/`IndexError` from the
  align-mode block arithmetic (`get_temporal_copy_spec`). Same consumer, but it *raises*.
- `../custom-ops-runtime-faults/gdn-conv1d-accepted-tokens-varlen.md` (#10372) — host abort
  `numAcceptedTokens[0]=4 exceeds varlen segment length=3`. Same number, printed.
- Not `../accuracy-benchmarks/long-output-repetition.md` unless the MTP + async axis is absent or
  the repetition is temperature/queue-shaped: there is no per-request state-copy step there.

## The defect is row ownership, not the copy's synchronization

Two index spaces meet at this value, and that is the whole bug:

1. iteration N writes the counts **per iteration-N row** — `_update_states_after_model_execute`
   (`vllm/v1/worker/gpu_model_runner.py:1495-1540`) computes
   `num_accepted_tokens.gpu[:num_reqs]`, then either the align-mode fused postprocess
   (`mamba_utils.postprocess_mamba_align_gpu`, `:1520-1529`, which *also* takes
   `input_batch.num_accepted_tokens_cpu_tensor` as an argument) or a non-blocking
   `copy_` into `input_batch.num_accepted_tokens_cpu_tensor[:num_reqs]` (`:1536-1538`),
   each branch then recording the event (`:1534` fused, `:1540` copy);
2. iteration N+1 reads them **through `prev_positions`** — current row → previous row, `-1` for a
   request that had no previous row (`_compute_prev_positions`,
   `vllm/v1/worker/gpu_model_runner.py:1685-1698`, keyed on `prev_req_id_to_index`).

Between the two, at the top of iteration N+1, `_update_states` mutates the same container:
`add_request` writes the default into the row it hands the new request —
`self.num_accepted_tokens_cpu[req_index] = 1` (`vllm/v1/worker/gpu_input_batch.py:466`) — and
`condense()` (`:683`) slides surviving requests down through `_swap_states`, which swaps the
counts between rows (`:659-664`). Both run at `gpu_model_runner.py:1437-1443`, i.e. **before**
`_prepare_inputs` gathers at `:2019-2032`.

A **third** writer sits on the far side of the gather, and only on a hybrid `align` launch: in
`execute_model`, `mamba_utils.preprocess_mamba` resets the row to `1` for every request whose
recurrent state was copied to a new block, and the runner re-syncs that row to the GPU so the mamba
kernel reads `init_token_idx = 0` from the right slot (`gpu_model_runner.py:4171-4184`; Ascend's
copy of the same block at `model_runner_v1.py:2109-2134`, whose comment states the reset). So one
row is written by three different concerns inside a single iteration — the gather's write-back, the
postprocess D2H and this reset — and what iteration N+1 reads can come from any of them: the row is
scratch space, never a history of the previous iteration.

Either way, a previous row index can now belong to a different request, or hold `1`. The reporter's
worked
example, with the values they confirmed:

```text
iteration N      previous_counts[4] = 3 (request B)      previous_counts[11] = 4 (request A)
iteration N+1    current rows: 0→A (prev 11), 1→C (new), 2→B (prev 4)
                 prev_positions = [11, -1, 4]
                 mutable row 11 now reads 1 (a new request took it), row 4 still 3
gather from InputBatch   -> [1, 1, 3]      gather from an iteration-N snapshot -> [4, 1, 3]
```

A wrong count is not a cosmetic error: the value is a **copy offset** into the recurrent state —
`offset = num_accepted_tokens - 1` and
`src_block_id = block_ids[cur_block_idx + num_accepted_tokens - 1]`
(`vllm/model_executor/layers/mamba/mamba_utils.py:304,331`). `1` instead of `4` moves the Mamba/GDN
state copy three tokens, so the next decode step starts from the wrong recurrent state: fixed-length
repetition or omission, intermittently, exactly as reported.

**`event.synchronize()` is not a refutation.** The sync at `model_runner_v1.py:1084` only guarantees
that the device-to-host copy completed; it says nothing about which request owns the row by then.
When a reviewer or a session says "the copy was already made non-blocking and synchronized
upstream (#38556)", that PR fixed the *ordering* of the copy and left the ownership question open —
check it, then re-state the gather's source, which is what decides this.

## Verification: three greps and one arithmetic (three read sites)

No replay is needed; the pin is enough. Pin the ref first (`scripts/sources-summary.sh`), then:

1. `rg -n 'num_accepted_tokens_cpu' vllm/v1/worker/gpu_input_batch.py` → the writers `:466`
   (`add_request` → `1`) and `:659-664` (`_swap_states` under `condense`). Their *existence* is
   half the proof; the other half is that neither is a snapshot restore.
2. `rg -n 'num_accepted_tokens_cpu\[|prev_positions' vllm_ascend/worker/model_runner_v1.py` →
   the gather, `np.where(new_mask, 0, prev_idx)` + `[new_mask] = 1` + write-back at `:1082-1098`
   inside `_prepare_inputs` (`:794`). **The block is duplicated**: upstream
   `gpu_model_runner.py:2019-2032` and the 310P fork
   `vllm_ascend/_310p/model_runner_310p.py:283-294` — three read sites at this pin, so a fix to
   one leaves the other two, and "the same vulnerable read exists in both NPUModelRunner and the
   vLLM core implementation" (the reporter's claim) is a two-tree, three-site statement that these
   two greps verify.
3. `rg -n 'num_accepted_tokens_event|prev_req_id_to_index = ' vllm/v1/worker/gpu_model_runner.py`
   → the writer/recorder (`:1495-1540`) and the map capture (`:3648`, `:4690`). This is the
   ordering fact: writer at iteration end, `add_request`/`condense` at iteration start, gather after.
4. Arithmetic, in `uv run --no-project python`: index the mutable vector with
   `[11, -1, 4]`, replace row 11 with `1`, and print `[1,1,3]` against `[4,1,3]`. That table *is*
   the answer to "why does it repeat tokens" and costs one command.

## The fix shape, and the test that would actually catch it

Take the iteration-N counts **out of the mutable container**: point the D2H copy (and the
align-mode fused postprocess argument, `gpu_model_runner.py:1520-1529`) at a runner-owned pinned
CPU buffer that `InputBatch.add_request()`/`condense()` never touch; in `_prepare_inputs` gather
*that* buffer through `prev_positions`, assign `1` to the `-1` rows, and only then write the
current-row values back into `InputBatch.num_accepted_tokens_cpu[:num_reqs]`. The last two steps
already exist in all three read sites — the missing piece is the gather's **source**, so a patch
that only reorganizes the write-back changes nothing.

**That buffer is a new one, and the name collides with an existing one.** The reporter's wording
("use the model runner's independent CPU/GPU accepted-token buffer as the previous-iteration
snapshot") reads as `self.num_accepted_tokens`, and reusing it is a no-op patch: that buffer
(`gpu_model_runner.py:762`) is this iteration's *destination* — the same block refills it from the
mutable row before `copy_to_gpu()` (`gpu_model_runner.py:2027-2039`; Ascend
`model_runner_v1.py:1089-1104`) — and the step consumes it downstream as the current value
(`model_runner_v1.py:1128` into `update_num_computed_tokens_for_batch_change`, `:3159` into
`GDNAttentionMetadataBuilder`). Write the snapshot where the counts are *produced*
(`_update_states_after_model_execute`, `gpu_model_runner.py:1495-1540`, **not** overridden
downstream — the Ascend runner says so itself at `model_runner_v1.py:2103-2107`). In `align` mode
that helper hands the mutable row tensor straight to the fused postprocess
(`:1524-1528`), so the snapshot there is either a second output argument to
`postprocess_mamba_align_gpu` or its own non-blocking D2H of `self.num_accepted_tokens.gpu` taken
right after the postprocess (at that moment it still holds iteration-N's counts, before the next
iteration's refill) with its own event — the same event pattern as `:1534`/`:1540`.

A regression test must be value-level and must build the reuse: `previous_counts[11] = 4` with a
current batch whose `prev_positions` is `[11, -1, 4]`, after the row-11 index has been re-taken by a
new request, asserting `[4, 1, 3]`. The pin's existing coverage does not do this: the only
accepted-token asserts in `tests/ut/` are **source-text** checks
(`tests/ut/_310p/test_mamba_align_fallback_310p_source.py:32-39`,
`assert "num_accepted_tokens_cpu_tensor[:num_reqs].copy_(…)" in src`), which pass with the bug
present — the same "coverage that cannot fail" pattern as rule 9 of
`../code-change-requests/index.md`. Say so explicitly rather than citing the UT as evidence.

## Not owed in the first message

The report supplies its own root cause, fix and expected behaviour, so the deliverable is a
**verification verdict + fix shape + test shape**, and the confirmation is the two writer/gather greps.
Second-order detail that has consumed recorded budgets without changing the verdict:

- which `mamba_cache_mode` the launch resolves to. `none` copies D2H (`:1536`) and `align` fuses
  it on device (`:1520`), both into the same mutable row, and both feed the same offset
  (`mamba_utils.py:304,331`). It selects *which consumer* is wrong, not whether the value is
  wrong: one grep, then move on — do not walk the auto-default (`models/config.py`) and the
  plugin's copy (`patch_mamba_config.py`) to answer it.
- the fused kernel's internals and `num_accepted_tokens_out` semantics;
  `update_num_computed_tokens_for_batch_change` (`vllm/v1/spec_decode/utils.py:567`) is the
  `num_computed_tokens` sibling with its own clamp and is not this value;
- whether the upstream PR that added the sync landed on the reporter's branch: cite the number as
  the reporter does, label it unverified, and put the gather's source first.

## What would redirect the search

- The async-off arm is **not** clean → this is not the lane; go to
  `../accuracy-benchmarks/long-output-repetition.md` and the sampling rows.
- The pin already carries a runner-owned snapshot for this value (`rg -n 'accepted_tokens.*snapshot|_prev_cpu'`)
  → post-pin fix, and the answer is the version lane plus that grep, not a mechanism.
- Corruption reproduces on a **stable** batch with no request finishing or preempting, i.e. no row
  was ever re-taken → ownership is intact and the wrong count comes from somewhere else: check
  `valid_sampled_token_count` (the `-1` counting branch) and the rejection sampler instead.

## Delivery record

Recorded #13863 spent **59 assistant turns and its entire 16 384-token output budget with no
message text at all** (`output_truncated`): the mechanism was confirmed by turn ~25, then the run
went on to the align-mode auto-default, the fused kernel, the mamba-cache patch, the 310P sibling
and upstream parity, and its reasoning blocks hit 7.7 K characters. Everything this page now
settles with two greps and one arithmetic was re-derived there. Rule 1 and
`../evidence-workspace/delivery-discipline.md` apply unchanged: the verdict ("your read is
confirmed at `f4a08bdd`/`0fc695fc`, at three sites, fix the gather's source, here is the test
shape") was shippable the moment the two writer sites were in hand.

The replay that followed this page is the shape to copy: **nine assistant turns, a 5.5 K-character
answer, `completed`** — locate call, page read, three pin greps, verdict first with two corrections,
then the change plan. Those corrections are what the two greps do *not* show and are what this page
gained: the `self.num_accepted_tokens` name collision (the reporter's wording is a no-op patch unless
a new buffer is created) and the align-mode snapshot variant. The extra deliberation was all on the
align-mode fix shape and stayed a one-grep branch for the reporter, which is where this page files it.
