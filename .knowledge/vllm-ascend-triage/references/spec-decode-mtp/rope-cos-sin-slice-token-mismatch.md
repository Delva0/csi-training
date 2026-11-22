# MTP drafter rope: `cos.view(num_tokens, -1)` — the cos/sin slice still sized for the *target's* padded step

## Report shape

An MTP / spec-decode serve dies at request time in the host-side reshape of the Triton
rope helper. There is no device frame, no `ERR`/`EZ` code and no plog line, because
nothing was launched:

```text
File ".../vllm_ascend/spec_decode/mtp_proposer.py", line 320, in _propose
    hidden_states = self.model(input_ids=input_ids,
...
File ".../vllm_ascend/ops/rotary_embedding.py", line 226, in _rope_forward_oot
    query, key = torch.ops.vllm.rope_forward_triton(q, k, cos, sin, ...
File ".../vllm_ascend/ops/triton/rope.py", line 155, in rope_forward_triton
    cos = cos.view(num_tokens, -1)
RuntimeError: shape '[6, -1]' is invalid for input of size 512
```

Recognition is the **pair** `cos = cos.view(num_tokens, -1)` in `ops/triton/rope.py`
and `_rope_forward_oot` in `ops/rotary_embedding.py`; grep both strings, never the
paste's line numbers (they are one or two lines off from the tag on any other ref).
Every frame above the reshape is model code (`glm4_moe_mtp.py` → `glm4_moe.py`
`self.attn` → `rotary_emb.forward_oot`) and proves only that a *drafter or target*
forward reached rope — the model, the checkpoint and `partial_rotary_factor` are not
the owner. `num_tokens` is `q.shape[0]`; the `size` is `cos.numel()`.

**The paste's `(Worker_TP<n> pid=…)` prefix is a *rank index*, not a size**, so it
never turns a matching case into "a different TP". `Worker_TP10` on a 16-chip box is
TP16's rank 10, and a rank number that reaches the world size at all proves the
deployment is at least that large (a TP10 launch has no rank 10 to print). Read sizes
from the reporter's own command or the group lines, not from the prefix — the segment
is `_TP{tp_rank}` with `tp_rank = get_tp_group().rank_in_group`
(`vllm/v1/executor/multiproc_executor.py:843-857` at vllm v0.14.1, `:1016-1030` at
v0.23.0; `../hccl-and-hardware/error-report-timestamps.md:179`); the recorded #6449
triage shipped "re-filed at TP10 instead of TP16" for a paste whose command says
`--tensor-parallel-size 16`.

## One division names the two counts

`cos` is the **process-global** slice returned by `get_cos_and_sin_slice()` — not a
per-call tensor — so the paste is a disagreement between the token count of *this*
forward and the token count of whichever forward last called `update_cos_sin`:

- `rows = numel / width`, where `width` is `_cos`'s last dim (`set_cos_and_sin`,
  `ops/rotary_embedding.py:87-97` at v0.14.0rc1: `head_size`, times
  `hf_text_config.partial_rotary_factor` when that attribute exists);
- `rows == num_tokens` → the slice matches and this page is not the lane;
- `rows != num_tokens` and `numel % num_tokens != 0` → the pasted `RuntimeError`;
- `rows != num_tokens` but `numel % num_tokens == 0` → **no exception, wrong rope**:
  the view succeeds with a row width the kernel then strides over (see the table
  below, `1` request). An accuracy complaint with this stack can be the same defect.

`bash <skill_root>/scripts/rope-cos-sin-slice-check.sh --num-tokens 6 --size 512
--capture-sizes 1,4,8,16 --num-spec 1 --max-num-seqs 16` prints the same division,
the pad map the launch resolves to, and the request counts that raise instead.

## Who writes the count — one bounded grep, not a model-constant hunt

The slice is refreshed only where `update_cos_sin(positions)` is called, and every
call passes *padded* positions:

```bash
rg -n 'update_cos_sin' vllm_ascend/        # the whole package, one call
```

At **vllm-ascend v0.14.0rc1** (`52d4acf`) that returns five sites and one telling
absence:

| Site | Count it writes |
|---|---|
| `worker/model_runner_v1.py:1375` (execute_model) and `:2067` (`_dummy_run`) | the **target's** `num_input_tokens`: `_prepare_inputs` sets it to `pad_for_cudagraph(total_num_scheduled_tokens)` while `use_aclgraph` and the total fits `cudagraph_batch_sizes[-1]` (`:587-591`), and `_preprocess` slices `self.positions.gpu[:num_input_tokens]` (vllm v0.14.1 `v1/worker/gpu_model_runner.py:2697`) — i.e. the **padded** length |
| `spec_decode/eagle_proposer.py:326` (`dummy_run`), `:506` (`_propose`), `:670` (`_run_merged_draft`) | the EAGLE drafter refreshes the global itself, with its own padded `num_input_tokens` |
| **`spec_decode/mtp_proposer.py` — no hit at all** | the MTP `_propose` (`:131`) computes `num_input_tokens` at `:245-249` and then calls `self.model(...)` at `:319-321` **without** refreshing the global |

So on an `mtp` launch the drafter's rope reads a slice written by the *target*, and
the failure appears exactly when the drafter's token count differs from the target's
padded count. `MtpProposer` subclasses `EagleProposer`, and the EAGLE `_propose`
carries the call — this is a missing propagation, not a rope-math defect: do not walk
`rope_forward_triton`'s kernel, `_triton_rope` or the model's rope parameters.

## Why the two counts differ (and why `--async-scheduling` is in the launch)

`pad_for_cudagraph` is `bs_to_padded_graph_size[n]` (vllm `config/vllm.py:339-344`,
table built at `config/compilation.py:1126-1138`): every count inside a bucket gap
maps **up** to the next capture size, every count above the largest capture size is
left alone.

The drafter pads only when `self.use_cuda_graph` holds
(`spec_decode/mtp_proposer.py:245-249`), and for `method: mtp` that flag is cleared
when async scheduling is on — `spec_decode/eagle_proposer.py:138-141`:
`use_cuda_graph = _use_aclgraph() and not speculative_config.enforce_eager`, then
`if self.method == "mtp": use_cuda_graph = use_cuda_graph and not use_async_scheduling`
(the comment at `:274-278` says aclgraph is disabled to avoid the per-step
synchronisation). With `--async-scheduling`, therefore: **target padded, drafter
raw**, and any step whose token count is not a capture size mismatches.

For `num_speculative_tokens: 1` a decode step schedules `1 + K = 2` tokens per request
(`uniform_decode_query_len = 1 + num_spec_tokens`, vllm
`v1/worker/gpu_model_runner.py:621`), so with `cudagraph_capture_sizes [1,4,8,16]`:

| requests in flight | scheduled tokens | target's padded count | `cos.numel()` at width 64 | outcome |
|---|---|---|---|---|
| 1 | 2 | 4 | 256 | 256 % 2 == 0 → **no crash, wrong rope** |
| 2 | 4 | 4 | 256 | correct |
| 3 | 6 | 8 | 512 | **the paste's `('[6, -1]' … size 512)`** |
| 4 | 8 | 8 | 512 | correct |
| 5 / 6 / 7 | 10 / 12 / 14 | 16 | 1024 | **crash** (`1024 % 10/12/14`) |
| 8 | 16 | 16 | 1024 | correct |
| ≥ 9 | ≥ 18 | unpadded | — | correct |

A prefill or mixed step whose scheduled-token total is 6 gives the same pair, so the
number to ask about is **that step's scheduled token count**, not the batch: `6` is
the whole classification, and the request count above is only how a decode step
reaches it.

**The tree's own GLM-4.x tutorial block is this launch shape**, so a reporter who
followed the docs is the *expected* way to hit it, not an exotic command. At
v0.14.0rc1 it is `docs/source/tutorials/GLM4.x.md:81-102`: `dp1tp16` (`:91`),
`--max-num-seqs 16` (`:96`), mtp `num_speculative_tokens: 1` with `model` = the
target directory (`:99`), `--compilation-config '{"cudagraph_capture_sizes":
[1,2,4,8,16,32], "cudagraph_mode": "FULL_DECODE_ONLY"}'` (`:100`),
`--async-scheduling` (`:101`) — plus the same `HCCL_BUFFSIZE` / `OMP_NUM_THREADS=10` /
`PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` / `HCCL_OP_EXPANSION_MODE=AIV`
exports. Run the same division on its list and the clean request counts are exactly
`1, 2, 4, 8, 16` and anything above 16; `3`, `5-7` and `9-15` raise
(`scripts/rope-cos-sin-slice-check.sh --capture-sizes 1,2,4,8,16,32 …`). So one
`rg -n 'cudagraph_capture_sizes' docs/source/tutorials/<page>.md` at the pin answers
"is this the tree's own recipe?" before anything is asked of the reporter — and for a
doc-audit of that page the same division *is* the finding.

## Do not derive the rope width from a model export

The width (`head_size` vs `head_size × partial_rotary_factor`) decides only whether
the division *raises* or *silently corrupts*; it does not change the owner, the grep
or the ladder. Two facts keep this from becoming a source hunt:

- `set_cos_and_sin` reads `hf_text_config.partial_rotary_factor` — the **checkpoint's
  own attribute**, not `rope_parameters` and not upstream's
  `config.rope_parameters.setdefault("partial_rotary_factor", 0.5)` (vllm
  `model_executor/models/glm4_moe.py:287`), which is what makes `rotary_dim = 64` for
  a `glm4_moe`-family 128-head model while the allocation can still be 128 wide.
- The snapshot usually cannot settle it: `transformers` is exported truncated and may
  carry no `glm4_moe` config at all, and the reporter's `transformers` version is
  often not the one in the tree. Deriving it from the paste instead is one line:
  `numel / padded_count` = width, and the *absence* of the crash on a step where the
  counts agree would mean the width is the head size (rows 2× too many, silent).

## Ladder

1. **Bound the failing step's token count — no restart, same body.**
   `temperature: 0`, one fixed prompt, the same capture sizes, and the concurrency
   walked 1 → 2 → 3 → 4: on the `[1,4,8,16]` / `K = 1` launch above, 2 in flight is
   clean, 3 raises, and **1 is the silent row** (wrong output, no exception) — so a
   clean single-request smoke test does not clear the build, and the curve is also why
   the reporter's run looked intermittent. Confirm the count from `--max-num-seqs`, the
   capture list and the concurrency actually used, not from the batch size alone.
2. **`--async-scheduling` off** — one restart; the drafter regains
   `use_cuda_graph`, pads to the same bucket as the target, and the counts agree
   (`eagle_proposer.py:141`) — *provided* `--speculative-config` does not also carry
   `enforce_eager: true`, which holds the drafter eager by itself and reproduces the
   crash. Cheapest unblock; costs the async-scheduling win.
3. **Cover every count you serve:** `cudagraph_capture_sizes` listing the actual
   `1+K` multiples (for `K = 1`, every even value up to `2 × --max-num-seqs`) leaves no
   bucket gap. The list holds **token** counts, so a concurrency `C` needs `2C` in it,
   not `C` — the tree's own advice ("we always ensure that the concurrency is actually
   included in the cudagraph_capture_sizes list",
   `docs/source/tutorials/Qwen3-Dense.md:395` at v0.14.0rc1) invites the wrong reading.
   Each capture costs time and memory, so state that.
4. **Repair (for a PR, not for the reporter):** refresh the global inside
   `MtpProposer._propose` after `num_input_tokens` is final, with the same slice the
   draft forward is about to use — `update_cos_sin(self.positions[:num_input_tokens])`
   — exactly as `eagle_proposer.py:506` does with
   `self._get_positions(num_input_tokens)`. The refresh is the two device ops the
   target already pays per step. A `try/except` around the view, or widening `_cos` to
   `max_num_batched_tokens` rows of *correct* values, is not the repair: the rows
   would still be the padded positions of another forward.

Not levers: `--dtype`, `--max-model-len`, card health, `ASCEND_LAUNCH_BLOCKING=1`, a
plog (host-side `.view`, nothing launched), the model's `rotary_dim`/`head_dim`, or any
`triton`/kernel build question.

**One trap on the other side: `"enforce_eager": true` inside `--speculative-config`
causes this crash rather than curing it.** That field only affects the *drafter*
(`eagle_proposer.py:138-139`), so it unpads the drafter while leaving the target on its
ACL-graph bucket — the same mismatch, without `--async-scheduling`. So a reporter who
already tried the per-spec `enforce_eager` row is running the failing shape, and the
global `--enforce-eager` (which removes the *target's* pad, `_use_aclgraph()` at
`model_runner_v1.py:392-393`) is a different flag: valid as a rung-2 alternative, at
the usual cost. Check which of the two the paste's command uses before crediting any
"toggled eager, still crashes" statement.

## Recheck before quoting

- Line numbers move and the proposer is refactored on later refs (a
  `AscendStep3p5MTPProposer` / `_run_merged_draft` split exists by the v0.19+ refs).
  Re-run the one grep per pin: **if `mtp_proposer.py` has gained the call, the defect
  is fixed at that ref** and the answer is the version, not this ladder.
- `set_cos_and_sin`'s width rule and `_cos`'s shape are allocation-time facts; a ref
  that starts recording per-request cos/sin (the file's own TODO) removes the global
  entirely.
- Whether the published fix looks like step 4 was **not** verifiable here: no PR or
  comment access in this snapshot, and the recorded case below closed no loop with the
  reporter. Say "the missing propagation is source-verified; the upstream repair shape
  is not".

## Recorded case

**#6449** (vllm-ascend `v0.14.0rc1`, vllm 0.14.1, GLM-4.7 + `mtp`, TP16, CANN 8.5.0,
`RuntimeError: shape '[6, -1]' is invalid for input of size 512` at
`ops/triton/rope.py:155`): the reporter's launch (posted afterwards) carries
`--async-scheduling`, `--max-num-seqs 16`,
`--speculative-config '{"num_speculative_tokens": 1, "method": "mtp", …}'` and
`--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes":
[1,4,8,16]}'`. That is the tree's own tutorial block (§ *Why the two counts differ*)
with the capture list cut to `[1,4,8,16]` and
`--max-model-len 131072 --gpu-memory-utilization 0.9 --served-model-name GLM-4.7
--reasoning-parser glm45 --tool-call-parser glm47 --trust-remote-code` added; its
`--speculative-config` `model` is the target directory (the in-checkpoint MTP head, not
a standalone draft), and the paste's `(Worker_TP10 pid=281)` is rank 10 of 16. So the
paste is exactly the pad gap in the table above, with `512 = 8 × 64` matching a padded
count and the `glm4_moe` rotary width, and `6` the failing step's scheduled tokens.
What that step *was* is not in the paste (a decode step of 3 in-flight requests at
`1 + K = 2` tokens each, or a prefill/mixed step of 6 scheduled tokens), and the crash
arrived one second after the engine's first `200 OK`, so do not present either
attribution as reporter-confirmed: the derivation is "6 scheduled tokens, padded to 8",
and rung 1 below is the confirmation.

The recorded triage run spent its whole 16 384-token output budget on the width
question (`_cos` 64 vs 128 wide, hunting a `glm4_moe` config in a truncated
`transformers` export) and shipped **no message text** — 48 messages, 1.64 M prompt
tokens, 24 tool calls, zero answer. The two greps this page starts with
(`update_cos_sin` across `vllm_ascend/`, and the division) were both available after
the first source read.
