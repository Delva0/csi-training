# PCP patch block under async spec decode: `cad.seq_lens_cpu` is `None`
# (`TypeError: 'NoneType' object does not support item assignment`, #8926)

## Signature

Request time, after a clean start (no load, capture or collective line anywhere in
the paste), one worker raises:

```text
TypeError: 'NoneType' object does not support item assignment
  File "…/vllm_ascend/spec_decode/eagle_proposer.py", line 1158, in set_inputs_first_pass
    cad.seq_lens_cpu[-num_prefill_reqs:] = seq_lens_p
```

with `propose_draft_token_ids` → `_propose` → `set_inputs_first_pass` above it and no
device frame. The launch carries **all three**: `--async-scheduling`,
`--speculative-config` with `num_speculative_tokens` > 0, and
`--prefill-context-parallel-size` > 1 (a full-graph / `FULL_DECODE_ONLY` block is
common but not required).

Recorded **#8926**: title says `v0.19.0rc1`, body's image is
`quay.io/ascend/vllm-ascend:v0.19.1rc1`, `--nnodes 2` A2 (`--tensor-parallel-size 8`,
`--enable-expert-parallel`), DeepSeek-V3.1-w4a8-mtp-QuaRot, `method: deepseek_mtp`,
K=3, `--prefill-context-parallel-size 2`, `--async-scheduling`,
`--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`. The worker prefix is
`Worker_PCP0_TP4_EP4`, so PCP is live on the failing rank. Verified at vllm-ascend
**v0.19.1rc1** (`da421af`) + vllm **v0.19.1** (`b1388b1`) — the reporter's own frame
lines match that pin one for one. Reporter's public follow-up (2026-05-08): PR **#8749**
resolves it, so the verdict is a **downstream defect fixed after the pin**; the PR's
diff is not readable in this replay, so the fix attribution is reporter-confirmed, not
source-verified.

## Why the attribute is `None` (source-verified at that pin)

| Location | Fact |
|---|---|
| `vllm_ascend/worker/model_runner_v1.py:2466-2469` | `if self.use_async_spec_decode: … seq_lens_cpu = None` — comment: *"GPU tensors are authoritative in async mode"* (`num_computed_tokens_cpu` too) |
| same call, `:2471-2481` | one `AscendCommonAttentionMetadata(...)` gets **both** `_seq_lens_cpu=self.optimistic_seq_lens_cpu[...]` (`:2479`) and `seq_lens_cpu=seq_lens_cpu` (`:2481`) — the CPU copy exists, just under the underscore name |
| `vllm_ascend/attention/utils.py:148,158` | `AscendCommonAttentionMetadata(CommonAttentionMetadata)` re-declares `seq_lens_cpu: torch.Tensor = None` as an **ordinary dataclass field** |
| vllm `vllm/v1/attention/backend.py:371,394-397` | upstream `seq_lens_cpu` is a deprecated **property** with a lazy fallback `if self._seq_lens_cpu is None: self._seq_lens_cpu = self.seq_lens.to("cpu")` — the downstream field shadows it, so the fallback never runs |
| vllm `vllm/v1/worker/gpu_model_runner.py:582-584` | `use_async_spec_decode = use_async_scheduling and num_spec_tokens > 0` — the three-flag launch above satisfies it |
| `eagle_proposer.py:1148,1152,1158` | the PCP metadata patch block runs only when the step has decodes **and** prefills; `:1157` writes the device `cad.seq_lens`, `:1158` the CPU one, unguarded |
| `vllm_ascend/worker/pcp_utils.py:164-175` | `init_batch_info`: `is_prefill = num_scheduled_tokens[:num_reqs] > self.decode_threshold`, `num_decode_reqs = first_prefill`, `num_prefill_reqs = num_reqs - num_decode_reqs` |

So the trigger needs a **mixed step** — one request scheduling more than
`1 + num_spec` tokens while others are decoding — not merely async + spec + PCP. That
is why the crash arrives when requests are staggered (a server under concurrency) and
not on a uniform `generate`.

## Read it as a defect, not a chart

`docs/source/user_guide/release_notes.md:21` (v0.19.1rc1 Features) states *"Eagle + PCP
+ Full Graph Mode: Support Eagle combined with PCP and full graph mode (#7924)"* — the
pin claims the combination, so the answer is "bug at this release", not "unsupported
combination" and not a model, checkpoint or weight problem.

## Ladder

1. **Drop `--async-scheduling`** (keep PCP=2, MTP, and the graph mode). Source-verified:
   with `use_async_spec_decode` False the `:2466` branch is not taken, so `:2481` passes
   a real CPU tensor and `:1158`'s operand exists. One axis; expect a decode-overlap
   performance cost, and say so.
2. **PCP=1 with async + MTP kept** — separates this block from the rest of the async
   path; if it still crashes the report is `async-spec-bookkeeping-desync.md`, not this
   page.
3. **Newer image / PR #8749.** A source walk at the pin cannot see the fix (one commit
   per repo in the snapshot); do not derive it.
4. A local `if cad.seq_lens_cpu is not None:` guard is **not** a fix: it silently skips
   the PCP CPU-side correction and the drafter then reads stale lengths. The repair
   writes the same slice into whichever copy the async path carries (`_seq_lens_cpu`) —
   and the acceptance check is output identity against the async-off run on a **mixed**
   workload (staggered concurrent long + short prompts), not the absence of a
   traceback.

## Coverage, and why the pin's own tests are silent

- `tests/ut/spec_decode/test_eagle_proposer.py` pins `self.runner.pcp_size = 1` at
  `:37,175,306,462,536,1242,1715` — the PCP block is never entered in UT.
- `tests/e2e/multicard/4-cards/long_sequence/test_mtp.py` does carry
  `test_pcp_dcp_mtp3_eager` (`:61-79`, `async_scheduling=True`, PCP 2, K=3), so
  async+PCP+MTP is exercised; but every graph row
  (`test_pcp_dcp_mtp3_piecewise_graph:82`, `test_pcp_dcp_mtp3_full_graph:106`) runs
  `async_scheduling=False`.
- No row builds a mixed step (a single `generate_greedy` over a uniform prompt list), so
  the `num_prefill_reqs > 0` arm — the only place the assignment runs — is untested.
  Label the mixed-step reading as inference from the test bodies: no test run was
  observed.

## Delivery note (recorded run)

The answer's first paragraph held the verdict and rung 1 (correct), then the evidence
table consumed the rest of the 16 384-token output budget and the ladder never landed
(`stop_reason: output_tokens`). Type the ranked causes and the ladder **before** the
table — see `SKILL.md` §Answer shape.
