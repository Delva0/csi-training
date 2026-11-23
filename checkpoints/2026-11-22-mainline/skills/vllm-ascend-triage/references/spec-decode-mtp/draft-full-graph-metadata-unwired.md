# Draft FULL-graph update reads a field nothing writes (`draft_attn_metadatas`)

Scope: the engine started and capture finished — the paste shows `Replaying aclgraph`
on the target workers — and the **first request** then kills the workers with

`AttributeError: 'ForwardContext' object has no attribute 'draft_attn_metadatas'`

deepest frame `vllm_ascend/attention/attention_v1.py` `update_graph_params`, stack
`worker/model_runner_v1.py propose_draft_token_ids` → `eagle_proposer.py _propose`
→ `_update_full_graph_params` → `compilation/acl_graph.py update_full_graph_params`
→ `impl_cls.update_graph_params`. A dense-attention drafter (`Qwen3` GQA family) on
`--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'` with
`--speculative-config` `method: eagle3|eagle`.

Not these siblings: a device dump whose frames are `execute_dummy_batch` /
`acl_graph.__call__` (`draft-graph-capture-device-fault.md`), an op-level refusal
with two numbers that must be equal (`fia-tnd-seqlen-mismatch.md`), or a KV-side
`107030` at the sequence cap (`max-model-len-boundary-fia-crash.md`). Those launch
something and the device/op complains; here Python cannot read an attribute, on
every rank in the same second, and the message names the missing field.

## The answer, in the first paragraph

**Real downstream defect at vllm-ascend v0.14.0rc1, not the reporter's flags,
hardware, CANN or quantization.** The parameter is accepted by the drafter's helper,
thrown away, and read back off `ForwardContext`, where nothing ever assigns it. Two
rungs, cheapest first:

1. `--compilation-config '{"cudagraph_mode":"PIECEWISE"}'` (or `--enforce-eager`).
   The call site is guarded by
   `if forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL:`
   (`spec_decode/eagle_proposer.py:563-566`), so piecewise/eager never enters the
   broken function. `--enforce-eager` also costs graph speed; PIECEWISE costs less.
2. Upgrade: the repair is in **v0.17.0rc1** (below). If their site can move, that
   beats a local patch; otherwise the one-line assignment is rung 3.

## Mechanism (vllm-ascend `52d4acf` = v0.14.0rc1 ↔ vllm `d7de043` = v0.14.1)

| Site | What it does |
|---|---|
| `spec_decode/eagle_proposer.py:563-566` | `if forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL:` → `self._update_full_graph_params(forward_context, num_input_tokens, multi_steps_attn_metadata)` — the per-draft-step metadata list *is* supplied |
| `spec_decode/eagle_proposer.py:1180-1183` | signature `(self, forward_context, num_tokens, draft_attn_metadatas=None)`; the parameter is **never used or forwarded** |
| `compilation/acl_graph.py:213-233` | `update_full_graph_params(...)` has no such parameter; only `vllm_config` / `speculative_config` / `num_dcp_pcp_tokens` reach `impl_cls.update_graph_params` |
| `attention/attention_v1.py:375-382` | `update_graph_params(...)` likewise accepts neither, so the value has no route to the reader |
| `attention/attention_v1.py:437-453` | `if forward_context.is_draft_model:` → `graph_params = get_draft_graph_params()` → `attn_metadata = forward_context.draft_attn_metadatas` → `AttributeError` |

Two corroborations that the intended value is exactly `multi_steps_attn_metadata`:
the next lines index it per step (`attn_metadata[0].keys()` at `:440`,
`len(graph_params.attn_params[num_tokens]) // num_layers` at `:456`), and
`rg -n 'draft_attn_metadatas' vllm_ascend/` returns **two** hits at this pin — that
read and that unused default. `ascend_forward_context.py` never sets it, and the
upstream vLLM v0.14.1 tree has zero occurrences of the name (verified: the field is
not an upstream rename that downstream missed).

**Why it fires on every rank at once, on this launch:** `attention/utils.py:44-46`
returns `False` from `using_paged_attention()` as soon as `speculative_config is not
None`, so the paged branch (`attention_v1.py:383-435`, which correctly reads
`forward_context.attn_metadata[key].seq_lens`) is unreachable under *any* spec-decode
launch; the FIA branch at `:437-439` is taken unconditionally, and only when
`forward_context.is_draft_model` — i.e. only for the drafter, which is why the target
model's `Replaying aclgraph` lines look healthy just before the death.

**Why dense-only:** the working sibling is MLA — `attention/mla_v1.py:724-740`
iterates `forward_context.attn_metadata` for the draft model too. So a DeepSeek-family
draft (MTP or eagle) never reaches `attention_v1.py:439`. Dense **MTP** does:
`MtpProposer(EagleProposer)` (`mtp_proposer.py:23`) inherits the same helper and calls
it at `:123` with two arguments only, and `forward_context.is_draft_model` is true
there as well — the scope line in the answer is "dense-attention drafter", not
"eagle only".

## Why the rc shipped with it — read the tree's own e2e, not its absence

`tests/e2e/singlecard/spec_decode/test_mtp_eagle_correctness.py` at v0.14.0rc1 holds
both shapes, and the difference is the statement:

- `test_deepseek_mtp_correctness` (`:52`) parametrizes
  `cudagraph_mode` `["PIECEWISE", "FULL_DECODE_ONLY"]` — the FULL-graph × MTP
  coverage that *did* run, on DeepSeek (MLA), which is exactly the backend that does
  not hit the bug.
- `test_llama_qwen3_eagle_correctness` (`:121`) is the dense one: `method` `eagle` /
  `eagle3`, `Qwen/Qwen3-8B` + `RedHatAI/Qwen3-8B-speculator.eagle3` in
  `VALID_COMBINATIONS` (`:42-46`), both `VllmRunner`s constructed with
  `cudagraph_mode="FULL_DECODE_ONLY"` (`:174-188`) — and it is decorated
  `@pytest.mark.skip(reason="Failed with CANN8.5, fix me")`. The file is wired in CI
  (`.github/workflows/_e2e_test.yaml:128`), so the skipped dense case is the
  project's own statement at that ref: this combination was known-failing, not
  unvalidated.

**An absence claim has to survive this file.** The earlier reading of this case —
"Qwen3 (dense) × eagle3 × FULL graph is not covered by any test in this tree" — is
wrong at the pin, and a maintainer can refute it with one `rg`. Grep the test file
before writing "no coverage".

## The repair, and how to tell whether the reporter's tree has it

At **vllm-ascend v0.17.0rc1 (`e20f0b1`)** the parameter is threaded end to end, the
same greps now return the plumbing:

- `compilation/acl_graph.py:223,233` — `draft_attn_metadatas=None` in the signature
  and passed through as the last positional argument;
- `attention/attention_v1.py:390-394` — `update_graph_params(..., draft_attn_metadatas=None)`;
- `attention/attention_v1.py:450-453` — `if _EXTRA_CTX.is_draft_model:` →
  `attn_metadata = draft_attn_metadatas` (the read moved off `ForwardContext`,
  which in the same refactor became `_EXTRA_CTX`);
- `spec_decode/eagle_proposer.py:1559-1567` — the helper forwards
  `draft_attn_metadatas=draft_attn_metadatas`;
- `mla_v1.py:737` — same parameter on the MLA side;
- and the dense e2e lost its skip marker (`test_llama_qwen3_eagle_correctness` at
  `:130` has no `pytest.mark.skip`), which is the tree's own statement that the
  combination now passes.

So the version scope is: **broken at v0.14.0rc1, repaired by v0.17.0rc1.** The
one-command check on any other ref is
`rg -n 'draft_attn_metadatas' vllm_ascend/` — two hits with a `ForwardContext` read
means broken; a parameter plus an assignment/thread means repaired. Do not ask for
another launch to decide it.

Local patch for a site that cannot upgrade (`spec_decode/eagle_proposer.py:1180`),
which satisfies both the read and the per-step indexing:

```python
def _update_full_graph_params(self, forward_context, num_tokens, draft_attn_metadatas=None):
    forward_context.draft_attn_metadatas = draft_attn_metadatas
    update_full_graph_params(
        self.runner.attn_backend, self.update_stream, forward_context, num_tokens,
        self.vllm_config, self.vllm_config.speculative_config)
```

The cleaner fix is the plumbing above (mirrors `mla_v1`), but it touches two backend
signatures and a shared helper.

## Ladder

1. `--compilation-config '{"cudagraph_mode":"PIECEWISE"}'` — the guard at
   `eagle_proposer.py:563` skips the broken call; measure the graph speed they lose
   before recommending it permanently.
2. `--enforce-eager` — same effect, no graphs at all.
3. Upgrade to a release whose `draft_attn_metadatas` is plumbed (v0.17.0rc1 or
   later); re-run the two greps on *their* tree first.
4. Local assignment patch (above) if 1-3 are unavailable — state that
   `num_speculative_tokens > 1` is untested even with the patch: the
   `attn_keys * (len(...) // num_layers)` block at `attention_v1.py:456` assumes
   per-step metadata, which the plumbing supplies and the bare assignment also
   supplies, but the recorded case only ran `n=1`.

Not levers: `draft_tensor_parallel_size`, `disable_padded_drafter_batch`,
`--quantization ascend`, `--enable-expert-parallel`, `--data-parallel-size`, TP size.
The recorded launch carried all of them and the traceback ends in a missing attribute.

## What to send back / record in the answer

The traceback's deepest three `vllm_ascend` frames, `collect_env.py`, the full
`--compilation-config` and `--speculative-config`, the target architecture family
(dense GQA vs MLA/DSA — that is the scope line), and `npu-smi`/CANN version only if
they claim the graph mode works elsewhere. Keep the mechanism paragraph short: this
shape needs one grep and four reads, and the recorded #6509 session shipped its whole
ranked answer — verdict, mechanism table, ladder, scope — inside a 9.6 K-completion
session. The expensive failure mode here is walking `acl_graph.py`'s capture path,
which is not where the exception is raised.
