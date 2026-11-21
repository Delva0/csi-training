# `method: draft_model` in serial mode: the hidden-state copy is sized by the draft

## When this page applies

A clean start, then the **first request** kills the worker with

```text
RuntimeError: The expanded size of the tensor (1024) must match the existing size (4096)
at non-singleton dimension 1.  Target sizes: [5, 1024].  Tensor sizes: [5, 4096]
… line 1008, in set_inputs_first_pass
self.hidden_states[:num_tokens] = target_hidden_states
```

on a launch carrying `--speculative-config '{"method": "draft_model", "model": <draft>,
"num_speculative_tokens": K}'` with **no** `parallel_drafting`. The pair's hidden sizes
differ (draft 1024, target 4096 above). The same launch with `method: eagle|eagle3|mtp`
is a different lane, and an equal-hidden pair runs this code path silently. Recorded
#7365: vllm-ascend v0.17.0rc1 (`e20f0b1a0d2f`) + vllm v0.17.0 (`b31e9326a7d9`),
Qwen3-8B target + Qwen3-0.6B draft, K=5, one 910B3.

## Step 0 — the two numbers *are* the classification

`Target sizes` is the destination buffer (the **draft's** hidden size), `Tensor sizes`
the target's hidden states. So `H_draft != H_target` is necessary for this crash; equal
numbers means the paste belongs somewhere else. One command closes it on the reporter's
box (their checkpoints, not ours):

```bash
python -c "from transformers import AutoConfig as C; print(C.from_pretrained('<draft>').hidden_size, C.from_pretrained('<target>').hidden_size)"
```

Heterogeneous pairs are an *intended* case upstream — `self.hidden_size =
self.draft_model_config.get_hidden_size()` carries the comment "the draft model's hidden
size can be different from the target model's (e.g., Llama 3.3 70B)"
(`vllm/v1/spec_decode/eagle.py:79-82` at v0.17.0) — which is why "your draft is too
small" is not the answer.

## Step 1 — find out *which* `__init__` computed `needs_extra_input_slots`

The failing copy sits in the `if not self.needs_extra_input_slots:` branch (the tree
calls it the "Default EAGLE pathway"); the sibling copy in the `else` branch is guarded
by `if self.pass_hidden_states_to_model:` (`vllm_ascend/spec_decode/eagle_proposer.py:1052`).
So the question is not about `draft_model` itself but about which flags the object was
built with:

```text
extra_slots_per_request       = 1 if not parallel_drafting else num_speculative_tokens
net_num_new_slots_per_request = extra_slots_per_request - (1 if pass_hidden_states_to_model else 0)
needs_extra_input_slots       = net_num_new_slots_per_request > 0
```

Those three lines run **once**, inside whichever `__init__` receives
`pass_hidden_states_to_model`. One call decides the ref:

```bash
rg -n 'needs_extra_input_slots|net_num_new_slots_per_request|class .*SpecDecodeBaseProposer' \
  <vllm_ascend>/vllm_ascend/spec_decode/ <vllm>/vllm/v1/spec_decode/
```

At v0.17.0rc1 that read gives the fork-order bug: `AscendDraftModelProposer(...)`
(`draft_proposer.py:13`) inherits `vllm_ascend`'s `SpecDecodeBaseProposer`, whose own
`super().__init__(vllm_config, device, runner)` is upstream **`EagleProposer.__init__`**
— and that one hard-codes `pass_hidden_states_to_model=True`
(`vllm/v1/spec_decode/eagle.py:1691-1696`). The base init therefore computes
`net = 1 − 1 = 0` → `needs_extra_input_slots = False`, sizes `self.hidden_states` from
the draft hidden (`eagle.py:161-163`), and only afterwards does the fork reassign
`self.pass_hidden_states_to_model = False` (`eagle_proposer.py:95`) without recomputing
anything. A `draft_model` launch runs the EAGLE fast path and copies target-width hidden
states into a draft-width buffer.

## Step 2 — owner: downstream at that ref, and the direction flips between refs

- **Upstream is not the owner.** Upstream `DraftModelProposer` passes the flag into the
  init that computes it (`vllm/v1/spec_decode/draft_model.py:19-30`), so upstream gets
  `net = 1 − 0 = 1` → `True` → the `else` branch, where the copy is guarded
  (`eagle.py:766`). Do not ship "reproduces on GPU / report upstream": the line exists
  upstream but is unreachable from `method: draft_model` there.
- **The flag is a fork artifact, so re-derive it per ref** rather than carrying a verdict
  across versions:

  | Ref | downstream base that computes the flags | serial `draft_model` → `needs_extra_input_slots` | what the reporter sees |
  |---|---|---|---|
  | v0.17.0rc1 (`e20f0b1a`) | `SpecDecodeBaseProposer(EagleProposer)` → upstream init with the flag `True` (`eagle_proposer.py:89,95`) | `False` | this `RuntimeError` on a heterogeneous pair; silent success on an equal pair |
  | v0.22.1rc1 (`5f6faa0c`) | `AscendSpecDecodeBaseProposer(SpecDecodeBaseProposer)` → upstream base init with the flag `False` (`llm_base_proposer.py:133-137`) | `True` | the init gates fire instead (`_raise_if_mrope` family, recorded #14453) — no hidden copy at all |

  The second row is the one that trips #14453's `NotImplementedError`; read
  `../upstream-downstream-drift/unsupported-path-raise.md` (inverse section) before
  using either verdict, and re-grep `class` + `super().__init__` at the reporter's ref.

## Step 3 — coverage, quotable as coverage only

At v0.17.0rc1 the tree's sole `draft_model` e2e case is
`tests/e2e/singlecard/spec_decode/test_v1_spec_decode.py:447-500`
(`test_parallel_drafting_acceptance`): `"parallel_drafting": True`, `K=8`,
`amd/PARD-Llama-3.2-1B` (hidden 2048) against `Llama-3.1-8B-Instruct` (4096). A
heterogeneous pair is therefore validated **only through the guarded branch**, and the
serial path has no case at all — which is why this crash was never caught. Say that; do
not read it as "the combination is unsupported".

## Step 4 — ladder

1. `"parallel_drafting": true` with the reporter's K: `extra = K`, so `net = K − 1 > 0`
   and the guarded branch runs (their K=5 is fine). **`K=1` recomputes to `net = 0` and
   lands back on this crash**, so never offer the flag without K ≥ 2. Read the upstream
   field text aloud: "requires the speculative model be trained to support parallel
   drafting" (`vllm/config/speculative.py:129-133`) — Qwen3-0.6B is not, so this is a
   crash-avoidance run with low acceptance, not the repair.
2. An EAGLE3 speculator for the same target (`eagle3` row of the same e2e file:
   `RedHatAI/Qwen3-8B-speculator.eagle3`) — the tree's validated draft-side route for
   this target, and the one that keeps acceptance meaningful.
3. `"method": "ngram"` / `"suffix"` — one launch that separates the target/verify side
   from the draft side.
4. Structural repair (vllm-ascend owns it): thread `pass_hidden_states_to_model` into the
   init that computes the three fields, or recompute them after the reassignment **and**
   allocate `is_rejected_token_mask` / `is_masked_token_mask`, because the `else` branch
   asserts both non-`None` (`eagle_proposer.py:1012-1013`) and the base init only
   allocates them when its own `needs_extra_input_slots` was `True`
   (`eagle.py:176-190`). An exception-free start is not acceptance: require
   `temperature=0` output token-identical to the same request without spec decode, and
   `vllm:spec_decode_num_accepted_tokens_total` > 0.

## Recorded delivery

#7365's session reached the mechanism in 24 tool turns and shipped a full answer (the
buffer, the branch, the guard, the ladder) — but with the owner inverted, "not an
Ascend-specific defect … a GPU run of your snippet should fail identically", which is
the one claim that changes the reporter's next action. The cheap correction is Step 1's
grep: *which* `__init__` receives the flag, and whether the fork reassigns it afterwards.
