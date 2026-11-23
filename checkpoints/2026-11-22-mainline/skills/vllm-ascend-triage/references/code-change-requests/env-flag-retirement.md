# Retiring a `VLLM_ASCEND_*` env switch: the sweep, worked on #1262

**Recorded #1262** — "*[Bug]: Remove `VLLM_ASCEND_ENABLE_NPU_MROPE` before 0.9.1
final release*", body = the PR link plus one review-comment link. Pins:
vllm-ascend **v0.9.0rc2 `8dd686d`**, vllm **v0.9.0 `5873877`**. PRs, comments
and release branches are unavailable in the workspace.

## Step 1 — two greps decide whether the change is even in this snapshot

```bash
rg -n 'VLLM_ASCEND_ENABLE_NPU_MROPE' <vllm-ascend root>      # the identifier
rg -n -i 'mrope' <vllm-ascend root>/vllm_ascend --include=*.py # the mechanism
rg -n -i 'mrope' <vllm-ascend root>/csrc                      # a custom kernel?
```

At v0.9.0rc2 both come back empty apart from position plumbing
(`vllm_ascend/worker/model_runner.py`, `worker/model_runner_v1.py`: `uses_mrope`,
`_calc_mrope_positions`, the `mrope_positions` buffers) — no `envs.py` key, no
`npu_mrope` op, no `mrope` in `csrc/` (only `csrc/kernels/pos_encoding_kernels.cpp`).
That is the verdict sentence: **the pinned tree predates the PR, so the removal
cannot be applied here**; the checklist below is for the branch that carries it.
Do not substitute a later release's knowledge for the pinned ref.

**A release-note bullet is not proof the code is gone.** At v0.23.0
(`5cb98caaadef`) `docs/source/user_guide/release_notes.md:1408` (under
`## v0.12.0rc1`, `:1395`) says *"we removed the env `COMPILE_CUSTOM_KERNELS`. You
can not set it to 0 now."*, while `vllm_ascend/envs.py:43` still reads it (default
`1`) and `Dockerfile:58` still passes it as a build arg. When a request asks you
to remove a switch "already retired in release N", read the note **and** the
definition: the definition is the behaviour, and the stale note is a separate doc
defect (recorded #17718, graded in
`../tutorial-verification/missing-guidance-and-contributor-docs.md`).

## Step 2 — the touch-point sweep (anchors verified at v0.9.0rc2)

| Site | What to change | Anchor at this ref |
|---|---|---|
| definition | delete the dict entry; the dict is the single source of truth | `vllm_ascend/envs.py:29` inside the `begin-env-vars-definition` (`:27`) / `end-…` (`:138`) markers, lazy `__getattr__` at `:141` |
| doc row | **nothing** — the page is a `literalinclude` of that marked block, so deleting the entry deletes the row. Never hand-edit the doc, and say so, because reviewers expect a doc hunk | `docs/source/user_guide/env_vars.md:5-9` |
| reader | drop the `envs_ascend.VLLM_ASCEND_ENABLE_<x>` read and keep the shape/dtype predicate next to it | e.g. `vllm_ascend/ops/rotary_embedding.py:28-29` (`fp16 && neox_style && head_size % 32 == 0 && CUSTOM_OP_ENABLED`) is the predicate pattern that survives |
| tests / CI | `@patch.dict(os.environ, {...})` sites for the same family | `tests/singlecard/test_offline_inference.py:111`, `tests/multicard/test_offline_inference_distributed.py:64,86` |
| release note | one bullet under the target release: the switch is gone / the path is now unconditional | convention at `docs/source/user_guide/release_notes.md:34` (the DBO bullet) |

## Step 3 — the default is the behaviour change

Read the lambda before proposing anything: a `lambda: bool(int(os.getenv("<X>", '0')))`
entry means the path is **off by default**, so deleting the switch turns it on
for everyone, and the release note (not the code comment) is where that is
recorded. Users' scripts that set `=0` keep working silently: the variable is
only read through the dict, so a stale export cannot raise. State that, because
it removes the "we need a deprecation warning" argument.

## Step 4 — check reachability of the code the switch gates

A downstream rope/op shim bound as `<Class>.forward_oot = …` is reached only
through upstream `CustomOp.forward` → `self._forward_method`
(`vllm/model_executor/custom_op.py:18-23`, `dispatch_forward` at `:70-99`,
out-of-tree branch `:96-97`). Any upstream subclass that defines its own `forward` never
takes it: at vllm v0.9.0 `MRotaryEmbedding.forward` is defined at
`vllm/model_executor/layers/rotary_embedding.py:1008`, so the tree's
`RotaryEmbedding.forward_oot = rope_forward_oot`
(`vllm_ascend/ops/rotary_embedding.py:273`) does **not** cover mRoPE. Consequence
for the answer: the flag in #1251 must gate a *new* binding (a `forward` rebind or
a separate mrope op), so the removal touches that binding, not just `envs.py` —
and any claim "the flag gates the fused rope path" has to name the binding line.

## Step 5 — validation, stated as manual when there is no coverage

`rg -l 'qwen2_5_vl|qwen2_vl' tests/` returns nothing at v0.9.0rc2: the VL path has
no unit or e2e test, so a default flip is guarded by no CI. The answer's last
section is therefore "run Qwen2.5-VL on one NPU, one image request and one
text-only request, eager and graph mode, and diff against the pre-removal build",
plus the mRoPE consumers to watch (`model_runner_v1.py` `mrope_positions`
buffers, `_calc_mrope_positions`, the `positions[:, :num_input_tokens]` slice) —
a wrong mRoPE shows up as garbage VL answers, not a crash.

## Open items the snapshot could not settle (say them)

- The PR's own diff, the review comment, the flag's exact spelling/default and
  the target release branch are unavailable; only the absence at v0.9.0rc2 is
  verified.
- Whether the paired `torch_npu` exports an `npu_mrope` op: no hit in the
  `Ascend/pytorch` export (v7.1.RC1.alpha002, **truncated** 1190/1240 files) and
  none in `Ascend/op-plugin`; a truncated export is not evidence of absence, so
  label it unverified.
