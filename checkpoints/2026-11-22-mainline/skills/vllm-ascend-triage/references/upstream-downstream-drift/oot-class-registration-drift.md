# Drift through the name-keyed OOT class registry

Recorded #6784: image `quay.io/ascend/vllm-ascend:v0.14.0rc1-a3-openeuler`,
LongCat-Flash, DP/TP/EP worker, crash during `profile_run`:

```
File "/vllm-workspace/vllm/vllm/model_executor/layers/fused_moe/zero_expert_fused_moe.py", line 152, in forward
  topk_weights, topk_ids = self.select_experts(
File ".../torch/nn/modules/module.py", line 1964, in __getattr__
  raise AttributeError(
AttributeError: 'ZeroExpertFusedMoE' object has no attribute 'select_experts'
```

The reporter's follow-up line — "没有成功走到 vllm-ascned 的ops定义里边去？" ("it never
reached vllm-ascend's op definitions?") — is the right question and the answer is yes.

## Why this belongs here even though the failing frame is upstream

This topic is keyed on "the frame is in `vllm_ascend/{ops,patch,models}`". That
gate misses the shape where the crash is raised **inside an upstream vLLM class
that vllm-ascend is supposed to replace**: the object is upstream's, the
substitution is downstream's, and the owner is downstream. Route here whenever
an `AttributeError`/`TypeError` names an upstream vLLM layer class together with
a vllm-ascend version, or when the reporter asks whether the Ascend implementation
was reached.

## Mechanism (verified)

- vllm-ascend `v0.14.0rc1` (`52d4acfa51fb`): `vllm_ascend/utils.py`
  `register_ascend_customop()` builds `REGISTERED_ASCEND_OPS` — a **name → class
  dict** — whose MoE entries are only `"FusedMoE": AscendFusedMoE` and
  `"SharedFusedMoE": AscendSharedFusedMoE` (`:713-714`), then loops
  `CustomOp.register_oot(_decorated_op_cls=op_cls, name=name)` (`:738-739`).
- vllm `v0.13.0` (`72506c9834`): `vllm/model_executor/custom_op.py:20-39`
  `CustomOp.__new__` dispatches on `cls.__name__` through `op_registry_oot`;
  `register_oot` (`:177-197`) stores by name. So **any upstream class whose name
  is not in the map instantiates as itself**, keeps the upstream forward body and
  gets no Ascend behaviour, silently.
- Consequence: a new upstream layer class introduced by an upstream refactor is
  invisible to vllm-ascend until someone adds it to that dict. Recheck the
  dispatch form at the reporter's version — vllm `v0.18.0` has both
  `PluggableLayer` (`custom_op.py:31`, `register_oot` at `:83`) and `CustomOp`
  (`:102`), and moved the MoE routing into `fused_moe/router/`; vllm-ascend
  `v0.18.0` still registers by name (`vllm_ascend/utils.py:696-697`
  `CustomOp.register_oot(..., name=name)`) and its map at `:634-656` *still* lists
  only `FusedMoE`/`SharedFusedMoE` with no `ZeroExpertFusedMoE` entry. The keying
  and the gap both survive the refactor, so the check below works on either
  version. `AscendFusedMoE` itself is a plain
  `class AscendFusedMoE(FusedMoE)` (`vllm_ascend/ops/fused_moe/fused_moe.py:158`),
  so modelling the new class on the same pattern is not the problem — the
  *registration* is.

## The traceback file is version evidence, not an upstream bug

- Pinned vllm `v0.13.0` has **no** `fused_moe/zero_expert_fused_moe.py`; zero
  experts are inline: `FusedMoE.select_experts` at `fused_moe/layer.py:1523`
  returns `(topk_weights, topk_ids, zero_expert_result)`, and
  `models/longcat_flash.py:294-310` passes `zero_expert_num=` /
  `zero_expert_type=` into `FusedMoE(...)`.
- The later pinned vllm `v0.18.0` (case #11519) shows the refactor this image
  must already carry: `ZeroExpertFusedMoE(FusedMoE)` at
  `fused_moe/zero_expert_fused_moe.py:13`, its `forward` calling
  `self.select_experts` at `:152`, and `select_experts` no longer defined on
  `FusedMoE` — routing moved to `fused_moe/router/base_router.py:203` and
  `router/fused_moe_router.py:22`.
- Therefore: the file missing from the *pinned* tree means the image ships a
  newer upstream than the snapshot pinned, not that upstream lacks the file.
  Write it as a snapshot-coverage gap and stop looking for the line. (Kill this
  misreading early: it costs several calls and produces a wrong "upstream bug"
  claim.)

## Where the real pairing is (2 calls, not a requirements hunt)

- `docs/source/community/versioning_policy.md:21-45` release compatibility
  matrix; at v0.14.0rc1 the matching row is `:26`
  `v0.14.1rc1 | v0.14.0/v0.14.1`, and `:58` pairs `main` with
  `d68209402ddab3f54a09bc1f4de9a9495a283b60, v0.14.1 tag`.
- `.github/workflows/pr_test_full.yaml:78` and `pr_test_light.yaml:88,100`
  pin `vllm_version: [d68209402ddab3f54a09bc1f4de9a9495a283b60, v0.14.1]`.
- `requirements.txt` does **not** pin vllm at this commit (its only `vllm`
  mention is an opencv comment at `:19`) — a recorded call grepped it for
  `vllm==` and learned nothing. Get the pair from the matrix/CI, and ask the
  reporter for `pip show vllm` / `vllm.__version__`.

## Ranked causes and the one check each

1. **Downstream registration gap** (most likely): the reporter's vllm-ascend does
   not register the upstream class the model now builds, so upstream code runs
   with a hook the platform layer is expected to supply. Check in the image:
   `python -c "from vllm.model_executor.custom_op import CustomOp as C;
   print(sorted(C.op_registry_oot)); from
   vllm.model_executor.layers.fused_moe.zero_expert_fused_moe import
   ZeroExpertFusedMoE as Z; print(hasattr(Z,'select_experts'), Z.__mro__[:3])"`.
2. **Half-landed upstream refactor in the image's vllm** (CI tests both a main
   commit and the release tag, so an image may carry the dev commit). Check the
   installed `vllm.__version__` / `pip show vllm` and whether that commit's
   `ZeroExpertFusedMoE` inherits the class that owns `select_experts`.
3. **Image/vllm pairing the reporter did not intend** (A3 images are separate
   builds): ask for the exact tag and the vllm version printed by the engine.
4. **Ruled out at the pins, do not repeat:** "LongCat-Flash is unsupported"
   (`docs/source/user_guide/release_notes.md:71` — supported since #3833;
   `vllm_ascend/quantization/modelslim_config.py:169`;
   `vllm_ascend/worker/model_runner_v1.py:2273` has a `longcat_flash` case), and
   "the op map is wrong for this model" (the model builds the ordinary MoE
   layer).

## Cheapest action for the reporter

Move to the release the compatibility matrix pairs with the installed vllm, or to
a vllm-ascend whose registration map covers the class — both are same-day
actions. Patching the upstream forward, or deleting `zero_expert_*` from the
checkpoint, are patch proposals: label them as such.

## Still open

Whether vllm `v0.14.0/v0.14.1` defines `select_experts` on `FusedMoE` cannot be
settled here — vllm v0.14.x is not in the snapshot, and v0.13.0/v0.18.0 bracket
it without being it. If it does, cause 1 weakens and cause 2 (a mixed install)
becomes the lead. The check under cause 1 settles it on the reporter's host.
