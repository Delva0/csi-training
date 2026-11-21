# `AttributeError: '<Ascend…310>' object has no attribute '_<name>'` — the 310P sibling drifted from the A2 class

Scope: a 310P report whose model **loads** (weights OK, fp16 cast OK) and then dies in
the **first forward** — `profile_run` / `determine_num_available_blocks` — with
`AttributeError: '<SomeClass>310' object has no attribute '_something'`, where the
missing name is a helper the caller uses as a method (`self._x(...)`). The owner is
almost never the model: `vllm_ascend/_310p/**` carries hand-copied siblings of classes
in `vllm_ascend/ops/**`, and a copy can carry the *caller* without the *callee*.
Deliverable: which class, which name, where the definition that the copy lost lives,
and the one-file fix. Not this topic: an `ImportError`/`AttributeError` at import time
(`../startup-failures/plugin-load-module-missing.md`), a `check_ascend_device_type()`
assertion before load (`device-type-build-vs-runtime.md`), or a forbidden family on
this SoC with no traceback (`atlas-300i-310p.md`).

## Procedure (four greps, no source walk)

1. **Read the class header and its `__init__`** — `sed -n '<class_start>,<+40>p'`. Record
   the base tuple and every attribute actually assigned. A 310P subclass of the
   upstream op class (`class AscendSharedFusedMoE310(SharedFusedMoE, AscendFusedMoE310)`)
   typically calls the *Ascend* parent's `__init__` directly, so anything the class does
   not assign itself must come from that parent or not exist at all.
2. **Grep the missing name across the whole package**, not the file:
   `grep -rn '_shared_experts_part' vllm_ascend/`. If the only definitions sit in the
   non-310P sibling under `ops/`, it is copy drift — say so with both `path:line`s.
3. **Grep the registry entry that routes 310P to that class**
   (`utils.py` `if is_310p(): … REGISTERED_ASCEND_OPS.update({...})`). That is what makes
   a model on this SoC instantiate the copy at all; it also tells you whether the drift
   is reachable for *this* model family (here: `"SharedFusedMoE"` → the 310 class).
4. **Check the second latent contract in the same copy before answering.** A copied
   caller usually copies the *return shape* too: compare the sibling's signature and
   return with the parent's. If `forward_impl` returns a bare tensor and the copy does
   `a, b = Parent.forward(…)`, the reporter hits that error as soon as the first one is
   patched — name both defects in one answer and label the second as "expect this next".
5. **State coverage as coverage**: `grep -rln '<Class>' tests/` empty at the pin means no
   unit/e2e test constructs the class — quotable as *why it shipped broken*, never as the
   cause.

## Worked case: #12856 (Atlas 300I Duo ×8, image `vllm-ascend:v0.18.0-310p-openeuler`, Qwen3.6-35B-A3B)

Reporter's own run had already retired two lanes — without `--dtype half` the card
rejects bf16 (`AclNN_Parameter_Error(EZ1001): Self dtype DT_BF16 not support in current
soc version`), and without `--enforce-eager` the crash is a dynamo unpack error
(`Can't unpack a tensor of 4096 rows into a tuple of 2 elements`, 4096 = their
`--max-num-batched-tokens`). Both are consistent with **one** broken class, not two
causes:

| Claim | Anchor (vllm-ascend v0.18.0, `e18643f8`) |
|---|---|
| 310P shared-expert MoE class, bases `SharedFusedMoE, AscendFusedMoE310` | `vllm_ascend/_310p/fused_moe/fused_moe.py:262` |
| `__init__` assigns `_routed_input_transform`, `_shared_experts`, `use_overlapped`, `shared_expert_stream`, `_gate` — and nothing named `_shared_experts_part*` | `:263-277` |
| the caller uses them | `:301-302` (`_forward_shared_experts`), reached from `:315` (`forward_impl`) |
| the definitions the copy lost | `vllm_ascend/ops/fused_moe/fused_moe.py:612` (gate_up), `:616` (act_fn → down_proj, plus the Qwen3-Next `expert_gate` sigmoid) |
| the two-tuple unpack vs a bare-tensor return | `:290` (`shared_out, fused_out = AscendFusedMoE310.forward(…)`) against `:259` (`return routed_out`) |
| registry routing | `vllm_ascend/utils.py:663,686` |
| no test constructs it | `grep -rln "AscendSharedFusedMoE" tests/` → empty |

Answer given, and the shape to reuse: Q1 "supported?" → quote the release's own 310P
closing note (`docs/source/tutorials/hardwares/310p.md:207-209` at v0.18.0: early access
Qwen3-MoE/VL, **preview** Qwen3.5/3.6 "follow #7394"); Q3 "misconfiguration?" → no,
source-verified downstream bug; Q2 "timeline?" → not answerable here (the referenced
issue is future/absent on a `one_commit_git` snapshot) — say that instead of guessing.
Fix ladder: add the two helpers from the A2 sibling (or collapse to
`return self._shared_experts(hidden_states)`), then expect the tuple contract, then
sanity-check the shared-expert contribution — the A2 class re-creates its runner *after*
assigning `_shared_experts` ("The parent class created the runner before
self._shared_experts was set", `ops/fused_moe/fused_moe.py:594-596`) while the 310P
`__init__` assigns it *after* `AscendFusedMoE310.__init__` has already passed
`shared_experts` to `_init_runner`.

## Traps

- **Do not read the upstream parent literally when the snapshot's vllm pin postdates the
  vllm-ascend pin.** At #12856 the pinned vllm was v0.24.0 while vllm-ascend was
  v0.18.0, and `vllm/model_executor/layers/fused_moe/shared_fused_moe.py` **does not
  exist** there (the layer was refactored into `fused_moe/layer.py` + `experts/`), so no
  amount of grepping upstream can tell you what the parent provided at the reporter's
  vllm 0.18.0. The reporter's own `AttributeError` *is* the proof that the parent did not
  provide the name; say that, and mark the upstream-side question unverifiable.
- **A 310P capability verdict and a 310P code bug can coexist.** Answer both, in that
  order, but do not let "preview family" replace the traceback analysis: the traceback
  here names a class that is broken for *any* model with a gated shared expert.
- The A2 helper body is 3 lines and carries a Qwen3-Next-specific `expert_gate` sigmoid.
  Copying it verbatim is the right rung; whether a Qwen3.6 checkpoint needs that gate is
  a separate, unverified claim — rank it as "check the shared-expert term numerically",
  not as a fact.
