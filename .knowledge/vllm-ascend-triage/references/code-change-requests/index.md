# Code-change requests: the issue asks for a source change, not a diagnosis

## Scope

Reports whose deliverable is an **edit to the tree**, not a fault: "remove
`<ENV_VAR>` before <release> final release", "delete the temporary switch added
in PR #N", "deprecate / rename <flag|config key>", "this TODO should be done
before the release". The body is often only two links — the PR and one review
comment — with no environment dump, no log and no reproduction.

A second shape lands here with a traceback attached: an inference-only optimization
that is correct for serving but breaks **live weight reload** in an RL rollout
worker — "gate `dispose_layer(kv_b_proj)` behind a capability flag", usually with
`AssertionError: param_data.shape == loaded_weight.shape` and `load_format: dummy`.
The deliverable is a change plan plus the one check that proves the emptied
parameter, not a load-phase diagnosis (`live-weight-reload-vs-disposal.md`); an RL
weight sync that fails **inside a collective** is
`../hccl-and-hardware/weight-sync-hccl-cascade.md` instead.

Read this topic when the paste contains a **task statement about code** — with no
symptom (a flag to remove) or with the traceback the proposed change is about (the
disposal shape above). It is the wrong entry when an ordinary run failed (use the
lane A/B/C table in `SKILL.md`), when the ask is "how do I turn F on"
(`../feature-enable-recipes/index.md`), or when the ask is to grade a tutorial
page (`../tutorial-verification/index.md`).

## Rules

1. **Answerability is decided before content: does the named identifier exist at
   the pinned ref?** One `rg -n '<IDENT>' <vllm-ascend root>` (plus one grep for
   the *mechanism* token when the identifier is a flag for an op — `mrope`,
   `dbo`, the kernel name). `absent at the pinned commit` is a finding, not a
   dead end: it dates the snapshot relative to the PR and turns the deliverable
   into "the removal checklist, applied on the branch that carries PR #N".
   Cite the pinned `ref`/`commit` from `SOURCES.json` and say that the PR diff,
   its comments and the release it targets are not in the workspace.
2. **Retirement is a touch-point sweep, and the doc page is generated.** The
   answer is a checklist with `path:line` anchors, not prose about the flag. The
   per-tree anchors and the worked case are in `env-flag-retirement.md`; the
   general order is definition → readers → tests/CI env sets → doc row →
   release note.
3. **Name the sanctioned gate before proposing a removal.** vllm-ascend keeps
   private bring-up switches *and* capability gates; the removal answer has to
   say which predicate survives (`CUSTOM_OP_ENABLED`, `enable_custom_op()`,
   quantization/shape predicates) so the reviewer does not read the proposal as
   "delete the gating".
4. **Check that the code the flag gates is reachable at all.** A downstream
   `*_forward_oot` addition only runs through `CustomOp`'s dispatch; an upstream
   class that defines its own `forward` bypasses it silently. See rule 7 of
   `../upstream-downstream-drift/index.md` and step 4 of
   `env-flag-retirement.md`. Without this check the answer can describe a flag
   around an op that never executes.
5. **State the behaviour change, and where a released version records it.** A
   flag whose default is `'0'` flips to always-on when the switch is deleted;
   users who set `0` lose the old path. That belongs in the release note
   (this tree's convention is one bullet per feature/removal in
   `docs/source/user_guide/release_notes.md`), and it is the sentence a reviewer
   looks for first.
6. **Validation is manual when the tree has no test for the path.** `rg -l
   '<model|op>' tests/`; if it returns nothing, say "no automated coverage at
   this ref, so the change needs a manual run" and name the run (model, one NPU,
   one image and one text-only request). Never present a doc block or a tutorial
   command as a test.
7. **The delivery gates of `SKILL.md` rule 1 apply unchanged.** A cleanup task
   has no crash to triage, which is exactly why sessions go silent: the recorded
   #1262 replay spent 15 assistant turns on bare tool calls and put its first
   message text on the last turn. First message = the absence verdict plus the
   first two checklist rows; the rest of the sweep can follow.
8. **A parameter that is emptied in place is not a shape mismatch, and its disposer
   owns the reload contract.** `x.set_(torch.empty((0,), …))` keeps the name and
   registration, so the standard weight loader asserts on the *destination*; the fix
   must gate that disposal or neutralize/rehydrate its loader, which this tree already
   does for DSA-CP layer shards. Check `model.get_parameter(<key>).shape` first, and do
   not gate the flag on `weight_transfer_config` alone — a trainer with its own
   transport calls `model.load_weights()` directly. Worked case and anchors:
   `live-weight-reload-vs-disposal.md` (#15463).
9. **When the edit is to a test that "passes" while the device faults, the edit alone
   may not remove the fault.** Shape-only asserts on an asynchronous NPU hold whether
   the kernel computed anything or aborted, so such a test is not coverage; the parser
   is `ASCEND_LAUNCH_BLOCKING=1`, and the replacement is a numerical assert against the
   tree's own reference implementation. Then trace the proposed input change against
   the line the fault passes through: aligning a shape or a length does **not** remove
   a null-optional read in the op, so ranking it as the cheapest rung overstates it.
   Worked case, with the reference-implementation layout trap:
   `../custom-ops-runtime-faults/fla-chunk-varlen-null-chunk-offsets.md`.
