# A highlights bullet is not the tree: three co-signs, then a dated answer

## Scope

A user quotes the release's own **highlights line** — "Support EAGLE-3 for
speculative decoding. [#1032]" under `## v0.9.1rc1` — and reports that the feature
is missing from the later rcs, asking whether the final release will carry it.
Siblings: "rc2 dropped feature X", "the notes say it is supported but it raises",
"does release Y support Z at all". No log, no traceback: the graded object is a
claim about one release line, so the deliverable is a verdict at a named commit
plus a date.

Distinct from a maintainer bulletin (`index.md` scope, `bulletin-claim-check.md`):
there the claim is a risk statement about tracked bugs; here it is a capability
claim about a feature the reporter wants to use.

## Procedure: four reads, in this order

1. **The consumer at the pin, keyed on the method/flag literal.** Grep the literal
   the user would pass (`eagle`, `eagle3`, `suffix`, the flag name), not the topic
   word: `rg -n '"<literal>"' <pin>/vllm_ascend/`. Read the dispatch chain that
   assigns the object under that literal. Two outcomes are both statements by the
   project at that ref: a clean refusal (`raise NotImplementedError("<x> ...
   doesn't work on vllm-ascend currently")`), and **an absence** — a literal that
   matches none of the `elif` arms leaves the result variable unbound, so the user
   gets `UnboundLocalError` instead of a diagnosable message. Say which one the
   reporter will see; it changes what they should not bother trying.
2. **The tree's own test for exactly that feature.** `rg -n -i '<feature>'
   <pin>/tests/`; a `pytest.skip("Not current support for the test.")` on the
   parametrization that names the feature is the project's statement at that ref
   and is citable as such (SKILL.md step 5, #2071).
3. **The CI wiring.** `rg -n '<test file name>' <pin>/.github/workflows/` — a
   commented-out invocation beside a live sibling is a **coverage gap**, quotable
   as wiring, never as a refusal (the refusal is step 1's line).
4. **Upstream vs downstream.** Check the matching vllm pin accepts the same
   literal (`vllm/config.py` `SpeculativeMethod = Literal[...]`, the op/proposer
   module, the model file in `model_executor/models/`). If upstream has it and the
   plugin refuses it, the gate is downstream and "wait for the plugin" is the
   honest answer; if upstream lacks it too, say so and stop.

Two or three of these agreeing is a verdict. Keep the exact `path:line` for each:
they are the reply's evidence table.

## Answer shape for "will the official release have it?"

- **Do not promise.** The snapshot usually stops at the rc the report names, so
  anything landed after that commit is invisible. Say that, then hand the reporter
  the one command that dates it themselves:
  `git log --oneline <rc-tag>..<release-tag> -- <owning file>` — a touching commit
  is the answer, an empty range is the answer too.
- Name the tracked feature request (e.g. "Implement Eagle3 Acceleration") as the
  place to ask instead of a new issue.
- Give the **fallback that exists at that ref**, with its own limits: a different
  engine generation (`VLLM_USE_V1=0`) when the older path holds the feature and its
  e2e test carries no skip — plus the constraint the code states (e.g. a draft
  TP>1 refusal) — rather than a config guess that only hits the unbound-variable
  arm.
- Scope block: pin commits for both repos, and what "verified" means here (source
  at a commit, not a run).

## Why rc1 had it and rc2/rc3 did not

The later rcs are cut from the release branch, and the pin's own notes usually say
so for some other feature — at v0.9.1rc1 `release_notes.md:170`: Atlas 300I support
"**will NOT be included in v0.9.1-dev branch** … We will improve this from 0.9.2rc1
and later", repeated in the rc2 section at `:135`. Quote that sibling bullet as the
documented branch-carry mechanism, and label the rc1-working claim as the
reporter's observation: a single-commit export has no history, so no local diff can
show what the rc1 tree contained.

## Worked case: #2562 ("v0.9.1rc3 does not support Eagle3")

Anchors verified at vllm-ascend `763ed69c1c92` (v0.9.1rc3) + vllm `b6553be1bc75`
(v0.9.1):

| Anchor | What it establishes |
|---|---|
| `vllm_ascend/worker/model_runner_v1.py:1366-1368` | `elif … method == "eagle": raise NotImplementedError("eagle method for spec decode doesn't work on vllm-ascend currently")` — the V1 draft-token step refuses by name |
| `vllm_ascend/worker/model_runner_v1.py:1359-1378` | the chain covers `ngram` / `eagle` / `deepseek_mtp` only, so `method: "eagle3"` falls through to `return spec_token_ids` unassigned |
| `vllm_ascend/worker/model_runner_v1.py:234` | an `EagleProposer` *is* constructed, so the scaffolding exists and produces nothing (do not read it as support) |
| `tests/long_term/spec_decode_v1/test_v1_spec_decode.py:119` with `:107` | `pytest.skip("Not current support for the test.")` on the `use_eagle3` parametrization |
| `.github/workflows/vllm_ascend_test_long_term.yaml:102` vs `:100` | the v1 spec-decode file is commented out while v1 MTP runs |
| `docs/source/user_guide/release_notes.md:171` / `:3-31` / `:33-136` | rc1 claims EAGLE-3 (#1032); neither the rc2 nor the rc3 section lists an eagle PR |
| `vllm_ascend/patch/worker/patch_common/patch_eagle.py:70` | leftover of #1032: `EagleProposer.prepare_inputs` rewritten from the triton kernel to torch (rationale at `vllm_ascend/patch/__init__.py:139-150`) |
| `vllm_ascend/patch/worker/patch_common/patch_spec_decode_worker.py:91-99` | eagle handling that *does* exist lives on the **V0** `SpecDecodeWorker` path (lm_head load; `TP > 1` refused) |
| `tests/long_term/spec_decode_v0/e2e/test_eagle_correctness.py` (no skip) | V0 eagle is the covered path at that ref, hence the fallback |
| vllm `b6553be1bc75`: `vllm/config.py:2340`, `v1/spec_decode/eagle.py:100,116`, `model_executor/models/llama_eagle3.py` | upstream accepts `method: "eagle3"` at the matching pin, so the blocker is downstream only |

Reply that landed (and the shape to reuse): verdict at the commit, the three
co-signs as a table, the branch-exclusion precedent, the V0 fallback with its
`TP==1` limit, the `git log <rc>..<release>` dating command, and an explicit
"cannot be confirmed from this snapshot".
